from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import os
import copy
import math
import pickle
import p2_resnet3D_modified_1 as resnet3D_modified_1
import p2_resnet3D_modified_2 as resnet3D_modified_2
import p2_resnet3D_modified_3 as resnet3D_modified_3
import argparse

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print(torch.version.cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '/home/data/zfl/data'
results_base_model = '/home/data/zfl/results/results_from_scratch_modified.txt'
num_output = 1
batch_size = 8
num_epochs = 70
model_name_list = ['resnet3D_modified_2','resnet3D_modified_3','resnet3D_modified_1']                   # 'C3D','squeezenet3D','shufflenet3D','mobilenet3D'
learning_rate_list_large = [0.001,0.0001,0.00001]
learning_rate_list_small = [0.00005,0.00001,0.000005]
weight_decay_list = [0.01,0.001,0.0001]
device_ids = [0]
C3D_basic_channel_num = 16
image_depth = 140
image_width = 160
image_length = 160
# 140, 120, 160


# get data
class HSIDataset(Dataset):

    def __init__(self, excel_file, root_dir, transform=None):
        """
        Args:
            excel_file (string): Path to the excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_frame = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0]) + '_resized.hdr'
        fr = open(img_name, 'rb')
        hsi = pickle.load(fr)
        fr.close()
        # 以下是 transforms.ToTensor的功能，没有单独修改ToTensor函数，这里将ToTensor的功能嵌入数据集读取过程：hsi通道顺序由H x W x C 变换为 C x H x W，文件类型转为tensor，原本的ToTensor里还要除以255变换为0-1，这里反射率本来就是0-1，不需要
        # HSI类型是spectral.image.ImageArray，可视为numpy.ndarray，np.transpose()可对高纬np数组变换维度，与Tensor.permute()对高维tensor的转置效果相同，但torch.Transpose()只能操作2维tensor，连续使用transpose也可实现permute的效果。
        image = torch.from_numpy(hsi)                  # .transpose((2, 0, 1))

        label = self.name_frame.iloc[idx, 1]
        label = torch.from_numpy(np.array(label))      # 需先转为np.array再转为tensor

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_data():
    print("Initializing Datasets and Dataloaders...")
    HSI_datasets = {x: HSIDataset(os.path.join(data_dir, x + '_resized_merged') + '/' + x + '_labels.xlsx',
                                  os.path.join(data_dir, x + '_resized_merged'))
                    for x in ['train', 'val']}                                     # 对train、val、test，得到x和y一一对应的数据集      , 'test'
    # dataloaders_dict = {
    #     'train': torch.utils.data.DataLoader(HSI_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    #     'val': torch.utils.data.DataLoader(HSI_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #     # 'test': torch.utils.data.DataLoader(HSI_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #     }                                                                                  # 对train、val、test，将数据集构建成dataloader
    return HSI_datasets




# net
class C3D(nn.Module):
    def __init__(self,
                 n,
                 image_depth,
                 image_width,
                 image_length,
                 num_classes,
                 init_weights=True):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(n,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))
        self.group2 = nn.Sequential(
            nn.Conv3d(n, 2*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(2*n,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.group3 = nn.Sequential(
            nn.Conv3d(2*n, 4*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(4*n,track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(4*n, 4*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(4*n,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.group4 = nn.Sequential(
            nn.Conv3d(4*n, 8*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8*n,track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(8*n, 8*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8*n,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.group5 = nn.Sequential(
            nn.Conv3d(8*n, 8*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8*n,track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(8*n, 8*n, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8*n,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_depth = int(math.floor(image_depth / 16))
        last_width = int(math.ceil(image_width / 32))
        last_length = int(math.ceil(image_length / 32))
        self.fc1 = nn.Sequential(
            nn.Linear((8*n * last_depth * last_width * last_length), 64*n),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(64*n, 64*n),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(64*n, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



# train and validation
def train_model(gpu, model, dataset , num_epochs=25):
    since = time.time()
    val_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())                    # state_dict（状态字典）：获取模型当前的参数,以一个有序字典形式返回。这个有序字典中,key 是各层参数名,value 就是参数。
    best_loss = 10000000000000000000000.0

    ###############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    ###############################################################

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], num_replicas=args.world_size,
                                                                    rank=rank)

    dataloaders = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=4,
                                              sampler=train_sampler)

    ################################################################

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:                                    # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()                                             # Set model to training mode
            else:
                model.eval()                                              # Set model to evaluate mode
            running_loss = 0.0
            for i, batch in enumerate(dataloaders[phase]):
                inputs = batch['image'].unsqueeze(1).to(device)               # print('gpu上的图像大小：{}'.format(inputs.shape)
                labels = batch['label'].float().to(device)
                optimizer.zero_grad()                                     # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):            # track history if only in train
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    if phase == 'train':                                  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                print('[ {} {} ] running_loss: {:.4f}  current_batch_loss: {:.4f} '.format(epoch, i, running_loss, loss.item()))           # print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss))
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
        scheduler.step()
        print('current learning rate is: {}'.format(scheduler.get_lr()))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)                                 # load_state_dict:将 state_dict 中的参数加载到当前网络
    return model, val_loss_history, train_loss_history




# test
def val_model(model, dataloaders):

    model.eval()
    with torch.no_grad():                                                                      # 在gpu上测试，不需要计算梯度，设为eval模式，且加上不计算梯度的条件，否则内存会溢出，转到cpu上速度慢，要去除DataParallel wrap，model = model.module.cpu()
        for phase in ['val', 'train']:
            labels_tensor = torch.tensor([]).to(device)
            outputs_tensor = torch.tensor([]).to(device)
            for batch in dataloaders[phase]:
                inputs = batch['image'].unsqueeze(1).to(device)
                labels = batch['label'].float().to(device)
                outputs = model(inputs).squeeze()
                outputs_tensor = torch.cat((outputs_tensor,outputs),0)
                labels_tensor = torch.cat((labels_tensor,labels),0)

            mse = criterion(outputs_tensor, labels_tensor).cpu().detach().numpy().item()       # detach是去除grad，转为正常的tensor；numpy是转为numpy.ndarray；item是获取array的内容

            y_pred = outputs_tensor.cpu().detach().numpy()                                     # 也可用 np.array(outputs_tensor.tolist()) 转为numpy.ndarray
            y_true = labels_tensor.cpu().detach().numpy()
            SSE = np.sum((y_true - y_pred) ** 2)
            SST = np.sum((y_true - np.mean(y_true)) ** 2)
            r_squared = 1 - (float(SSE)) / SST                                                 # 回归模型用R2决定系数评估，而不是相关系数R：corr = np.corrcoef(outputs_tensor.tolist(),labels_tensor.tolist())[0,1]

            print('mse of '+phase+' is:', mse)
            print('r2 of '+phase+' is:', r_squared)

            if phase == 'val':
                val_mse = mse
                val_r2 = r_squared
                val_y_pred = y_pred
                val_y_true = y_true
            else:
                test_mse = mse
                test_r2 = r_squared
                test_y_pred = y_pred
                test_y_true = y_true
        return val_mse,val_r2,val_y_pred,val_y_true,test_mse,test_r2,test_y_pred,test_y_true


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-data_dir', '--data_dir', default='/hsi-data/train_resized_merged', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-result_dir', '--result_dir', default='', type=str, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()
data_dir = args.data_dir
results_base_model = args.result_dir

# run and save results
HSI_datasets = get_data()
                                                                    # nn.CrossEntropyLoss()

args.world_size = args.gpus * args.nodes
mp.spawn(train_model, nprocs=args.gpus, args=(args,))

for model_name in model_name_list:
    if model_name in ['resnet3D_modified_1','resnet3D_modified_2','resnet3D_modified_3']:
        learning_rate_list = learning_rate_list_large
    if model_name in ['C3D']:
        learning_rate_list = learning_rate_list_small
    for learning_rate in learning_rate_list:
        for weight_decay in weight_decay_list:
            # if model_name == 'C3D':
            #     net = C3D(n=C3D_basic_channel_num, image_depth=image_depth, image_width=image_width, image_length=image_length, num_classes=num_output)
            if model_name == 'resnet3D_modified_1':
                net = resnet3D_modified_1.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'resnet3D_modified_2':
                net = resnet3D_modified_2.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'resnet3D_modified_3':
                net = resnet3D_modified_3.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            # if model_name == 'squeezenet3D':
            #     net = squeezenet3D.get_model(version=1.1, sample_duration=image_depth, sample_size_w=image_width, sample_size_l=image_length, num_classes=num_output)
            # if model_name == 'shufflenet3D':
            #     net = shufflenet3D.get_model(groups=3, num_classes=num_output)
            # if model_name == 'mobilenet3D':
            #     net = mobilenet3D.get_model(num_classes=num_output)
            # print(net)

            # net = torch.nn.DataParallel(net, device_ids = device_ids)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Total number of trainable parameters of " + model_name + " is: {}".format(net_total_params))
            print("Initial learning rate is: {}".format(learning_rate))
            print("weight decay is: {}".format(weight_decay))

            # net, val_loss_history, train_loss_history = train_model(net, dataloaders_dict, num_epochs=num_epochs)
            net, val_loss_history, train_loss_history = mp.spawn(train_model, nprocs=args.gpus, args=(net, HSI_datasets, num_epochs,))

            # val_mse,val_r2,val_y_pred,val_y_true,test_mse,test_r2,test_y_pred,test_y_true = val_model(net, HSI_datasets)
            results = {}
            results['val_loss_history'] = [np.round(h,4) for h in val_loss_history]
            results['train_loss_history'] = [np.round(h, 4) for h in train_loss_history]
            # results['val_mse'] = np.round(val_mse,4)
            # results['val_r2'] = np.round(val_r2,4)
            # results['val_y_pred'] = [np.round(h,4) for h in val_y_pred]
            # results['val_y_true'] = [np.round(h, 4) for h in val_y_true]
            # results['test_mse'] = np.round(test_mse,4)
            # results['test_r2'] = np.round(test_r2,4)
            # results['test_y_pred'] = [np.round(h, 4) for h in test_y_pred]
            # results['test_y_true'] = [np.round(h, 4) for h in test_y_true]
            with open(results_base_model, 'a') as f:
                f.write('modified_model_' + model_name + '_learning_rate of ' + str(learning_rate) + '_weight_decay of ' + str(weight_decay) + ':' + '\n')
                for item in results:
                    f.write('\n')
                    f.write(item)
                    f.write('\n')
                    f.write(str(results[item]))
                    f.write('\n')
                f.write('\n' + '##########################################' + '\n' + '\n')


print('Finished Training')

