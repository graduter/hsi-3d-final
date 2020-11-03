from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import time
import os
import copy
import math
import pickle
import p2_resnet3D_a_1 as resnet3D_a_1
import p2_resnet3D_a_2 as resnet3D_a_2
import p2_resnet3D_a_3 as resnet3D_a_3
import p2_resnet3D_a_4 as resnet3D_a_4
import p2_resnet3D_a_5 as resnet3D_a_5
import p2_resnet3D_a_6 as resnet3D_a_6
import p2_resnet3D_a_7 as resnet3D_a_7
import p2_resnet3D_a_8 as resnet3D_a_8
import argparse

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print(torch.version.cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '/home/data/zfl/data'
results_base_model = '/home/data/zfl/results/results_a.txt'
num_output = 1
batch_size = 8
num_epochs = 70

model_name_list = ['p2_resnet3D_a_1','p2_resnet3D_a_2','p2_resnet3D_a_3','p2_resnet3D_a_4','p2_resnet3D_a_5','p2_resnet3D_a_6','p2_resnet3D_a_7','p2_resnet3D_a_8']
learning_rate_list_large = [0.001]
weight_decay_list = [0.01]
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
    # for i in range(len(HSI_datasets['train'])):
    #     sample = HSI_datasets['train'][i]
    #     print(i, sample['image'].shape, sample['label'])                                 # HSI_datasets 循环拿数据方式
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(HSI_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': torch.utils.data.DataLoader(HSI_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 'test': torch.utils.data.DataLoader(HSI_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }                                                                                  # 对train、val、test，将数据集构建成dataloader
    return dataloaders_dict
    # for batch in dataloaders_dict['train']:                                              # or: for i,batch in enumerate(dataloaders_dict['train'])
    #     print(batch['image'].shape, batch['label'].shape)                                # dataloader的batch循环方式
    # for inputs, labels in dataloaders_dict['test']:                                      # 之前RGB图像dataloader的batch循环方式
    #     print(inputs.shape)
    #     print(labels)




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
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    val_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())                    # state_dict（状态字典）：获取模型当前的参数,以一个有序字典形式返回。这个有序字典中,key 是各层参数名,value 就是参数。
    best_loss = 10000000000000000000000.0
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
def test_model(model, dataloaders):
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
parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-data_dir', '--data_dir', default='/hsi-data/train_resized_merged', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-result_dir', '--result_dir', default='', type=str, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('-model_name', '--model_name', default='', type=str, metavar='N',
#                     help='model name')
args = parser.parse_args()
data_dir = args.data_dir
results_base_model = args.result_dir
# model_name_list = args.model_name
num_epochs=args.epochs

print(model_name_list)


# run and save results
dataloaders_dict = get_data()
criterion = nn.MSELoss()                                                                     # nn.CrossEntropyLoss()

for model_name in model_name_list:
    for learning_rate in learning_rate_list_large:
        for weight_decay in weight_decay_list:
            if model_name == 'p2_resnet3D_a_1':
                net = resnet3D_a_1.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_2':
                net = resnet3D_a_2.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_3':
                net = resnet3D_a_3.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_4':
                net = resnet3D_a_4.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_5':
                net = resnet3D_a_5.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_6':
                net = resnet3D_a_6.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_7':
                net = resnet3D_a_7.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            if model_name == 'p2_resnet3D_a_8':
                net = resnet3D_a_8.generate_model(model_depth=18, n_classes=num_output, n_input_channels=1)
            print(net)
            net = torch.nn.DataParallel(net, device_ids = device_ids)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            net = net.to(device)
            net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("Total number of trainable parameters of " + model_name + " is: {}".format(net_total_params))
            print("Initial learning rate is: {}".format(learning_rate))
            print("weight decay is: {}".format(weight_decay))
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            net, val_loss_history, train_loss_history = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
            val_mse,val_r2,val_y_pred,val_y_true,test_mse,test_r2,test_y_pred,test_y_true = test_model(net, dataloaders_dict)
            results = {}
            results['val_loss_history'] = [np.round(h,4) for h in val_loss_history]
            results['train_loss_history'] = [np.round(h, 4) for h in train_loss_history]
            results['val_mse'] = np.round(val_mse,4)
            results['val_r2'] = np.round(val_r2,4)
            results['val_y_pred'] = [np.round(h,4) for h in val_y_pred]
            results['val_y_true'] = [np.round(h, 4) for h in val_y_true]
            results['test_mse'] = np.round(test_mse,4)
            results['test_r2'] = np.round(test_r2,4)
            results['test_y_pred'] = [np.round(h, 4) for h in test_y_pred]
            results['test_y_true'] = [np.round(h, 4) for h in test_y_true]
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

