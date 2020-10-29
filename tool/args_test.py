import argparse

parser = argparse.ArgumentParser()

parser.add_argument("echo", help="echo the string you use here")
parser.add_argument("square", help="display a square of a given number",
                    type=int)
parser.add_argument('-g', '--gpus', default=1, type=int,help='number of gpus per node')

args = parser.parse_args()

print(args.square ** 2)
print('gpus:'  + str(args.gpus))