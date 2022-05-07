
import torch
import models
import time
import argparse
from models.cnn import CNN

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperspectral Image Super-resolution')
    
    parser.add_argument('--conv2d', type=bool, default=False, help='Whether or not use 3D Convolutions')

    opt = parser.parse_args()
    
    N = 500
    input = torch.randn(1,16,64,64)
    input.cuda()

    start = time.time()

    with torch.no_grad():
        for i in range(N):
            model = CNN(in_channel=16, out_channel=500)
            result = model(input)
            # result.cpu()

    end = time.time()

    print('{:.4f}'.format((end-start) / N * 1000))
