import torch.nn as nn 


class Conv_ReLU_Block(nn.Module):
    def __init__(self, n_feats):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))

class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, n_feats=128):
        super(CNN, self).__init__()
        self.n_feats = n_feats
        
        self.input = nn.Conv2d(in_channel, n_feats, kernel_size=1)
        self.cnn_layer = self.make_layer(Conv_ReLU_Block, 1)
        self.output = nn.Conv2d(n_feats, out_channel, kernel_size=1)
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.n_feats))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input(x)
        x = self.cnn_layer(x)
        x = self.output(x)
        return x
    
    