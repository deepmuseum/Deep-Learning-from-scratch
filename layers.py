import numpy as np
from functions import conv2d

class Conv2D:
    def __init__(self,in_channels,out_channels, kernel_size,stride=1,padding=0,padding_mode='zeros'):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.padding_mode=padding_mode
        self.bias = np.zeros(out_channels//in_channels)
        self.weight=np.random.normal(0,1,(out_channels//in_channels,*kernel_size,in_channels))

    def forward(self,input):
        return conv2d(input, self.weight, self.bias, self.stride, self.padding)


