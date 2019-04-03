import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, bias=False, **kwargs)
        
    def forward(self, input):
        input = self.conv(input)
        return F.leaky_relu(input, inplace=True)
    
    
    
class FrameBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, kernel_sizes, paddings, pooling=False):
        super(FrameBlock, self).__init__()
        
        self.pooling = pooling
        
        self.conv1 = ConvBlock(channels_in, channels_out, kernel_sizes[0], padding=paddings[0])
        self.conv2 = ConvBlock(channels_out, channels_out, kernel_sizes[1], padding=paddings[1])
        
        if pooling:
            self.pool = nn.MaxPool2d(2)
            
    def forward(self, input):
        n, s, c, h, w = input.size()
        input = self.conv1(input.view(-1, c, h, w))
        input = self.conv2(input)
        
        if self.pooling:
            input = self.pool(input)
        
        _, c, h, w = input.size()
        return input.view(n, s, c, h, w)  
    
    
    
class SetBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, kernel_sizes, paddings, pooling=False):
        super(SetBlock, self).__init__()
        
        self.pooling = pooling
        
        self.set_conv1 = ConvBlock(channels_in, channels_out, kernel_sizes[0], padding=paddings[0])
        self.set_conv2 = ConvBlock(channels_out, channels_out, kernel_sizes[1], padding=paddings[1])
        
        if pooling:
            self.pool = nn.MaxPool2d(2)
            
    def forward(self, input):
        output = self.set_conv1(input)
        output = self.set_conv2(output)
        
        if self.pooling:
            output = self.pool(output)
            
        return output