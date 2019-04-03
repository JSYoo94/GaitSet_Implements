import torch
import torch.nn as nn

from network.network_block import *

class ConvLayer(nn.Module):
    
    " Feature Extraction Convolutional & Set Pooling Layer "
    
    def __init__(self):
        super(ConvLayer, self).__init__()
        
        self.conv1 = FrameBlock(1, 64, [5, 3], [2, 1], True)
        self.conv2 = FrameBlock(64, 128, [3, 3], [1, 1], True)
        self.conv3 = FrameBlock(128, 256, [3, 3], [1, 1], False)
        
        self.set_conv1 = SetBlock(64, 128, [3, 3], [1, 1], True)
        self.set_conv2 = SetBlock(128, 256, [3, 3], [1, 1], False)
        
    def forward(self, input):
        
        output = self.conv1(input)
        gl_output = self.set_conv1(torch.max(output, 1)[0])
        
        output = self.conv2(output)
        gl_output = self.set_conv2(gl_output + torch.max(output, 1)[0])
        
        output = self.conv3(output)
        output = torch.max(output, 1)[0]
        
        gl_output = gl_output + output
        
        return output, gl_output
    
class HPMLayer(nn.Module):
    
    " Horizontal Pyramid Mapping Layer "
    
    def __init__(self, num_bins):
        super(HPMLayer, self).__init__()
        
        self.num_bins = num_bins
        
        self.fc_bin = nn.ParameterList([
                nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.zeros((sum(num_bins) * 2, 256, 512))))])
        
    def forward(self, lc_feature, gl_feature):
        
        feature = list()
        n, c, h, w = gl_feature.size()
        
        for num_bin in self.num_bins:
            
            latent_z = lc_feature.view(n, c, num_bin, -1)
            latent_z = latent_z.mean(3) + latent_z.max(3)[0]
            feature.append(latent_z)
            
            latent_z = gl_feature.view(n, c, num_bin, -1)
            latent_z = latent_z.mean(3) + latent_z.max(3)[0]
            feature.append(latent_z)
            
        feature = torch.cat(feature, 2).permute(2, 0, 1)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2)
        
        return feature