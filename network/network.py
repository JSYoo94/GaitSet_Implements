import torch.nn as nn
from torch.autograd import Variable

from network.network_layer import *

class SetNet(nn.Module):
    
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.F = ConvLayer()
        self.H = HPMLayer([1, 2, 4, 8, 16])
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)
                
    def forward(self, seqs):
        
        input = torch.FloatTensor(seqs).unsqueeze(2)
        input = Variable(input).cuda()
        
        lc, gl = self.F(input)
        feature = self.H(lc, gl)
        
        return feature