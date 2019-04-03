import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    
    def __init__(self, batch_size, margin, is_hard=True):
        super(TripletLoss, self).__init__()
        
        self.batch_size = batch_size
        self.margin = margin
        self.is_hard = is_hard
        
    def forward(self, feature, label):
        """
            [ input ]
            
                [ feature ] : [ n, m, d ]
                [ label ] : [ n, m ]
            
            - n : batch_size
            - m : 62 ( scale is 5, so 1 + 2 + 4 + 8 + 16 ) * 2 ( local and global )
            - d : hidden_dim
        """
        n, m, d = feature.size()
        
        """ 
            [ hard ]
            
                [ hp_mask ] : hard positive mask
                [ hn_mask ] : hard negative mask
        """
        hp_mask = ( label.unsqueeze(1) == label.unsqueeze(2) ).byte().view(-1)
        hn_mask = ( label.unsqueeze(1) != label.unsqueeze(2) ).byte().view(-1)
        
        dist = self.batch_dist(feature)
        dist_mean = dist.mean(1).mean(1)
        dist = dist.view(-1)
        
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
        
        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        
        return hard_loss_metric_mean, dist_mean
        
    def batch_dist(self, input):
        tmp = torch.sum( input**2 , 2 )
        dist = tmp.unsqueeze(2) + tmp.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul( input, input.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        
        return dist