import random
import torch.utils.data as tordata

from datetime import datetime

class TripletSampler( tordata.sampler.Sampler ):
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        while(True):
            sample_indices = []
            label_list = random.sample(list(self.dataset.set_label), self.batch_size[0])
            
            for _label in label_list:
                
                s_idx = random.choices(self.dataset.index_dict[_label], k=self.batch_size[1])
                
                sample_indices += s_idx
             
            yield sample_indices
                                       
    def __len__(self):
        return self.dataset.data_size
    
def collate_fnn( batch ):
    
    batch_size = len(batch)
    
    seqs = [batch[i][0] for i in range(batch_size)]
    view = [batch[i][1] for i in range(batch_size)]
    label = [batch[i][2] for i in range(batch_size)]
    
    def select_frame(index):
        sample = seqs[index]
        
        frame_num = 16              
        if len(sample) < frame_num:
            print(view[index], label[index])
            sampled_num = 0  
            frame_list = []
            while(sampled_num != frame_num):
                if ( frame_num - sampled_num ) / len(sample) < 1:
                    frame_list += random.choices([i for i in range(len(sample))], k = frame_num - sampled_num)
                    sampled_num += frame_num - sampled_num
                else:
                    frame_list += random.choices([i for i in range(len(sample))], k = len(sample))
                    sampled_num += len(sample)
        
        else:
            frame_list = random.choices([ i for i in range(len(sample))], k = frame_num)
            
        _ = [ sample[i] for i in frame_list ]
        
        return _
    
    seqs = list(map(select_frame, range(len(seqs))))
    
    return seqs, view, label