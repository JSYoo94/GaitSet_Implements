import random
import torch.utils.data as tordata

class TripletSampler( tordata.sampler.Sampler ):
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        while(True):
            sample_indices = []
            label_list = random.sample(list(self.dataset.set_label), self.batch_size[0])
            
            for _label in label_list:
                
                s_idx = random.choices([i for i, __label in enumerate(self.dataset.label) if _label == __label], k = self.batch_size[1])
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
        
        if len(sample) < 16:
            print(label[index], view[index])
            
        frame_list = random.choices([ i for i in range(len(sample))], k = 16)
        _ = [ sample[i] for i in frame_list ]
        
        return _
    
    seqs = list(map(select_frame, range(len(seqs))))
    
    return seqs, view, label