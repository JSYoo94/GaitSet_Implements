import torch.utils.data
import numpy as np

class OU_ISIR(torch.utils.data.Dataset):
    
    def __init__(self, seq_path, label, view):
        
        self.seq_path = seq_path
        self.label = label
        self.view = view
        
        self.data_size = len(self.label)
        
        self.set_label = set(self.label)
        self.set_view = set(self.view)
        
        self.index_dict = {}
        
        for l in self.set_label:
            self.index_dict[l] = [i for i, _label in enumerate(self.label) if l == _label]

        
    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)
            
    def load_data(self, index):
        return self.__getitem__(index)
    
    def __loader__(self, path):                    
        return np.load(path)    
    
    def __getitem__(self, index):
        
        data = self.__loader__( self.seq_path[index]) / 255.0

        return data, self.view[index], self.label[index]
    
    def __len__(self):
        return len(self.label)