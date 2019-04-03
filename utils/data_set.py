import os, cv2
import torch.utils.data
import numpy as np

class OU_ISIR(torch.utils.data.Dataset):
    
    def __init__(self, seq_path, label, view):
        
        self.seq_path = seq_path
        self.label = label
        self.view = view
        
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        
        self.set_frame = [None] * self.data_size
        self.set_label = set(self.label)
        self.set_view = set(self.view)
        
    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)
            
    def load_data(self, index):
        return self.__getitem__(index)
    
    def __loader__(self, path):                    
        return np.load(path)    
    
    def __getitem__(self, index):
        
        if self.data[index] is None:
            self.data[index] = self.__loader__( self.seq_path[index] ) / 255.0

        return self.data[index], self.view[index], self.label[index]
    
    def __len__(self):
        return len(self.label)