import os
import numpy as np
import csv

from utils.data_set import OU_ISIR

def load_OU_ISIR(dataset_path, shuffle):
    
    train_IDs = []
    test_IDs = []

    train_seq_path = []
    test_seq_path = []
    
    train_view = []
    test_view = []
    
    train_label = []
    test_label = []
    
    """ Read Train, Test ID List """
    
    with open('./data/OU_ISIR/ID_list.csv', 'r') as ID_list:
        lines = ID_list.readlines()
        
    cooked = csv.reader(lines)
    
    for record in cooked:
        if record[0] == 'Training subject ID':
            continue
            
        if not record[0] == '':
            train_IDs.append(int(record[0]))
            
        if not record[1] == '':
            test_IDs.append(int(record[1]))
            
    """ Read View, Path """
    
    for _view in os.listdir( dataset_path ):
        view_path = os.path.join(dataset_path, _view)
               
        for _seq in os.listdir(view_path):
            
            if int(_seq[0:5]) in train_IDs:
                train_seq_path.append( os.path.join( view_path, _seq ) )
                train_view.append( _view[0:3] )
                train_label.append( int(_seq[0:5]))
            elif int(_seq[0:5]) in test_IDs:
                test_seq_path.append( os.path.join( view_path, _seq ) )
                test_view.append( _view[0:3] )
                test_label.append( int(_seq[0:5]))
                
    train_dataset = OU_ISIR( train_seq_path, train_label, train_view )
    test_dataset = OU_ISIR( test_seq_path, test_label, test_view )
    
    return train_dataset, test_dataset