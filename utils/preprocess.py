import os
import cv2
import numpy as np
import argparse

from warnings import warn

def extract(input_path):
    
    " [ Extract file from tar.gz raw file ] "
    os.makedirs(output_path, exist_ok=True)
    for i, tar_gz in enumerate(os.listdir(input_path)):
        if (i % 5) == 0:
            print(' [ Extract ] _ target _ : ' + tar_gz)
        with tarfile.open( os.path.join(input_path, tar_gz), 'r:gz') as tar:
            for tarinfo in tar:
                if not os.path.exists( os.path.join(output_path, tarinfo.name) ):
                    tar.extract( tarinfo, output_path)
                    
def process(input_path):
    
    " [ Process crop_pedestrian for extracted images ] "
    
    for i, (path, subdurs, files) in enumerate(os.walk(input_path)):
        os.makedirs( path.replace('extracted', 'processed'))
        

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data_path', default='./data/raw', type=str,
                   help ='Path of raw dataset.')

option = parser.parse_args()

DATA_PATH = option.data_path
EXTRACTED = "extracted"
PROCESSED = "processed"

target_width = 64
target_height = 64