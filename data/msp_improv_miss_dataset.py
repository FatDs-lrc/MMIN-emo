import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy
import random


class MSPimprovMissDataset(BaseDataset):
        
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        self.set_name = set_name
        data_path = "/data6/lrc/MSP-IMPROV_feature/{}/miss/{}/"

        label_path = "/data6/lrc/MSP-IMPROV_feature/target/miss/{}/"
        if set_name != 'trn':
            self.miss_type = np.load(label_path.format(cvNo) + f"{set_name}_type.npy")

        self.acoustic_data = np.load(data_path.format('audio', cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format('text', cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format('face', cvNo) + f"{set_name}.npy")
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)

        print(f"MSP-IMPROV_Miss dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
            
        acoustic = torch.from_numpy(self.acoustic_data[index]).float()
        lexical = torch.from_numpy(self.lexical_data[index]).float()
        visual = torch.from_numpy(self.visual_data[index]).float()
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        
        ans = {
            'acoustic': acoustic, 
            'lexical': lexical,
            'visual': visual,
            'label': label,
            'index': index,
        }

        if self.set_name != 'trn':
            ans['miss_type'] = self.miss_type[index]

        return ans
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
    
    opt = test()
    a = MSPimprovMissDataset(opt, 'val')
    data = next(iter(a))
    for k, v in data.items():
        if len(v.shape) == 0:
            print(k, v)
        else:
            print(k, v.shape)