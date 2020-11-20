import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapMissALDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.acoustic_ft_type
        lexical_ft_type = opt.lexical_ft_type
        self.set_name = set_name
        data_path = "/data3/lrc/Iemocap_feature/cv_level/miss_modality_AL/{}/{}/"
        label_path = "/data3/lrc/Iemocap_feature/cv_level/miss_modality_AL_target/{}/"

        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        if set_name != 'trn':
            self.miss_index = np.load(label_path.format(cvNo) + f"{set_name}_index.npy")
        
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index][0].decode()
        
        ans = {
            'acoustic': acoustic, 
            'lexical': lexical,
            'label': label,
            'index': index,
            'int2name': int2name
        }
        if self.set_name != 'trn':
            ans['miss_index'] = torch.tensor(self.miss_index[index])
        
        return ans
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        acoustic_ft_type = 'IS10'
        lexical_ft_type = 'text'
    
    opt = test()
    a = IemocapMissALDataset(opt, set_name='val')
    print(next(iter(a)))