import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class Iemocap6emoDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.acoustic_ft_type
        lexical_ft_type = opt.lexical_ft_type
        visual_ft_type = opt.visual_ft_type
        data_path = "/data3/lrc/Iemocap_6emo/feature/{}/{}/"
        label_path = "/data3/lrc/Iemocap_6emo/target/{}/"

        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format(visual_ft_type, cvNo) + f"{set_name}.npy")
        # mask for text feature
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index][0]
        # a_mask = torch.from_numpy(self.a_mask[index])
        return {
            'acoustic': acoustic, 
            'lexical': lexical,
            'visual': visual,
            'label': label,
            'index': index,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        acoustic_ft_type = 'IS10'
        visual_ft_type = 'denseface'
        lexical_ft_type = 'bert'
    
    opt = test()
    a = Iemocap6emoDataset(opt, 'val')
    data = next(iter(a))
    for k, v in data.items():
        print(k, v) 