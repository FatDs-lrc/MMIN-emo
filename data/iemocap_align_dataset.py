import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapAlignDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = 'aligned_comparE_norm'
        lexical_ft_type = 'aligned_glove'
        visual_ft_type = 'aligned_denseface'
        data_path = "/data3/lrc/Iemocap_feature/cv_level/feature/{}/{}/"
        label_path = "/data3/lrc/Iemocap_feature/cv_level/aligned_target/{}/"

        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format(visual_ft_type, cvNo) + f"{set_name}.npy")
        # mask for text feature
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index]
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
    
    opt = test()
    a = IemocapAlignDataset(opt, 'trn')
    d = next(iter(a))
    for key, value in d.items():
        print(key, value.shape)