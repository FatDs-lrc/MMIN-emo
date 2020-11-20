import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy
import random


class IemocapMissDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--miss_num', type=str, default='1', choices=['1', '2', 'mix'], \
                                                    help='missing modality number, mix for all 6 combinations')
        return parser
        
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.acoustic_ft_type
        lexical_ft_type = opt.lexical_ft_type
        visual_ft_type = opt.visual_ft_type
        self.set_name = set_name
        self.miss_num = opt.miss_num
        data_path = "/data3/lrc/Iemocap_feature/cv_level/miss_modality_{}".format(self.miss_num)+"/{}/{}/"
        if self.miss_num == 'mix':
            label_path = "/data3/lrc/Iemocap_feature/cv_level/miss_modality_mix_target/{}/"
            if set_name != 'trn':
                self.miss_type = np.load(label_path.format(cvNo) + f"{set_name}_type.npy")
        else:
            label_path = "/data3/lrc/Iemocap_feature/cv_level/target/{}/"
        
        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format(visual_ft_type, cvNo) + f"{set_name}.npy")
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)

        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def mask2length(self, mask):
        ''' mask [total_num, seq_length, feature_size]
        '''
        _mask = np.mean(mask, axis=-1)        # [total_num, seq_length, ]
        length = np.sum(_mask, axis=-1)       # [total_num,] -> a number
        # length = np.expand_dims(length, 1)
        return length
    
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

        if self.set_name != 'trn' and self.miss_num == 'mix':
            ans['miss_type'] = self.miss_type[index]

        return ans
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        miss_num = 1
    
    opt = test()
    a = IemocapModalityMissingDataset(opt, 'val')
    print(next(iter(a)))