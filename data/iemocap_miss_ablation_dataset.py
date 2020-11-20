import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapMissAblationDataset(BaseDataset):
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--miss_type', type=str, choices=['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl' ], \
                                                    help='missing modality scenario')
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
        data_path = "/data3/lrc/Iemocap_feature/cv_level/feature/{}/{}/"
        label_path = "/data3/lrc/Iemocap_feature/cv_level/target/{}/"
        self.miss_type = opt.miss_type
        assert self.miss_type in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
        if 'a' in self.miss_type:
            self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
            
        if 'v' in self.miss_type:
            self.visual_data = np.load(data_path.format(visual_ft_type, cvNo) + f"{set_name}.npy")

        if 'l' in self.miss_type:
            self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def mask2length(self, mask):
        ''' mask [total_num, seq_length, feature_size]
        '''
        _mask = np.mean(mask, axis=-1)        # [total_num, seq_length, ]
        length = np.sum(_mask, axis=-1)       # [total_num,] -> a number
        # length = np.expand_dims(length, 1)
        return length
    
    def __getitem__(self, index):
        # acoustic
        if self.miss_type[0] == "a":
            acoustic = torch.from_numpy(self.acoustic_data[index])
        else:
            acoustic = torch.zeros([1582])
        
        # visual
        if self.miss_type[1] == "v":
            visual = torch.from_numpy(self.visual_data[index])
        else:
            visual = torch.zeros([50, 342])

        # lexical
        if self.miss_type[2] == 'l':
            lexical = torch.from_numpy(self.lexical_data[index])
        else:
            lexical = torch.zeros([22, 1024])
        
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index][0].decode()
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
        miss_type = 'zvl'
        acoustic_ft_type = 'IS10'
        visual_ft_type = 'denseface'
        lexical_ft_type = 'text'
    
    opt = test()
    a = IemocapMissAblationDataset(opt, 'trn')
    print(next(iter(a)))