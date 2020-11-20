import os
import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapAblationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--feat_type', type=str, default='ef', choices=['ef', 'multi', 'single'])
        parser.add_argument('--feat_modality', type=str, default='A', choices=['A', 'V', 'L'])
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        feat_type = opt.feat_type
        path_lookup = {
            'multi': '/root/lrc_git/translation/checkpoints/multi_ablation_test',
            'ef': '/root/lrc_git/translation/checkpoints/ef_ablation_test',
            'single_A': '/root/lrc_git/translation/checkpoints/EF_raw_A_run1_test', 
            'single_V': '/root/lrc_git/translation/checkpoints/EF_raw_V_run1_test', 
            'single_L': '/root/lrc_git/translation/checkpoints/EF_raw_L_run1_test', 
        }
        key = opt.feat_type
        if opt.feat_type == 'single':
            key += '_'+opt.feat_modality

        if set_name == 'tst':
            set_name = 'test'
        
        root = path_lookup[key] + str(cvNo)
        feat_name = os.path.join(root, f'{opt.feat_modality}_{set_name}_feat.npy')
        label_name = os.path.join(root, f'{set_name}_label.npy')
        self.feat = np.load(feat_name)
        self.label = np.load(label_name)
        
        print(f"IEMOCAP Ablation dataset {set_name} created with total length: {len(self)}")
    
    
    def __getitem__(self, index):
        feat = torch.from_numpy(self.feat[index])
        label = torch.tensor(self.label[index])
        return {
            'feat': feat,
            'label': label
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        feat_type = 'ef'
        feat_modality = 'L'
        
    
    opt = test()
    a = IemocapAblationDataset(opt, 'trn')
    d = next(iter(a))
    for key, value in d.items():
        print(key, value.shape)