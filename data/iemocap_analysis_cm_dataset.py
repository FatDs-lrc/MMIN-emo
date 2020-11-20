import os
import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapAnalysisCMDataset(BaseDataset):
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--data_type', type=str, 
                            choices=['L', 'L_recon', 'A', 'A_recon', 'V', 'V_recon'], \
                            help='analysis data type')
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        self.set_name = set_name if set_name != 'tst' else 'test'
        # load feats
        if 'L' in opt.data_type:
            data_root = f'checkpoints/analysis_recon_avz/{cvNo}'
        elif 'A' in opt.data_type:
            data_root = f'checkpoints/analysis_recon_zvl/{cvNo}'
        elif 'V' in opt.data_type:
            data_root = f'checkpoints/analysis_recon_azl/{cvNo}'
        else:
            raise ValueError('data type should contain AVL')

        if not 'recon' in opt.data_type:
            self.data = np.load(os.path.join(data_root, f'{opt.data_type}_feat_{self.set_name}.npy'))
        else:
            self.data = np.load(os.path.join(data_root, f'recon_{opt.data_type.split("_")[0]}_feat_{self.set_name}.npy'))

        self.label = np.load(os.path.join(data_root, f'{self.set_name}_label.npy')).astype(np.int)
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
    
        feat = self.data[index]
        label = self.label[index]
        ans = {
            'feat': torch.from_numpy(feat).float(),
            'label': torch.tensor(label)
        }
        return ans
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        acoustic_ft_type = 'A'
        lexical_ft_type = 'L'
        visual_ft_type = 'V'
    
    opt = test()
    a = IemocapAnalysisLDataset(opt, set_name='val')
    data = next(iter(a))
    for key, value in data.items():
        print(key, value)