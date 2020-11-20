import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapDeepMissDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.acoustic_ft_type
        lexical_ft_type = opt.lexical_ft_type
        visual_ft_type = opt.visual_ft_type
        # data_path = "/data3/lrc/Iemocap_feature/ef_pretrained/{}/{}/" # 
        # label_path = "/data3/lrc/Iemocap_feature/ef_pretrained/target/{}/"
        data_path = '/data2/lrc/Iemocap_feature/multi_fusion_reps/{}/{}/'
        label_path = '/data2/lrc/Iemocap_feature/multi_fusion_reps/target/{}/'
        self.set_name = set_name
        # load feats
        self.acoustic_data = np.load(data_path.format('A', cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format('L', cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format('V', cvNo) + f"{set_name}.npy")
        # load target
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
       
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        if self.set_name != 'trn':
            miss_index = {
                0: [1,0,0], # AZZ
                1: [0,1,0], # ZVZ
                2: [0,0,1], # ZZL
                3: [1,1,0], # AVZ
                4: [1,0,1], # AZL
                5: [0,1,1], # ZVL
            }[index%6]
            miss_type = {
                0: 'azz',
                1: 'zvz',
                2: 'zzl',
                3: 'avz',
                4: 'azl',
                5: 'zvl'
            }[index%6]
            index //= 6
    
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        if self.set_name != 'trn':
            acoustic = miss_index[0] * acoustic.clone()
            visual = miss_index[1] * visual.clone()
            lexical = miss_index[2] * lexical.clone()
        
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index]
        ans = {
            'acoustic': acoustic, 
            'lexical': lexical,
            'visual': visual,
            'label': label,
            'index': index,
            'int2name': int2name,
            # 'miss_index': miss_index
        }
        if self.set_name != 'trn':
            ans['miss_index'] = miss_index
            ans['miss_type'] = miss_type
        
        return ans
    
    def __len__(self):
        if self.set_name != 'trn':
            return len(self.label) * 6
        else:
            return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        acoustic_ft_type = 'A'
        lexical_ft_type = 'L'
        visual_ft_type = 'V'
    
    opt = test()
    a = IemocapDeepMissDataset(opt, set_name='tst')
    # print(next(iter(a))['miss_index'])
    # print(next(iter(a))['miss_index'])
    print(len(a))
    for d in a:
        print(d['miss_index'], d['miss_type'], d['int2name'], d['label'])
        print(torch.sum(d['acoustic']), torch.sum(d['visual']), torch.sum(d['lexical']))
        input()