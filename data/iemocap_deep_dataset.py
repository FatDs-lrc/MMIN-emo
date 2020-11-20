import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapDeepDataset(BaseDataset):
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
        # data_path = '/data2/lrc/Iemocap_feature/multi_fusion_reps/{}/{}/'
        # label_path = '/data2/lrc/Iemocap_feature/multi_fusion_reps/target/{}/'

        data_path = '/data2/lrc/Iemocap_feature/early_fusion_reps/{}/{}/'
        label_path = '/data2/lrc/Iemocap_feature/early_fusion_reps/target/{}/'

        self.set_name = set_name
        # load feats
        self.acoustic_data = np.load(data_path.format('A', cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format('L', cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format('V', cvNo) + f"{set_name}.npy")
        # load target
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        # if set_name != 'trn':
        #     self.miss_index = np.load(label_path.format(cvNo) + f"{set_name}_miss_index.npy")

        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        # if index % 2 ==0:
        #     miss_index = torch.tensor([1, 0])
        # else:
        #     miss_index = torch.tensor([0, 1])
        
        # if self.set_name != 'trn':
        #     index //= 2
    
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
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
        return ans
    
    def __len__(self):
        # if self.set_name != 'trn':
        #     return len(self.label) * 2
        # else:
        #     return len(self.label)
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        acoustic_ft_type = 'A'
        lexical_ft_type = 'L'
        visual_ft_type = 'V'
    
    opt = test()
    a = IemocapDeepDataset(opt, set_name='val')
    # print(next(iter(a))['miss_index'])
    # print(next(iter(a))['miss_index'])
    print(len(a))
    for d in a:
        print(d['miss_index'])
        input()