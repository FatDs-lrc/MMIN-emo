import os, sys
import time
import json
import numpy as np
from tqdm import tqdm
from data import create_dataset, create_trn_val_tst_dataset
from models.utils.config import OptConfig
from models import create_model
from utils.logger import get_logger
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval_latent(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_latent = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        latent = model.latent.detach().cpu().numpy()
        label = data['label']
        total_latent.append(latent)
        total_label.append(label)
    
    # calculate metrics
    total_latent = np.concatenate(total_latent)
    total_label = np.concatenate(total_label)
    
    # save test results
    np.save(os.path.join(save_dir, '{}_latent.npy'.format(phase)), total_latent)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    teacher_path = 'checkpoints/new_cra_AE256,128,64_blocks5_run2'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [4]
    opt.serial_batches = True
    # opt.dataset_mode = "iemocap_miss"
    # setattr(opt, 'miss_num', 'mix')
    # opt.dataset_mode = 'iemocap_10fold'
    # setattr(opt, 'miss_num', 'mix')
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)

        save_root = 'checkpoints/analysis_latent/' + str(cv)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    
        eval_latent(model, val_dataset, save_root, phase='val')
        eval_latent(model, tst_dataset, save_root, phase='test')
       
        