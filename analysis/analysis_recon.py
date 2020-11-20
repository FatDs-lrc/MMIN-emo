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

def eval_avz(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_L = []
    total_recon_L = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        recon_fusion = model.recon_fusion.detach().cpu().numpy()
        L = model.T_embd_L.detach().cpu().numpy()
        rL = recon_fusion[:, 128: 256]
        label = data['label']

        total_L.append(L)
        total_recon_L.append(rL)
        total_label.append(label)
    
    # calculate metrics
    total_L = np.concatenate(total_L)
    total_recon_L = np.concatenate(total_recon_L)
    total_label = np.concatenate(total_label)
    
    # save test results
    np.save(os.path.join(save_dir, 'L_feat_{}.npy'.format(phase)), total_L)
    np.save(os.path.join(save_dir, 'recon_L_feat_{}.npy'.format(phase)), total_recon_L)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)


def eval_azl(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_V = []
    total_recon_V = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        recon_fusion = model.recon_fusion.detach().cpu().numpy()
        V = model.T_embd_V.detach().cpu().numpy()
        rV = recon_fusion[:, 256:]
        label = data['label']

        total_V.append(V)
        total_recon_V.append(rV)
        total_label.append(label)
    
    # calculate metrics
    total_V = np.concatenate(total_V)
    total_recon_V = np.concatenate(total_recon_V)
    total_label = np.concatenate(total_label)
    
    # save test results
    np.save(os.path.join(save_dir, 'V_feat_{}.npy'.format(phase)), total_V)
    np.save(os.path.join(save_dir, 'recon_V_feat_{}.npy'.format(phase)), total_recon_V)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

def eval_zvl(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_A = []
    total_recon_A = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        recon_fusion = model.recon_fusion.detach().cpu().numpy()
        A = model.T_embd_A.detach().cpu().numpy()
        rA = recon_fusion[:, :128]
        label = data['label']

        total_A.append(A)
        total_recon_A.append(rA)
        total_label.append(label)
    
    # calculate metrics
    total_A = np.concatenate(total_A)
    total_recon_A = np.concatenate(total_recon_A)
    total_label = np.concatenate(total_label)
    
    # save test results
    np.save(os.path.join(save_dir, 'A_feat_{}.npy'.format(phase)), total_A)
    np.save(os.path.join(save_dir, 'recon_A_feat_{}.npy'.format(phase)), total_recon_A)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    import sys
    miss_type = sys.argv[1]
    teacher_path = 'checkpoints/new_cra_AE256,128,64_blocks5_run2'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [4]
    opt.serial_batches = True
    opt.dataset_mode = "iemocap_10fold"
    opt.model = "new_cra_ablation"
    
    # for avz
    if miss_type == 'avz':
        setattr(opt, "miss_type", "avz")
    # for azl 
    elif miss_type == 'azl':
        setattr(opt, "miss_type", "azl")
    # for vzl 
    elif miss_type == 'zvl':
        setattr(opt, "miss_type", "zvl")
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)
        # for avz
        if miss_type == 'avz':
            save_root = 'checkpoints/analysis_recon_avz/' + str(cv)
        # for azl
        elif miss_type == 'azl':
            save_root = 'checkpoints/analysis_recon_azl/' + str(cv)
        # for zvl
        elif miss_type == 'zvl':
            save_root = 'checkpoints/analysis_recon_zvl/' + str(cv)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # for avz
        if miss_type == 'avz':
            eval_avz(model, dataset, save_root, phase='trn')
            eval_avz(model, val_dataset, save_root, phase='val')
            eval_avz(model, tst_dataset, save_root, phase='test')

        # for azl
        if miss_type == 'azl':
            eval_azl(model, dataset, save_root, phase='trn')
            # eval_azl(model, val_dataset, save_root, phase='val')
            # eval_azl(model, tst_dataset, save_root, phase='test')

        # for zvl
        if miss_type == 'zvl':
            eval_zvl(model, dataset, save_root, phase='trn')
            # eval_zvl(model, val_dataset, save_root, phase='val')
            # eval_zvl(model, tst_dataset, save_root, phase='test')
        
        