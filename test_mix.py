import os
import time
import numpy as np
from opts.test_opts import TestOptions
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score

def test(opt, phase, part=None):
    total_acc = []
    total_wap = []
    total_wf1 = []
    total_uar = []
    total_f1 = []
    root = os.path.join(opt.checkpoints_dir, opt.name)
    for cv in range(1, 11):
        cv_dir = os.path.join(root, str(cv))
        cur_cv_pred = np.load(os.path.join(cv_dir, '{}_{}_pred.npy'.format(phase, part)))
        cur_cv_label = np.load(os.path.join(cv_dir, '{}_{}_label.npy'.format(phase, part)))
        acc = accuracy_score(cur_cv_label, cur_cv_pred)
        wap = precision_score(cur_cv_label, cur_cv_pred, average='weighted')
        wf1 = f1_score(cur_cv_label, cur_cv_pred, average='weighted')
        uar = recall_score(cur_cv_label, cur_cv_pred, average='macro')
        f1 = f1_score(cur_cv_label, cur_cv_pred, average='macro')
        total_acc.append(acc)
        total_wap.append(wap)
        total_wf1.append(wf1)
        total_uar.append(uar)
        total_f1.append(f1)
        if not opt.simple:
            print('{:.4f}\t{:.4f}\t{:.4f}'.format(acc, uar, f1))
    
    acc = '{:.4f}±{:.4f}'.format(float(np.mean(total_acc)), float(np.std(total_acc)))
    wap = '{:.4f}±{:.4f}'.format(float(np.mean(total_wap)), float(np.std(total_wap)))
    wf1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_wf1)), float(np.std(total_wf1)))
    uar = '{:.4f}±{:.4f}'.format(float(np.mean(total_uar)), float(np.std(total_uar)))
    f1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_f1)), float(np.std(total_f1)))
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(float(np.mean(total_acc)), float(np.mean(total_uar)), float(np.mean(total_f1))))
    if not opt.simple:
        print('{:.4f}\t{:.4f}\t{:.4f}'.format(float(np.std(total_acc)), float(np.std(total_uar)), float(np.std(total_f1))))
        print('%s result acc %s uar %s f1 %s' % (phase, acc, uar, f1))
    # print('%s result:\nacc %s wap %s wf1 %s uar %s f1 %s' % (phase.upper(), acc, wap, wf1, uar, f1))

if __name__ == '__main__':
    opt = TestOptions().parse()    # get training options
    print(opt.name)
    # for part in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
    #     print(part + ':')
    #     print('----------------------')
    #     test(opt, 'val', part)
    #     # print('\n')
    #     print('----------------------')
    #     # print('\n')
    #     test(opt, 'test', part)
    #     print('----------------------')
    print('VAL:')
    for part in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        print(part + ':')
        test(opt, 'val', part)
    
    print("TST:")
    for part in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        print(part + ':')
        test(opt, 'test', part)
        