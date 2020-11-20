import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score

def test(f, phase):
    total_acc = []
    total_wap = []
    total_wf1 = []
    total_uar = []
    total_f1 = []
    root = os.path.join('./checkpoints', name)
    for cv in range(1, 11):
        cv_dir = os.path.join(root, str(cv))
        cur_cv_pred = np.load(os.path.join(cv_dir, '{}_pred.npy'.format(phase)))
        cur_cv_label = np.load(os.path.join(cv_dir, '{}_label.npy'.format(phase)))
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
        f.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(acc, uar, f1))
    
    acc = '{:.4f}±{:.4f}'.format(float(np.mean(total_acc)), float(np.std(total_acc)))
    wap = '{:.4f}±{:.4f}'.format(float(np.mean(total_wap)), float(np.std(total_wap)))
    wf1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_wf1)), float(np.std(total_wf1)))
    uar = '{:.4f}±{:.4f}'.format(float(np.mean(total_uar)), float(np.std(total_uar)))
    f1 = '{:.4f}±{:.4f}'.format(float(np.mean(total_f1)), float(np.std(total_f1)))
    f.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(float(np.mean(total_acc)), float(np.mean(total_uar)), float(np.mean(total_f1))))
    f.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(float(np.std(total_acc)), float(np.std(total_uar)), float(np.std(total_f1))))
    # print('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
    f.write('%s result:\nacc %s wap %s wf1 %s uar %s f1 %s\n' % (phase.upper(), acc, wap, wf1, uar, f1))

if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    out_path = 'today_tasks/results/{}'.format(name)
    f = open(out_path, 'w')
    f.write(name + "\n")
    test(f, 'val')
    f.write('\n----------------------------------------\n')
    test(f, 'test')
    f.close()