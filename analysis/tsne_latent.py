import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def load_data(cv):
    root = 'checkpoints/analysis_latent/{}/'.format(cv)
    latent = np.load(os.path.join(root, 'test_latent.npy'))
    label = np.load(os.path.join(root, 'test_label.npy'))
    return latent, label

def plot(cv):
    file_dir = os.path.dirname(__file__)
    save_root = os.path.join(file_dir, 'tsne', 'analysis_latent')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    latent, label = load_data(cv)
    print("Finish loading data: ")
    print("Latent:", latent.shape)
    print("Label:", label.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(latent)
    # init graph
    plt.figure(figsize=(5, 5))
    # define colors
    colors_label = ['red', 'blue', 'green', 'gold']
    cat0_data = y[np.where(label==0)]
    cat1_data = y[np.where(label==1)]
    cat2_data = y[np.where(label==2)]
    cat3_data = y[np.where(label==3)]
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['Ang', 'Hap', 'Neu', 'Sad'], fontsize=12, loc='lower right', markerscale=4)
    plt.savefig(os.path.join(save_root, '{}.png'.format(cv)))

for i in range(1, 2):
    plot(i)