import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def load_data(cv):
    root = 'checkpoints/analysis_ef/{}/'.format(cv)
    A = np.load(os.path.join(root, 'A_feat_test.npy'))
    V = np.load(os.path.join(root, 'V_feat_test.npy'))
    L = np.load(os.path.join(root, 'L_feat_test.npy'))
    label = np.load(os.path.join(root, 'test_label.npy'))
    return A, V, L, label

def plot_modality(cv):
    file_dir = os.path.dirname(__file__)
    save_root = os.path.join(file_dir, 'tsne', 'analysis_ef')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    A, V, L, label = load_data(cv)
    total_data = np.concatenate([A,V,L], axis=0)
    label = np.concatenate([label, label, label], axis=0)
    print("Finish loading data: ")
    print("Total data:", total_data.shape)
    print("Total label:", label.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(total_data)
    A_tsne = y[:len(A)]
    V_tsne = y[len(A): len(A)+len(V)]
    L_tsne = y[len(A)+len(V): ]
    # init graph
    plt.figure(figsize=(5, 5))
    # define colors
    # colors = ['red', 'green', 'orange']
    colors = ['red', 'green', 'blue']
    colors_label = ['red', 'blue', 'green', 'gold']
    # plt.axis('equal')
    # plot modality tsne
    plt.scatter(A_tsne[:, 0], A_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(V_tsne[:, 0], V_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.legend(['A', 'V', 'L'])
    plt.savefig(os.path.join(save_root, "{}_modality.png".format(cv)))

def plot_emo(cv):
    file_dir = os.path.dirname(__file__)
    save_root = os.path.join(file_dir, 'tsne', 'analysis_ef')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    A, V, L, label = load_data(cv)
    total_data = np.concatenate([A,V,L], axis=0)
    print("Finish loading data: ")
    print("Total data:", total_data.shape)
    print("Total label:", label.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(total_data)
    A_tsne = y[:len(A)]
    V_tsne = y[len(A): len(A)+len(V)]
    L_tsne = y[len(A)+len(V): ]
    # init graph
    fig, (ax_a, ax_v, ax_l) = plt.subplots(ncols=3, figsize=(12, 5))
    colors_label = ['crimson', 'navy', 'forestgreen', 'goldenrod']
    # plot emotion category for A
    cat0_data = A_tsne[np.where(label==0)]
    cat1_data = A_tsne[np.where(label==1)]
    cat2_data = A_tsne[np.where(label==2)]
    cat3_data = A_tsne[np.where(label==3)]
    ax_a.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    ax_a.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    ax_a.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    ax_a.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    ax_a.legend(['ang', 'hap', 'neu', 'sad'])
    ax_a.set_box_aspect(1)
    ax_a.set_title('Acoustic')

    # plot emotion category for V
    cat0_data = V_tsne[np.where(label==0)]
    cat1_data = V_tsne[np.where(label==1)]
    cat2_data = V_tsne[np.where(label==2)]
    cat3_data = V_tsne[np.where(label==3)]
    ax_v.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    ax_v.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    ax_v.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    ax_v.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    ax_v.legend(['ang', 'hap', 'neu', 'sad'])
    ax_v.set_box_aspect(1)
    ax_v.set_title('Visual')

    # plot emotion category for L
    cat0_data = L_tsne[np.where(label==0)]
    cat1_data = L_tsne[np.where(label==1)]
    cat2_data = L_tsne[np.where(label==2)]
    cat3_data = L_tsne[np.where(label==3)]
    ax_l.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    ax_l.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    ax_l.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    ax_l.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    ax_l.legend(['ang', 'hap', 'neu', 'sad'])
    ax_l.set_box_aspect(1)
    ax_l.set_title('Textual')

    plt.savefig(os.path.join(save_root, "{}_emo.png".format(cv)))
    

# for i in range(1, 11):
#     plot(i)
# for i in range(1, 11):
#     plot_modality(i)
#     plot_emo(i)
plot_modality(1)