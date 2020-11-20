import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def load_data(cv):
    A_root = 'checkpoints/analysis_recon_zvl/{}/'.format(cv)
    L_root = 'checkpoints/analysis_recon_avz/{}/'.format(cv)
    V_root = 'checkpoints/analysis_recon_azl/{}/'.format(cv)

    A = np.load(os.path.join(A_root, 'A_feat_test.npy'))
    rA = np.load(os.path.join(A_root, 'recon_A_feat_test.npy'))
    A_label = np.load(os.path.join(A_root, 'test_label.npy'))

    L = np.load(os.path.join(L_root, 'L_feat_test.npy'))
    rL = np.load(os.path.join(L_root, 'recon_L_feat_test.npy'))
    L_label = np.load(os.path.join(L_root, 'test_label.npy'))

    V = np.load(os.path.join(V_root, 'V_feat_test.npy'))
    rV = np.load(os.path.join(V_root, 'recon_V_feat_test.npy'))
    V_label = np.load(os.path.join(V_root, 'test_label.npy'))
    return A, rA, A_label, L, rL, L_label, V, rV, V_label

def plot_AVL(cv):
    file_dir = os.path.dirname(__file__)
    save_root = os.path.join(file_dir, 'tsne', 'analysis_recon')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    A, rA, A_label, L, rL, L_label, V, rV, V_label = load_data(cv)
    # init graph
    plt.figure(figsize=(6, 6))
    # colors = ['orange', 'brown', 'green', 'teal']
    colors = ['red', 'lightcoral', 'blue', 'deepskyblue', 'green', 'springgreen']
    total_data = np.concatenate([A, rA, L, rL, V, rV], axis=0)
    print("Total data:", total_data.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(total_data)
    A_tsne = y[:len(A)]
    rA_tsne = y[len(A): len(A)+len(rA)]
    L_tsne = y[len(A)+len(rA): len(A)+len(rA)+len(L)]
    rL_tsne = y[len(A)+len(rA)+len(L): len(A)+len(rA)+len(L)+len(rL)]
    V_tsne = y[len(A)+len(rA)+len(L)+len(rL): len(A)+len(rA)+len(L)+len(rL)+len(V)]
    rV_tsne = y[len(A)+len(rA)+len(L)+len(rL)+len(V):]
    
    plt.scatter(A_tsne[:, 0], A_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(rA_tsne[:, 0], rA_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.scatter(rL_tsne[:, 0], rL_tsne[:, 1], s=1, c=colors[3], cmap=plt.cm.Spectral)
    plt.scatter(V_tsne[:, 0], V_tsne[:, 1], s=1, c=colors[4], cmap=plt.cm.Spectral)
    plt.scatter(rV_tsne[:, 0], rV_tsne[:, 1], s=1, c=colors[5], cmap=plt.cm.Spectral)
    plt.legend(['a', 'a-Imagined', 't', 't-Imagined', 'v', 'v-Imagined'],
                    fontsize=12, loc='lower right', markerscale=4)
    plt.savefig(os.path.join(save_root, "{}_AVL.png".format(cv)))
    '''
    colors_label = ['red', 'blue', 'green', 'gold']
    # plot L
    L_data = np.concatenate([L, rL], axis=0)
    L_label = np.concatenate([L_label, L_label], axis=0)
    print('Total L: ', L_data.shape)
    print('Label L: ', L_label.shape)
    ts_l = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y_l = ts_l.fit_transform(L_data)
    L_tsne = y_l[:len(L)]
    rL_tsne = y_l[len(L):]

    # plot recon and raw L 
    plt.subplot(221)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(rL_tsne[:, 0], rL_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.legend(['L', 'rL'])
    # plot emotion category of L
    plt.subplot(222)
    cat0_data = y_l[np.where(L_label==0)]
    cat1_data = y_l[np.where(L_label==1)]
    cat2_data = y_l[np.where(L_label==2)]
    cat3_data = y_l[np.where(L_label==3)]
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    
    # plot V
    V_data = np.concatenate([V, rV], axis=0)
    V_label = np.concatenate([V_label, V_label], axis=0)
    print('Total V: ', V_data.shape)
    print('Label V: ', V_label.shape)
    ts_v = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y_v = ts_v.fit_transform(V_data)
    V_tsne = y_v[:len(V)]
    rV_tsne = y_v[len(V):]
    # plot recon and raw V
    plt.subplot(223)
    plt.scatter(V_tsne[:, 0], V_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.scatter(rV_tsne[:, 0], rV_tsne[:, 1], s=1, c=colors[3], cmap=plt.cm.Spectral)
    plt.legend(['V', 'rV'])
    # plot emotion category of L
    plt.subplot(224)
    cat0_data = y_v[np.where(V_label==0)]
    cat1_data = y_v[np.where(V_label==1)]
    cat2_data = y_v[np.where(V_label==2)]
    cat3_data = y_v[np.where(V_label==3)]
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    plt.savefig(os.path.join(save_root, "{}.png".format(cv)))
    '''

def plot_recon_L(cv):
    file_dir = os.path.dirname(__file__)
    save_root = os.path.join(file_dir, 'tsne', 'analysis_recon')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    A, rA, A_label, L, rL, L_label, V, rV, V_label = load_data(cv)
    # init graph
    plt.figure(figsize=(3, 3))
    # colors = ['orange', 'brown', 'green', 'teal']
    colors = ['red', 'lightcoral', 'blue', 'deepskyblue', 'green', 'springgreen']
    total_data = np.concatenate([A, rA, L, rL, V, rV], axis=0)
    print("Total data:", total_data.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(total_data)
    A_tsne = y[:len(A)]
    rA_tsne = y[len(A): len(A)+len(rA)]
    L_tsne = y[len(A)+len(rA): len(A)+len(rA)+len(L)]
    rL_tsne = y[len(A)+len(rA)+len(L): len(A)+len(rA)+len(L)+len(rL)]
    V_tsne = y[len(A)+len(rA)+len(L)+len(rL): len(A)+len(rA)+len(L)+len(rL)+len(V)]
    rV_tsne = y[len(A)+len(rA)+len(L)+len(rL)+len(V):]

    # init graph
    fig, (ax_L, ax_rL) = plt.subplots(ncols=2, figsize=(12, 5))
    colors_label = ['crimson', 'navy', 'forestgreen', 'goldenrod']
    # plot emotion category for A
    cat0_data = L_tsne[np.where(L_label==0)]
    cat1_data = L_tsne[np.where(L_label==1)]
    cat2_data = L_tsne[np.where(L_label==2)]
    cat3_data = L_tsne[np.where(L_label==3)]
    ax_L.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    ax_L.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    ax_L.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    ax_L.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    ax_L.legend(['ang', 'hap', 'neu', 'sad'])
    ax_L.set_box_aspect(1)
    ax_L.set_title('L raw')

    # plot emotion category for A
    cat0_data = rL_tsne[np.where(L_label==0)]
    cat1_data = rL_tsne[np.where(L_label==1)]
    cat2_data = rL_tsne[np.where(L_label==2)]
    cat3_data = rL_tsne[np.where(L_label==3)]
    ax_rL.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    ax_rL.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    ax_rL.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    ax_rL.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    ax_rL.legend(['ang', 'hap', 'neu', 'sad'])
    ax_rL.set_box_aspect(1)
    ax_rL.set_title('L reconstructed')
    plt.savefig(os.path.join(save_root, "{}_L_emo.png".format(cv)))

# for i in range(1, 11):
#     plot(i)
# plot_recon_L(1)
# plot(1)
plot_AVL(1)
