import networkx as nx
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import subprocess

def layer_adjmat(layer_path, dist='cosine', sampsize=.05, samp=None):
    """
    Create adjacency matrix from embedded layer representations
    """
    sentlen = 50000

    #obtain sample
    if (sampsize is not None) and sampsize < 1:
        samp = get_sample(sampsize, sentlen)
        lens = get_lens(layer_path, samp)
    elif samp:
        lens = get_lens(layer_path, samp)
    else:
        samp = range(sentlen)
        lens = 741753 #Hard coded total

    samp_len = len(samp)

    # Initialize reading
    h5f = h5py.File(layer_path, 'r')
    adj_mat = np.zeros((lens, lens))
    current_idx = [0, 0]

    # Fill matrix
    for idx_i, i in enumerate(samp):
        print('Sentence %d out of %d' % (idx_i, samp_len))
        sent_i = h5f.get(str(i))[()]
        sent_i, sent_i_len = check_shape(sent_i)
        self_dist(adj_mat, current_idx, sent_i, dist)
        current_idx[1] += sent_i_len
        for idx_j in range(idx_i + 1, samp_len):
            j = samp[idx_j]
            sent_j = h5f.get(str(j))[()]
            sent_j, sent_j_len = check_shape(sent_j)
            current_idx = get_dist(adj_mat, sent_i, sent_j, current_idx, sent_i_len, dist)

        # Restart index counter at the next diagonal value
        idx_h = current_idx[0] + sent_i_len
        current_idx = [idx_h, idx_h]
    h5f.close()

    return adj_mat, samp

def get_sample(sampsize, sentlen):
    n_samp = int(sampsize * sentlen)
    samp = sorted(np.random.choice(range(sentlen), size=n_samp, replace=False))
    return samp


def check_shape(sent):
    if len(sent.shape) == 1:
        sent = sent.reshape(1, -1)
    return sent, sent.shape[0]


def self_dist(adj_mat, c_idx, sent_i, dist='euclindean'):
    """
    Compute upper triangular metrics for cosine similarity
    """
    if dist == 'cosine':
        cossim = cosine_similarity(sent_i, sent_i)
    elif dist == 'euclidean':
        cossim = euclidean_distances(sent_i, sent_i)
    else:
        raise ValueError
    cossim = np.triu(cossim, 1)
    s_len = cossim.shape[0]
    adj_mat[c_idx[0]: c_idx[0] + s_len, c_idx[1]: c_idx[1] + s_len] = cossim


def get_dist(adj_mat, sent_i, sent_j, c_idx, s_i_len, dist='euclidean'):
    if dist == 'cosine':
        cossim = cosine_similarity(sent_i, sent_j)
    elif dist == 'euclidean':
        cossim = euclidean_distances(sent_i, sent_j)
    else:
        raise ValueError
    s_j_len = cossim.shape[1]
    adj_mat[c_idx[0]: c_idx[0] + s_i_len, c_idx[1]: c_idx[1] + s_j_len] = cossim
    c_idx[1] += s_j_len
    return c_idx


def get_lens(layer_path, samp):
    h5f = h5py.File(layer_path, 'r')
    sent_lens = [h5f.get(str(sent_idx)).shape[0] for sent_idx in samp]
    h5f.close()
    return sum(sent_lens)

def layer_adjmat_tofile(layer_path, outpath='', dist='cosine', sampsize=.05, samp=None):
    """
    Create adjacency matrix from embedded layer representations and write to file
    """
    sentlen = 50000

    #obtain sample
    if samp is not None:
        lens = get_lens(layer_path, samp)
    elif (sampsize is not None) and sampsize < 1:
        n_samp = int(sampsize * sentlen)
        samp = sorted(np.random.choice(range(sentlen), size=n_samp, replace=False))
        lens = get_lens(layer_path, samp)
    else:
        samp = 0 #range(sentlen)
        lens = 741753 #Hard coded total

    samp_len = len(samp)

    # Initialize reading
    h5f = h5py.File(layer_path, 'r')
    current_idx = [0, 0]
    idx_h = 0

    # Fill matrix
    for idx_i, i in enumerate(samp):
        print('Sentence %d out of %d' % (idx_i, samp_len))
        sent_i = h5f.get(str(i))[()]
        sent_i, sent_i_len = check_shape(sent_i)
        adj_mat = np.zeros((sent_i_len, lens))
        self_dist(adj_mat, current_idx, sent_i, dist)
        idx_h += sent_i_len
        current_idx = [0, idx_h]
        for idx_j in range(idx_i + 1, samp_len):
            j = samp[idx_j]
            sent_j = h5f.get(str(j))[()]
            sent_j, _ = check_shape(sent_j)
            current_idx = get_dist(adj_mat, sent_i, sent_j, current_idx, sent_i_len, dist)
        current_idx = [0, idx_h]
        # Write block matrix
        write_submat(outpath, adj_mat, idx_i)
    h5f.close()
    return samp

def reformat_zeros(outpath):
    cmd = "sed -i 's/0.00000000/0/g' {}".format(outpath)
    subprocess.call(cmd, shell=True)


def write_submat(outpath, mat, idx_i):
    if idx_i == 0:
        w = open(outpath, 'w')
    else:
        w = open(outpath, 'a+')
    np.savetxt(w, mat, fmt='%.8f', delimiter=' ')
    w.close()
    reformat_zeros(outpath)


