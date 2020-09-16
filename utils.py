import networkx as nx
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def layer_adjmat(layer_path, outpath='', sampsize=.05):
    """
    Create adjacency matrix from embedded layer representations
    """
    sentlen = 52082

    #obtain sample
    if (sampsize is not None) or sampsize < 1:
        n_samp = int(sampsize * sentlen)
        samp = sorted(np.random.choice(range(sentlen), size=n_samp, replace=False))
        lens = get_lens(layer_path, samp)
    else:
        samp = range(sentlen)
        lens = 741753 #Hard coded total

    samp_len = len(samp)

    # Initialize reading
    h5f = h5py.File(layer_path, 'r')
    adj_mat = np.zeros((lens, lens))
    current_idx = [0, 0]

    for idx_i, i in enumerate(samp):
        print('Sentence %d out of %d' % (idx_i, samp_len))
        sent_i = h5f.get(str(i))[()]
        sent_i = check_shape(sent_i)
        sent_i_len = self_dist(adj_mat, current_idx, sent_i)
        current_idx[1] += sent_i_len
        for idx_j in range(idx_i + 1, samp_len):
            j = samp[idx_j]
            sent_j = h5f.get(str(j))[()]
            sent_j = check_shape(sent_j)
            current_idx = get_dist(adj_mat, sent_i, sent_j, current_idx, sent_i_len)

        # Restart index counter at the next diagonal value
        idx_h = current_idx[0] + sent_i_len
        current_idx = [idx_h, idx_h]

    return adj_mat

def check_shape(sent):
    if len(sent.shape) == 1:
        sent = sent.reshape(1, -1)
    return sent, sent.shape[0]


def self_dist(adj_mat, c_idx, sent_i):
    """
    Compute upper triangular metrics for cosine similarity
    """
    cossim = cosine_similarity(sent_i, sent_i)
    cossim = np.triu(cossim, 1)
    s_len = cossim.shape[0]
    c = c_idx[0]
    adj_mat[c: c + s_len, c: c + s_len] = cossim

    return s_len


def get_dist(adj_mat, sent_i, sent_j, c_idx, s_i_len):
    cossim = cosine_similarity(sent_i, sent_j)
    s_j_len = cossim.shape[1]
    adj_mat[c_idx[0]: c_idx[0] + s_i_len, c_idx[1]: c_idx[1] + s_j_len] = cossim
    c_idx[1] += s_j_len

    return c_idx


def get_lens(layer_path, samp):
    h5f = h5py.File(layer_path, 'r')
    sent_lens = [h5f.get(str(sent_idx)).shape[0] for sent_idx in samp]
    h5f.close()
    return sum(sent_lens)
