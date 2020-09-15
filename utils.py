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
    current_idx = 0
    for idx_i, i in enumerate(samp):
        sent_i = h5f.get(str(i)).get[()]
        self_dist(adj_mat, current_idx, sent_i)
        for idx_j in range(idx_i + 1, samp_len):
            j = samp(idx_j)
            sent_j = h5f.get(str(j)).get[()]
            get_dist(sent_i, sen_j)


def self_dist(adj_mat, current_idx, sent_i):
    cossim = 1 - cosine_similarity(sent_i, sent_i)
    cossim = np.triu(cossim, 1)
    sent_len = sent_i.shape[0]

    for i in range(sent_len):
        for j in range(i + 1, sent_len):
            ajd_mat[current_idx + i, current_idx + j] =



def get_lens(layer_path, samp):
    i = 0
    h5f = h5py.File(layer_path, 'r')
    sent_lens = [h5f.get(str(sent_idx)).shape[0] for sent_idx in samp]
    h5f.close()
    return sum(sent_lens)
