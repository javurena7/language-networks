import matplotlib.pyplot as plt; plt.ion()
import constructor as cons
import pickle
import numpy as np


sentlen = 52082
cases = [0, 1, 11, 12]
lstring = "layer%d_bert-base-cased.h5"
ostring = "layer%d_%s.txt"

def run_small_samp(data_path, out_path, sampsize=.01):
    samp = cons.get_sample(sampsize, sentlen)
    pickle.dump(samp, open(out_path + 'sample.p', 'wb'))
    dists = ['cosine', 'euclidean']
    for case in cases:
        fname = data_path + lstring % (case)
        for dist in dists:
            print("Computing layer %d with %s distance." %(case, dist))
            oname = out_path + ostring % (case, dist)
            _ = cons.layer_adjmat_tofile(fname, oname, sampsize=None, samp=samp, dist=dist)



def plot_histograms(data_path, out_path):
    dists = ['cosine', 'euclidean']
    fig, axs = plt.subplots(1, 2)
    for i, dist in enumerate(dists):
        for case in cases:
            mat = np.loadtxt(out_path + ostring % (case, dist))
            lens = mat.shape[0]
            data = mat[np.triu_indices(lens, k=1)].flatten()
            axs[i].hist(data, 100, alpha=.5, label='layer %s' % case, density=True)
        axs[i].legend(loc=0)
        axs[i].set_xlabel(dist)
    fig.tight_layout()
    fig.savefig(out_path + 'histograms.pdf')


def plot_dists_bylayer(data_path, out_path):
    dists = ['cosine', 'euclidean']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, case in enumerate(cases):
        mat_e = np.loadtxt(out_path + ostring % (case, 'euclidean'))
        mat_c = np.loadtxt(out_path + ostring % (case, 'cosine'))
        lens = mat_e.shape[0]
        data_e = mat_e[np.triu_indices(lens, k=1)].flatten()
        data_c = mat_c[np.triu_indices(lens, k=1)].flatten()
        aj = i % 2
        ai = int(i / 2)
        axs[ai, aj].hexbin(data_e, data_c) #, alpha=.01)
        axs[ai, aj].set_title('layer %s' % case)
        axs[ai, aj].set_xlabel('euclidean dist')
        axs[ai, aj].set_ylabel('cosine sim')

    fig.tight_layout()
    fig.savefig(out_path + 'dists_bylayer.pdf')



def plot_dists_evol(data_path, out_path, basis=0, cases=[1, 11, 12]):
    dists = ['cosine', 'euclidean']
    #_ = cases.pop(basis)
    fig, axs = plt.subplots(2, len(cases))

    mat_e = np.loadtxt(out_path + ostring % (basis, 'euclidean'))
    mat_c = np.loadtxt(out_path + ostring % (basis, 'cosine'))
    lens = mat_e.shape[0]
    data_eb = mat_e[np.triu_indices(lens, k=1)].flatten()
    data_cb = mat_c[np.triu_indices(lens, k=1)].flatten()

    for i, case in enumerate(cases):
        mat_e = np.loadtxt(out_path + ostring % (case, 'euclidean'))
        mat_c = np.loadtxt(out_path + ostring % (case, 'cosine'))
        data_e = mat_e[np.triu_indices(lens, k=1)].flatten()
        data_c = mat_c[np.triu_indices(lens, k=1)].flatten()

        axs[0, i].hexbin(data_eb, data_e)
        axs[1, i].hexbin(data_cb, data_c)
        axs[0, i].set_title('layer %s' % case)
        axs[0, i].set_xlabel('euclidean layer %s' % basis)
        axs[1, i].set_xlabel('cosine layer %s' % basis )
        axs[0, i].set_ylabel('euclidean layer %s' % case)
        axs[1, i].set_ylabel('cosine layer %s' % case )

    fig.tight_layout()
    fig.savefig(out_path + 'dists_evol_%d.pdf' % basis)



