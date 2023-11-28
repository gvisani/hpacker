
import os
import gzip, pickle
import e3nn
from e3nn import o3


def download_w3j_coefficients(lmax=14):
    print('e3nn version: ', e3nn.__version__)

    def get_wigner_3j(lmax):
        w3j_matrices = {}
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(abs(l2 - l1), min(l2 + l1, lmax) + 1):
                    w3j_matrices[(l1, l2, l3)] = o3.wigner_3j(l1, l2, l3).numpy()
        return w3j_matrices
    
    if not os.path.exists('cg_coefficients'):
        os.mkdir('cg_coefficients')
    
    with gzip.open('cg_coefficients/w3j_matrices-lmax=%d-version=%s.gz' % (lmax, e3nn.__version__), 'wb') as f:
        pickle.dump(get_wigner_3j(lmax), f)


def get_w3j_coefficients(lmax=14):
    requested_file = 'cg_coefficients/w3j_matrices-lmax=%d-version=%s.gz' % (lmax, e3nn.__version__)

    # download them if they do not exist
    if not os.path.exists(requested_file):
        download_w3j_coefficients(lmax=lmax)

    with gzip.open('cg_coefficients/w3j_matrices-lmax=%d-version=%s.gz' % (lmax, e3nn.__version__), 'r') as f:
        return pickle.load(f)
