"""Module for converting complex spherical harmonics to real basis"""

import numpy as np

def change_basis_complex_to_real(l: int, dtype=None, device=None) -> np.ndarray:
    """
    Function to convert chang of basis matrix from complex to real spherical
    harmonics
     
    Taken from e3nn: https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_wigner.py
    """
    # taken from e3nn
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l - abs(m)] = 1j / 2**0.5
        q[l + m, l + abs(m)] = -(-1)**abs(m) * 1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l - abs(m)] = 1 / 2**0.5
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
    q = q  # No factor of 1j

    return q

