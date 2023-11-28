

Contents of this folder:

1. **compute_reconstructor_params.py**. Contains functions to compute *redundant* internal coordinates (also referred to as *reconstruction parameters*) from data. To run this on a set of PDB structures, the structures must be first preprocessed using `get_structural_info.py` and `get_neighborhoods.py` (with the `--central_residue_only flag), and the resulting hdf5 file provided as input to the script.

2. **reconstruction__{numpy/torch}.py**. Reconstructor class, for reconstructing atomic coordinates of residues using provided $\chi$ angles or the plane norms, alongside precomputed *redundant* internal coordinates (also referred to as *reconstruction parameters*). numpy: one residue at a time, using numpy operations. torch: multiple residues in batches, possibly on GPU as well.

3. **utils__{numpy/torch}.py**. Utility functions for the respective Reconstructors, such as computing the nrom of a plane given three points, etc.

34 **tests__{numpy/torch}.py**. Some tests for the respective Reconstructors.





