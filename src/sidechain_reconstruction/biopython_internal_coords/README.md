
Contents of this folder:

1. **biopython_pdb_exploration.ipynb**. Notebook used for understanding how biopython's internal_coords can be used to reconstruct atomic coordinates given internal coordinates. This is the basis for the code in the rest of this folder. Importantly, the notebook contains plots and scripts that identify which internal coordinates hsould be used, and the dependence bdtween certain internal coordinates and $\chi$ angles.

2. **compute_reconstruction_params.py**. Computes *redundant* internal coordinates (also referred to as *reconstruction parameters*) - as considered by biopython - from data. Run this by providing a file containing one PDB ID per row as well as a directory where to find the PDB files.

3. **full_structure_reconstructor.py**. Contains the FullStructureReconstructor class, which contains methods for reconstructing atomic coordinates of residues using provided $\chi$ angles, alongside precomputed *redundant* internal coordinates (also referred to as *reconstruction parameters*). The reconstruction process is done using biopython's internal_coords module. Many methods are taken from the DLPacker codebase. The idea is to wrap any $\chi$-angle predicting method into a class that inherits from FullStructureReconstructor.

4. **constants.py**. Self-explanatory. Stuff like identifying which atoms form which $\chi$ angles for which residue, etc.
