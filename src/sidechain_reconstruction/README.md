
This folder contains several different ways of placing side-chain atom coordinates by providing the residues' inferent degrees of freedom, such as the $\chi$ angles or the plane norms that ultimately define the $\chi$ angles.

Placing side-chain atoms requires two ingredients:
1. Knowing its degrees of freedom ($\chi$ angles or proxies for them)
2. Knowing all the other fixed *redundant* internal coordinates, which are either fixed or deterministically determined from the $\chi$ angles, and how to piece them together to fully reconstruct the atomic structure

Currently, we have two ways of doing this:
1. **manual**. Reconstructs only the atoms accociated with the $\chi$ angles, given the $\chi$ angles or the plane norms. Redundant internal coordinates are computed from data. PROs: differentiable; can be done in batches on torch, probably also on GPU; fast. CONs: lacks implementation of the remaining redundant internal coordinates which has to be done from scratch.
2. **biopython**. Leverages biopython's internal_coords module to reconstruct atomic coordinates given internal coordinates ($\chi$ angles are provided, and redundant internal coordinates are inferred from data). PROs: avoids manual writing of the code of the reconstruction process, which takes time to do. CONs: CPU-only; kinda slow; not differentiable.
