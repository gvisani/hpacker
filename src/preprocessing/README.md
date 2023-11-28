
## Extracting neighborhoods and computing zernike projections (zernikegrams) from a list of PDBs

The pipeline is divided in three steps, each with its corresponding script:

1. `get_structural_info.py` takes a list of PDBs as input and uses pyrosetta to parse them, saving structural information (atom coordinates, elements, SASA, partial charge, etc.) into an hdf5 file.
2. `get_neighborhoods.py` takes the hdf5 file generated in step 1 as input and constructs neighborhoods around each atom in the PDBs. The neighborhoods are saved into a new hdf5 file.
3. `get_zernikegrams.py` takes the hdf5 file generated in step 2 as input and constructs zernikegrams for each neighborhood. The zernikegrams are saved into a new hdf5 file.

Each script also bears a function, by the same name, that allows users to run the steps individually within other scripts. Import them simply by running:

```python
    from protein_holography_pytorch.preprocessing_faster import get_structural_info, get_neighborhoods, get_zernikegrams
```

When processing large volumes of proteins, we recommend using the scripts to leverage multiprocessing for faster computation. Processing 1,200 PDBs takes 15 minutes on 25 cores and 96GB of RAM (less memory is probably fine, but we haven't tested it).


**Note:** the use of pyrosetta to parse PDBs is currently necessary for computing SASA and partial charge, which are used by H-CNN. It is **not** necessary for H-(V)AE, since our models are not trained with SASa and partial charge. We will soon add an option to skip pyrosetta and use only biopython to parse PDBs.


## TODO

1. Add biopython parsing (Adapt from Holographic-VAE repository)
2. Add SASA and partial charge calculations on top of biopython parsing (can use FreeSASA for SASA; I don't know what for partial charge, maybe have a look here: https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html)
