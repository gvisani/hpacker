// compile with: gcc -shared -o utils.so -fPIC utils.c

#include <stdio.h>


int in_backbone(int atom, int* backbone_atoms, int len) 
{
    for (int i = 0; i < len; i++) 
    {
        if (atom == backbone_atoms[i]) 
        {
            return 1;
        }
    }
    return 0;
}

/**
Params:
    - SIZE <- number of atoms (NOT padded length)
    - MAX_SITES <- assumed maximum number of amino acids in a protein
    - MAX_CHAINS <- assumed maximum number of chains in a protein
    - SIZE_BYTES <- number of bytes per string in res_id (most importantly, chain)
    - MIN_CHAIN_ASCII_VALUE <- minimum ASCII value of chain (e.g. 'A' = 65, '2' = 50)
    - chains <- bytes of chains (b'A', b'B', etc)
    - sites <- int[] of site numbers
    - names <- Bytes of atom names (note: happen to be 32 bytes => int*)
    - seen, removed <- zeroed bool arrays of size MAX_CHAINS * MAX_SITES
    - random <- float[] (max atom) array on [0, 1]
    - mask <- zeroed bool array of size (max_atoms)
    - p <- proportion of sidechains to keep
    - backbone_atoms <- bytes of backbone atoms
    - bb_len <- number of backbone atoms
    - central_chain, central_site <- site and chain of central amino acid

Returns:
    - None
    - Modifies mask

Notes:
    - res_ids are unique within protein by 1) chain and 2) site. 
      By assuming there are no more than 10 chains and 10,000 sites, 
      we can created a collision-free "hash" map for `seen` and `removed`
      in 100,000 bytes each (0.1 MB)
*/
int get_mask(int SIZE, int MAX_SITES, int MAX_CHAINS, int SIZE_BYTES, int MIN_CHAIN_ASCII_VALUE,
              char* chains, long* sites, int* names, 
              char* seen, char* removed, double* random, char* mask, 
              float p, int* backbone_atoms, int bb_len, 
              char central_chain, long central_site)
{   
    // printf("SIZE_BYTES: %d\n", SIZE_BYTES);
    int central_index = ((central_chain - MIN_CHAIN_ASCII_VALUE) * MAX_SITES + (central_site)) % (MAX_CHAINS * MAX_SITES); // (central_site + 100)
    // printf("central_index: %d\n", central_index);
    for (int i = 0; i < SIZE; i++) 
    {        
        int id_index = ((chains[i * SIZE_BYTES] - MIN_CHAIN_ASCII_VALUE) * MAX_SITES + (sites[i])) % (MAX_CHAINS * MAX_SITES); // (sites[i] + 100)
        // printf("i: %d, chains[i * SIZE_BYTES]: %d, chains[i * SIZE_BYTES] - 'A': %d, sites[i]: %d\n", i, chains[i * SIZE_BYTES], (chains[i * SIZE_BYTES] - 'A'), sites[i]);

        if (id_index < 0 || id_index >= MAX_CHAINS * MAX_SITES) 
        {
            printf("ERROR: id_index out of bounds: %d\n", id_index);
            printf("i: %d, chains[i * SIZE_BYTES]: %d, chains[i * SIZE_BYTES] - 'A': %d, sites[i]: %d\n", i, chains[i * SIZE_BYTES], (chains[i * SIZE_BYTES] - 'A'), sites[i]);
            return 0;
        }
        
        if (! (seen[id_index]))
        {
            seen[id_index] = 1;
            removed[id_index] = random[i] < p || id_index == central_index; // by default the sidechain of the central residue is removed. Then re-instated at a later time
        }
        mask[i] = in_backbone(names[i], backbone_atoms, bb_len) | (!removed[id_index]);
    }
    return 1;
}