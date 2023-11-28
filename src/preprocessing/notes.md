

get_structural_info with pyrosetta may stall. It seems to infinitely loop on pdb 1NTH. It stalls either in "pose_from_pdb" or in "calculate_sasa" or "pose_coords_as_rows".

elif args.channels[0] == 'AAs':
    args.channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', b'S', b'T', b'W', b'Y', b'V', b'G']
