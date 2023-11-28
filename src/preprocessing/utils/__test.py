

import os, re
from collections import defaultdict

# read in the charges from special file
CHARGES_AMBER99SB = defaultdict(lambda: 0) # output 0 if the key is absent
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charges.rtp'), 'r') as f:
    for line in f:
        if line[0] == '[' or line[0] == ' ':
            if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                CHARGES_AMBER99SB[key] = defaultdict(lambda: 0)
            else:
                l = re.split(r' +', line[:-1])
                CHARGES_AMBER99SB[key][l[1]] = float(l[3])

print(list(CHARGES_AMBER99SB.keys()))
print(list(CHARGES_AMBER99SB['ZN'].keys()))
print(list(CHARGES_AMBER99SB['Zn'].keys()))

