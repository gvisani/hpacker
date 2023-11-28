
import torch
from torch import nn
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *

def get_edges_for_l3_and_L(l3, L, optimize_speed=True):
    # NOTE: As far as I can tell empirically, the computation of the minimum spanning tree with default
    # parameters is deterministic. We might run into trouble if it is not, but I am not 100% sure.

    import numpy as np
    import networkx as nx

    edges = []
    for l1 in range(L+1):
        for l2 in range(l1, L+1):
            if l3 >= np.abs(l1 - l2) and l3 <= l1 + l2:
                if optimize_speed:
                    edges.append((l1, l2, (2*l1+1)*(2*l2+1)))
                else:
                    edges.append((l1, l2, 1))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    
    MST = nx.minimum_spanning_tree(G, weight='weight')
    
    # add self-connections
    for l in range(L+1):
        if l3 <= l + l:
            MST.add_edge(l, l)
    
    # organize connections with l1 >= l2
    edges = [(max(edge), min(edge)) for edge in MST.edges]
    
    # sort connections
    edges = list(sorted(edges))
    
    return edges

def get_efficient_connections(L_in: int,
                              L_out: Union[int, Tuple[int, int]]) -> Dict[int, Dict[int, List[int]]]:
    if isinstance(L_out, int):
        L_out = (0, L_out)
    
    connections = {}
    for l3 in range(L_out[0], L_out[1]+1):
        edges = get_edges_for_l3_and_L(l3, L_in)
        for edge in edges:
            l1, l2 = edge[0], edge[1]
            if l1 not in connections:
                connections[l1] = {}
            if l2 not in connections[l1]:
                connections[l1][l2] = []
            connections[l1][l2].append(l3)
    return connections

class TP_nonlinearity(torch.nn.Module):
    '''
    Implements an SO(3) tensor product of a tensor with itself.

    '''
    def __init__(self, 
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 filter_ir_out: Optional[List[Union[str, o3.Irrep]]] = None,
                 ls_rule: str = 'full',
                 channel_rule: str = 'full',
                 filter_symmetric: bool = True): # only specified for rules in ['full', 'elementwise']. Always true for 'efficient'
        super().__init__()

        self.irreps_in = irreps_in
        self.filter_symmetric = filter_symmetric
        self.w3j_matrices = w3j_matrices

        self.ls = list(sorted(list(set(irreps_in.ls))))

        assert ls_rule in ['full', 'elementwise', 'efficient']
        self.ls_rule = ls_rule

        assert channel_rule in ['full', 'elementwise']
        self.channel_rule = channel_rule

        for irr in irreps_in:
            assert irr.ir.p == 1 # only implemented SO(3) tensor product, parity is not considered

        for ir in filter_ir_out:
            assert ir.p == 1 # only implemented SO(3) tensor product, parity is not considered

        if ls_rule in ['full', 'elementwise']:
            out = []
            for i_1, (mul_1, ir_1) in enumerate(irreps_in):
                for i_2, (mul_2, ir_2) in enumerate(irreps_in):
                    
                    if filter_symmetric and ir_2.l < ir_1.l: # crucial difference with standard full tensor product
                        continue
                    
                    if ls_rule == 'elementwise' and ir_1 != ir_2:
                        continue
                    
                    for ir_out in ir_1 * ir_2:
                        if filter_ir_out is not None and ir_out not in filter_ir_out:
                            continue
                        
                        if channel_rule == 'full':
                            out.append((mul_1 * mul_2, ir_out))
                        elif channel_rule == 'elementwise':
                            assert mul_1 == mul_2
                            out.append((mul_1, ir_out))
            
            self.irreps_out = o3.Irreps(out).sort().irreps.simplify()
            self.ls_out = [irr.ir.l for irr in self.irreps_out]
            self.set_ls_out = set(self.ls_out)
        
        elif ls_rule == 'efficient':
            # The current implementation makes the following two assumptions:
            # 1) irreps_in has all irreps from 0 to some L_in without holes, and in ascending order
            ls_in = [irr.ir.l for irr in self.irreps_in]
            assert ls_in[0] == 0
            for i in range(1, len(ls_in)):
                assert ls_in[i] == ls_in[i-1] + 1
            
            L_in = ls_in[-1]

            # 2) filter_ir_out has all irreps from some L_out_min to some L_out_max without holes, and in ascending order
            ls_out = [ir.l for ir in filter_ir_out]
            for i in range(1, len(ls_out)):
                assert ls_out[i] == ls_out[i-1] + 1
            
            L_out = (ls_out[0], ls_out[-1])

            self.connections = get_efficient_connections(L_in, L_out)

            # compute irreps_out
            l3_mul_counts = {}
            for (mul_1, ir_1) in irreps_in:
                if ir_1.l in self.connections:
                    for (mul_2, ir_2) in irreps_in:
                        if ir_2.l in self.connections[ir_1.l]:
                            for l3 in self.connections[ir_1.l][ir_2.l]:

                                if l3 not in l3_mul_counts:
                                    l3_mul_counts[l3] = 0

                                if channel_rule == 'full':
                                    l3_mul_counts[l3] += mul_1 * mul_2
                                elif channel_rule == 'elementwise':
                                    assert mul_1 == mul_2 # Critical!!!
                                    l3_mul_counts[l3] += mul_1

            out = []
            for l3 in l3_mul_counts:
                out.append((l3_mul_counts[l3], '%de' % (l3)))
            
            self.irreps_out = o3.Irreps(out).sort().irreps.simplify()
            self.ls_out = [irr.ir.l for irr in self.irreps_out]
            self.set_ls_out = set(self.ls_out)

    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output = {}
        for l3 in self.ls_out:
            output[l3] = []


        if self.ls_rule in ['full', 'elementwise']:
        
            for l1 in self.ls:
                for l2 in self.ls:

                    if self.ls_rule == 'elementwise' and l1 != l2:
                        continue
                    
                    if self.filter_symmetric and l2 < l1:
                        continue
                    
                    output_ls = [l for l in list(range(abs(l2-l1), l2+l1+1)) if l in self.set_ls_out]
                    
                    if len(output_ls) > 0:

                        if self.channel_rule == 'full':
                            outer_product = torch.einsum('bim,bjn->bijmn', x[l1], x[l2])
                            op_shape = outer_product.shape
                            outer_product = outer_product.view(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                        
                        elif self.channel_rule =='elementwise':
                            outer_product = torch.einsum('bim,bin->bimn', x[l1], x[l2])

                        for l3 in output_ls:
                            output[l3].append(torch.einsum('mnM,bimn->biM', self.w3j_matrices[(l1, l2, l3)], outer_product))
        

        elif self.ls_rule == 'efficient':

            for l1 in self.connections:
                for l2 in self.connections[l1]:
                    
                    if self.channel_rule == 'full':
                        outer_product = torch.einsum('bim,bjn->bijmn', x[l1], x[l2])
                        op_shape = outer_product.shape
                        outer_product = outer_product.reshape(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                    
                    elif self.channel_rule =='elementwise':
                        outer_product = torch.einsum('bim,bin->bimn', x[l1], x[l2])

                    for l3 in self.connections[l1][l2]:
                        output[l3].append(torch.einsum('mnM,bimn->biM', self.w3j_matrices[(l1, l2, l3)], outer_product))


        for l3 in self.ls_out:
            output[l3] = torch.cat(output[l3], axis=1)

        return output