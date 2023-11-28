
import torch
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *

from .nonlinearity import get_efficient_connections


class so3_tensor_product(torch.nn.Module):
    '''
    Implements an SO(3) tensor product of two tensors, assuming they have the same irreps

    '''
    def __init__(self, 
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 filter_ir_out: Optional[List[Union[str, o3.Irrep]]] = None,
                 ls_rule: str = 'full',
                 channel_rule: str = 'full'):
        super().__init__()

        self.irreps_in = irreps_in
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

    def forward(self, x1: Dict[int, Tensor], x2: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output = {}
        for l3 in self.ls_out:
            output[l3] = []


        if self.ls_rule in ['full', 'elementwise']:
        
            for l1 in self.ls:
                for l2 in self.ls:

                    if self.ls_rule == 'elementwise' and l1 != l2:
                        continue
                    
                    output_ls = [l for l in list(range(abs(l2-l1), l2+l1+1)) if l in self.set_ls_out]
                    
                    if len(output_ls) > 0:

                        if self.channel_rule == 'full':
                            outer_product = torch.einsum('bim,bjn->bijmn', x1[l1], x2[l2])
                            op_shape = outer_product.shape
                            outer_product = outer_product.view(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                        
                        elif self.channel_rule =='elementwise':
                            outer_product = torch.einsum('bim,bin->bimn', x1[l1], x2[l2])

                        for l3 in output_ls:
                            output[l3].append(torch.einsum('mnM,bimn->biM', self.w3j_matrices[(l1, l2, l3)], outer_product))
        

        elif self.ls_rule == 'efficient':

            for l1 in self.connections:
                for l2 in self.connections[l1]:
                    
                    if self.channel_rule == 'full':
                        outer_product = torch.einsum('bim,bjn->bijmn', x1[l1], x2[l2])
                        op_shape = outer_product.shape
                        outer_product = outer_product.reshape(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                    
                    elif self.channel_rule =='elementwise':
                        outer_product = torch.einsum('bim,bin->bimn', x1[l1], x2[l2])

                    for l3 in self.connections[l1][l2]:
                        output[l3].append(torch.einsum('mnM,bimn->biM', self.w3j_matrices[(l1, l2, l3)], outer_product))


        for l3 in self.ls_out:
            output[l3] = torch.cat(output[l3], axis=1)

        return output


class o3_tensor_product(torch.nn.Module):
    '''
    Implements an O(3) tensor product of two tensors, assuming they have the same irreps

    '''
    def __init__(self, 
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 filter_ir_out: Optional[List[Union[str, o3.Irrep]]] = None,
                 ls_rule: str = 'full',
                 channel_rule: str = 'full'):
        super().__init__()

        self.irreps_in = irreps_in
        self.w3j_matrices = w3j_matrices

        self.ls = list(sorted(list(set(irreps_in.ls))))

        assert ls_rule in ['full', 'elementwise', 'efficient']
        self.ls_rule = ls_rule

        assert channel_rule in ['full', 'elementwise']
        self.channel_rule = channel_rule

        if ls_rule in ['full', 'elementwise']:
            out = []
            for i_1, (mul_1, ir_1) in enumerate(irreps_in):
                for i_2, (mul_2, ir_2) in enumerate(irreps_in):
                    
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
            raise NotImplementedError()

    def forward(self, x1: Dict[int, Tensor], x2: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output = {}
        for l3 in self.ls_out:
            output[l3] = []


        if self.ls_rule in ['full', 'elementwise']:
        
            for l1 in self.ls:
                for l2 in self.ls:

                    if self.ls_rule == 'elementwise' and l1 != l2:
                        continue
                    
                    output_ls = [l for l in list(range(abs(l2-l1), l2+l1+1)) if l in self.set_ls_out]
                    
                    if len(output_ls) > 0:

                        if self.channel_rule == 'full':
                            outer_product = torch.einsum('bim,bjn->bijmn', x1[l1], x2[l2])
                            op_shape = outer_product.shape
                            outer_product = outer_product.view(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                        
                        elif self.channel_rule =='elementwise':
                            outer_product = torch.einsum('bim,bin->bimn', x1[l1], x2[l2])

                        for l3 in output_ls:
                            output[l3].append(torch.einsum('mnM,bimn->biM', self.w3j_matrices[(l1, l2, l3)], outer_product))
        

        elif self.ls_rule == 'efficient':
            pass

        for l3 in self.ls_out:
            output[l3] = torch.cat(output[l3], axis=1)

        return output
