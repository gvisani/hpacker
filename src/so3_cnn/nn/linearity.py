
import torch
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *


class SO3_linearity(torch.nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 irreps_out: o3.Irreps,
                 weights_initializer: Optional[Callable] = None,
                 scale_for_weights_init: float = 1.0,
                 bias: bool = False):

        '''
            We assume irreps_in and irreps_out are in sorted-and-simplified form (i.e. sorted l-values, no repetition of same l-values)
            
            We assume that there is only one parity for each l-value
            
            If irreps_in and irreps_out each have some irreps that the other does not have, then the intersection is taken
            and only the intersection l-values are returned.

            TODO: add L1 regularization layer if performance is not good
                Stackoverflow page explaining L1 vs. L2 in Pytorch: https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
                Note: "L1 regularization is not implemented as it does not actually induce sparsity as understood by weights being equal to zero."
        '''
        super().__init__()

        # determine the valid l-values to use
        self.ls = sorted(list(set(irreps_in.ls).intersection(set(irreps_out.ls))))

        # determine whether to apply bias (bias is requested and l=0 exists)
        self.bias = bias and (0 in self.ls)

        # filter out irreps_in and irreps_out
        self.irreps_in = o3.Irreps([irr for irr in irreps_in if irr.ir.l in self.ls])
        self.irreps_out = o3.Irreps([irr for irr in irreps_out if irr.ir.l in self.ls])

        # extract dimensions of in and out irreps for each l-value
        self.in_muls = {}
        for irr in self.irreps_in:
            if irr.ir.l in self.ls:
                self.in_muls[irr.ir.l] = irr.mul
        
        self.out_muls = {}
        for irr in self.irreps_out:
            if irr.ir.l in self.ls:
                self.out_muls[irr.ir.l] = irr.mul
        
        # weights initializer defaults to xavier (glorot) uniform
        if weights_initializer is None:
            weights_initializer = torch.nn.init.xavier_uniform_

        # create matrix for each l-value
        weights = {}
        for l in self.ls:
            weights[str(l)] = torch.nn.Parameter(weights_initializer(torch.zeros([self.in_muls[l], self.out_muls[l]]), gain=scale_for_weights_init))
        self.weights = torch.nn.ParameterDict(weights)

        if self.bias:
            # bias for the linear layer
            self.bias_params = torch.nn.Parameter(torch.zeros([self.out_muls[0], 1]))

    # @profile
    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output = {}
        for l in self.ls:
            x_l = x[l]
            
            out = torch.einsum('ij,bim->bjm', self.weights[str(l)], x_l)

            if l == 0 and self.bias:
                out += self.bias_params
            
            output[l] = out
        
        return output