
import numpy as np
import torch
from torch import nn
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *


NONLIN_TO_ACTIVATION_MODULES = {
    'swish': 'torch.nn.SiLU()',
    'sigmoid': 'torch.nn.Sigmoid()',
    'relu': 'torch.nn.ReLU()',
    'identity': 'torch.nn.Identity()'
}


class magnitudes_norm(torch.nn.Module):
    '''
    Basically just removes the norms!
    '''
    def __init__(self,
                 irreps_in: o3.Irreps,
                 return_magnitudes: bool = False,
                 eps: float = 1e-8):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.return_magnitudes = return_magnitudes
        self.eps = eps

    def forward(self, x: Dict[int, Tensor]) -> Tuple[Tensor, Dict[int, Tensor]]:
        '''
        'magnitude' is equivalent to 'component' in the e3nn naming convention.
        '''
        if self.return_magnitudes:
            magnitudes = []

        directions = {}
        for irr in self.irreps_in:
            feat = x[irr.ir.l]
            norm = feat.pow(2).sum(-1).mean(-1) # [batch]
            norm = torch.sqrt(norm + self.eps)

            directions[irr.ir.l] = feat / norm.unsqueeze(-1)
            
            if self.return_magnitudes:
                magnitudes.append(norm.squeeze())
        
        if self.return_magnitudes:
            return torch.cat(magnitudes, dim=-1), directions
        else:
            return directions


class signal_norm(torch.nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 normalization: str = 'component', # norm, component
                 affine: Optional[str] = 'per_feature', # None, unique, per_l, per_feature
                 balanced: Union[bool, float] = False,
                 eps: float = 1e-8):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.normalization = normalization
        self.affine = affine
        self.balanced = balanced
        self.eps = eps

        valid_affine_types = [None, 'unique', 'per_l', 'per_feature']
        if self.affine not in valid_affine_types:
            raise NotImplementedError('Affine of type "{}" not implemented. Implemented types include {}'.format(self.affine, valid_affine_types))
        
        valid_normalization_types = ['norm', 'component']
        if self.normalization not in valid_normalization_types:
            raise NotImplementedError('Normalization of type "{}" not implemented. Implemented types include {}'.format(self.normalization, valid_normalization_types))

        if self.balanced: # True or float
            multiplicative_value = 1.0 if isinstance(self.balanced, bool) else self.balanced
            if self.normalization == 'norm':
                self.balancing_constant = self.irreps_in.dim / multiplicative_value
            elif self.normalization == 'component':
                muls, ls = [], []
                for mul, ir in self.irreps_in:
                    muls.append(mul)
                    ls.append(ir.l)
                avg_mul = np.mean(muls)
                num_ls = len(ls)
                self.balancing_constant = float(avg_mul*num_ls) / multiplicative_value
        else:
            self.balancing_constant = 1.0
        
        # print(self.balancing_constant)

        weights = {}
        if self.affine == 'unique':
            self.weight = torch.nn.parameter.Parameter(torch.ones(1))
        
        if self.affine == 'per_l':
            for irr in self.irreps_in:
                weights[str(irr.ir.l)] = torch.nn.parameter.Parameter(torch.ones(1))
            self.weights = torch.nn.ParameterDict(weights)

            self.bias = torch.nn.parameter.Parameter(torch.zeros(1))
            
        elif self.affine == 'per_feature':
            for irr in self.irreps_in:
                weights[str(irr.ir.l)] = torch.nn.parameter.Parameter(torch.ones(irr.mul))
            self.weights = torch.nn.ParameterDict(weights)

            num_scalar = sum(mul for mul, ir in irreps_in if ir.is_scalar())
            self.bias = torch.nn.parameter.Parameter(torch.zeros((num_scalar, 1)))

    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        # compute normalization factors in a batch
        norm_factors = 0.0
        for l, feat in x.items():
            if self.normalization == 'norm':
                norm_factors += feat.pow(2).sum(-1).sum(-1)
            elif self.normalization == 'component':
                norm_factors += feat.pow(2).mean(-1).sum(-1)
            
        norm_factors = torch.sqrt(norm_factors + self.eps) / np.sqrt(self.balancing_constant)
        
        # normalize!
        output = {}
        for l in x:
            if self.affine == 'unique':
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors)) * self.weight
            if self.affine == 'per_l':
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors)) * self.weights[str(l)]
            elif self.affine == 'per_feature':
                output[l] = torch.einsum('bim,b,i->bim', x[l], torch.reciprocal(norm_factors), self.weights[str(l)])
            else: # no affine
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors))
            
            if l == 0:
                if self.affine in ['per_l', 'per_feature']:
                    output[l] += self.bias
        
        return output


# adapted from `https://github.com/NVIDIA/DeepLearningExamples/blob/master/DGLPyTorch/DrugDiscovery/SE3Transformer/se3_transformer/model/layers/norm.py`
class layer_norm_nonlinearity(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.
                 ┌──> feature_norm ──> LayerNorm() ──> Nonlinearity() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────────────┘
    """

    def __init__(self,
                 irreps_in: o3.Irreps,
                 nonlinearity: Optional[Union[nn.Module, str]] = nn.Identity(),
                 affine: bool = True,
                 normalization: str = 'component', # norm, component --> component further normalizes by 2l+1
                 eps: float = 1e-6):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        if isinstance(nonlinearity, str):
            self.nonlinearity = eval(NONLIN_TO_ACTIVATION_MODULES[nonlinearity])
        else:
            self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.eps = eps

        valid_normalization_types = ['norm', 'component']
        if self.normalization not in valid_normalization_types:
            raise NotImplementedError('Normalization of type "{}" not implemented. Implemented types include {}'.format(self.normalization, valid_normalization_types))

        self.layer_norms = nn.ModuleDict({
            str(irr.ir.l): nn.LayerNorm(irr.mul, elementwise_affine=affine)
            for irr in irreps_in
        })

    def forward(self, x: Dict[int, Tensor], *args, **kwargs) -> Dict[int, Tensor]:
        output = {}
        for l, feat in x.items():
            if self.normalization == 'norm':
                norm = feat.pow(2).sum(-1, keepdim=True)
            elif self.normalization == 'component':
                norm = feat.pow(2).mean(-1, keepdim=True)
            
            norm = torch.sqrt(norm + self.eps) # comment this out for experiment on pretrained model for CSE543 report

            new_norm = self.nonlinearity(self.layer_norms[str(l)](norm.squeeze(-1)).unsqueeze(-1))
            output[l] = new_norm * feat / norm

        return output


class batch_norm(torch.nn.Module):
    '''
    Adapted form the e3nn implementation to work with data stored in dicts.
    This code allows for a channels dimension alongside the multiplicities.
    '''
    def __init__(self,
                 irreps_in: o3.Irreps,
                 instance: bool = False,
                 layer: bool = False,
                 affine: bool = True,
                 normalization: str = 'component', # norm, component
                 reduce: str = 'mean', # only when normalizing multiple tensors at once
                 momentum: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()
        
        assert normalization in ['norm', 'component']
        assert reduce in ['mean', 'max']
        
        for irr in irreps_in:
            assert irr.ir.p == 1
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.layer = layer
        self.instance = instance
        assert not (self.instance and self.layer)
        self.affine = affine
        self.normalization = normalization
        self.reduce = reduce
        self.momentum = momentum
        self.eps = eps
        
        num_scalar = sum(mul for mul, ir in self.irreps_in if ir.is_scalar())
        num_features = self.irreps_in.num_irreps

        if self.instance or self.layer:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
        else:
            self.register_buffer('running_mean', torch.zeros(num_scalar))
            self.register_buffer('running_var', torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()
    
    def forward(self, x: Dict[int, Tensor]):
        input_type = type(x)
        assert input_type in [dict, list]
        
        output = {}
        if self.training and not (self.instance or self.layer):
            new_means = []
            new_vars = []

        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for irr_mul, ir in self.irreps_in:
            
            ## e3nn code, substituted with block right below
            # d = ir.dim
            # field = input[:, :, ix: ix + mul * d]  # [batch, sample, mul * repr]
            # ix += mul * d
            
            field = x[ir.l]
            orig_field_shape = field.shape
            if len(orig_field_shape) == 4:
                batch, sample, mul, d = orig_field_shape
            elif len(orig_field_shape) == 3:
                batch, mul, d = orig_field_shape
            else:
                print('error: len(orig_field_shape) = %d' % (len(orig_field_shape)))
                print(orig_field_shape)
            if irr_mul != mul:
                print(irr_mul)
                print(mul)
                print(orig_field_shape)
                print(x.keys())
                print(x[1].shape)
                print(x[4].shape)
                print(x[10].shape)
            assert irr_mul == mul
            assert ir.dim == d
            ix += mul * d
            
            # [batch, sample, mul, repr]
            field = field.view(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if self.training or self.instance or self.layer:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    if self.layer:
                        field_mean = field.mean(1).reshape(batch, mul).mean(-1).unsqueeze(-1)  # [batch, 1]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(self.running_mean[irm:irm + mul], field_mean)
                        )
                else:
                    field_mean = self.running_mean[irm: irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                if self.layer:
                    field = field - field_mean.reshape(-1, 1, 1, 1)
                else:
                    field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance or self.layer:
                if self.normalization == 'norm':
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == 'component':
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == 'mean':
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == 'max':
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                if not self.instance and not self.layer:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(self._roll_avg(self.running_var[irv: irv + mul], field_norm))
            else:
                field_norm = self.running_var[irv: irv + mul]
            irv += mul

            if self.layer:
                field_norm = (field_norm + self.eps).pow(0.5) # [batch, mul] --> finish l2-norm
                field_norm = field_norm.pow(2).mean(-1) # [batch] --> Mean square
                field_norm = (field_norm + self.eps).pow(-0.5).unsqueeze(-1) # [batch, 1] --> Root
            else:
                field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw: iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                bias = self.bias[ib: ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            ## e3nn code, substituted with block right below
            # fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

            output[ir.l] = field.view(*orig_field_shape)

        if ix != self.irreps_in.dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(self.irreps_in.dim, ix)
            raise AssertionError(msg)

        if self.training and not (self.instance or self.layer):
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        if self.training and not (self.instance or self.layer):
            torch.cat(new_means, out=self.running_mean)
            torch.cat(new_vars, out=self.running_var)

        return output