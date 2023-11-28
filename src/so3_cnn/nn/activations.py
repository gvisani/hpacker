NONLIN_TO_ACTIVATION_MODULES = {
    'silu': 'torch.nn.SiLU()',
    'sigmoid': 'torch.nn.Sigmoid()',
    'relu': 'torch.nn.ReLU()',
    'leaky_relu': 'torch.nn.LeakyReLU(negative_slope=0.01)',
    'identity': 'torch.nn.Identity()'
}
