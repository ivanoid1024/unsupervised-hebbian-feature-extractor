import torch as tr

from HEBFEXClass import HEBFEX

HFX = HEBFEX()

inp = tr.rand((HFX.inp_size,), device=HFX.device)
O, inh_O = HFX.infer(inp=inp)
