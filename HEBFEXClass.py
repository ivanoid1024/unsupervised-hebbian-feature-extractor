import torch as tr

class HEBFEX:

    def __init__(self, inp_size=784, O_size=1024, ):
        self.device = self.device = tr.device('cuda:0') if tr.cuda.is_available() else tr.device('cpu')

        self.i_W = tr.rand((inp_size, O_size), device=self.device) + 1e-3
        self.inh_LW = tr.zeros((O_size, O_size), device=self.device) + 1e-3

        print(f'{self.i_W.shape=} {self.inh_LW.shape=}')

    def train(self):
        pass

    def infer(self):
        pass