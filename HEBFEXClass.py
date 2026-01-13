import torch as tr


class HEBFEX:

    def __init__(self, inp_size=784, O_size=1024, inh_iter_cnt=100,):
        self.device = self.device = tr.device('cuda:0') if tr.cuda.is_available() else tr.device('cpu')

        self.i_W = tr.rand((O_size, inp_size), device=self.device) * 1e-3
        self.inh_LW = tr.rand((O_size, O_size), device=self.device)
        self.inh_LW.fill_diagonal_(0)

        print(f'{self.i_W.shape=} {self.inh_LW.shape=}')

        # attr
        self.inp_size = inp_size
        self.O_size = O_size
        self.inh_iter_cnt = inh_iter_cnt

    def train(self, inp: tr.Tensor, lr=0.01, inh_lr=0.01, ):
        pass

    def infer(self, inp: tr.Tensor,):
        O = self.i_W.mul(inp).sum(dim=1)
        print(f'{O.shape=}')

        inh_O = O.clone()

        inh_d = (1/self.inh_iter_cnt) * 0.1
        for iter in range(self.inh_iter_cnt):
            dO = self.inh_LW.mul(inh_O).sum(dim=1)

            inh_O += (dO - inh_O) * inh_d * -1
            
            inh_O.clip_(0)

            print(f'{(inh_O > inh_O.mean()).sum()}')

        return O.clone(), inh_O.clone()
