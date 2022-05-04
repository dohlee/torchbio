import torch
import torch.nn

class SeqTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, seq, n_action='zero', *args, **kwargs):
        base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        if n_action == 'zero':
            templates = torch.cat([torch.eye(4), torch.tensor([[0., 0., 0., 0.]])])
        elif n_action == 'quarter':
            templates = torch.cat([torch.eye(4), torch.tensor([[0.25, 0.25, 0.25, 0.25]])])
        else:
            raise ValueError(f'Invalid n_action: {n_action}')

        for base in seq:
            if base not in base2int:
                raise ValueError(f'Invalid base: {base}')

        x = templates[[base2int[base] for base in seq]][:, :4].T
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, seq, n_action='zero'):
        super().__init__()
        self.seq = seq
        self.n_action = n_action

    def comp(self):
        return torch.flip(self.data, dims[0])

    def rev(self):
        return torch.flip(self.data, dims=[1])
    
    def revcomp(self):
        return torch.flip(self.data, dims=[0, 1])
    
def seqtensor(seq, n_action='zero', **kwargs) -> SeqTensor:
    return SeqTensor(seq, n_action, **kwargs)
