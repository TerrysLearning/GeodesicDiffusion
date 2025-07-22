# This file generate initial velocity for the IVP computation
from model.text_inv import *
from model.pipeline import *

class IniVelocity():
    def __init__(self, pipe, ini_v_type='random', **kwargs):
        self.pipe = pipe 
        self.ini_v_type = ini_v_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_v0(self):
        if self.ini_v_type == 'random':
            return torch.randn(16384, device=self.pipe.device)
        if self.ini_v_type == 'given':
            # This part haven't done yet
            vs = torch.load(self.ini_v_path, weights_only=True)
            ts = torch.linspace(0, 1, len(vs), device=self.pipe.device)
            self.start_t = ts[self.ini_v_index]
            return vs[self.ini_v_index] 
