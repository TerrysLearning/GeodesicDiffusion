from model.utils import *

'This file implements the bisection sampler for BVP path optimisation'

class Bisection_sampler():
    # To sample the points on the path to optimise
    def __init__(self, 
            max_strength=3, 
            random_end=True, 
            only_new_points=False, 
            bisect_interval=100):
        assert max_strength > 0 and type(max_strength) == int
        control_dict = {}
        middle_num = 0
        for i in range(1,max_strength+1):
            middle_num += (middle_num + 1)
            t = torch.linspace(0, 1, middle_num+2)
            control_dict[i] = t[1:-1]
        if only_new_points:
            # if this is true, the path only optimise the new inserted points 
            # for example, after optimising [0.5], it will only optimise [0.25 , 0.75] not [0.25, 0.5, 0.75]
            control_dict_update = {}
            for i in range(1, max_strength+1):
                if i == 1:
                    control_dict_update[i] = control_dict[i]
                else:
                    t = control_dict[i]
                    t_prev = control_dict[i-1]
                    new_p = t[torch.isin(t, t_prev, invert=True)]
                    control_dict_update[i] = new_p
            control_dict = control_dict_update
        self.max_strength = max_strength
        self.random_end = random_end
        self.control_dict = control_dict
        self.cur_strength = 1
        self.only_new_points = only_new_points
        self.bisect_interval = bisect_interval
    
    def add_strength(self, cur_iter):
        if cur_iter is None:
            self.cur_strength += 1 # directly add strength without condition
        elif cur_iter >= self.cur_strength * self.bisect_interval: 
            self.cur_strength += 1
    
    def get_control_t(self):
        if self.cur_strength <= self.max_strength:
            return self.control_dict[self.cur_strength]
        elif self.cur_strength == self.max_strength + 1  and self.random_end:
            # perturbate around the max control points
            t = self.control_dict[self.max_strength]
            t_random = t + torch.randn_like(t) * (2**(-self.max_strength-1)) 
            if self.only_new_points:
                return torch.clip(t_random, 0.0001, 0.9999)
            else:
                t_combine = torch.clip(torch.cat([t, t_random]), 0.0001, 0.9999)
                return torch.unique(torch.sort(t_combine).values)
        else:
            return None
