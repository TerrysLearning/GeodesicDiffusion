from model.text_inv import *
from model.pipeline import *
from model.io import *
from model.spline import *
from model.score import *
from model.ini_velocity import *

class IVP_Optimiser():
    def __init__(self, iter_num, start_t=0, end_t=1):
        self.iter_num = iter_num
        self.start_t = start_t
        self.end_t = end_t
        self.lr = (end_t - start_t) / iter_num
    
    def get_cur_t(self, cur_iter):
        return self.start_t + cur_iter * self.lr

class Geodesic_IVP(Score_Distillation,
                   IVP_Optimiser,
                   IVP_IO,
                   TextInversion, 
                   IniVelocity):
    def __init__(self, 
            pipe: SimpleDiffusionPipeline,
            img0,
            test_name,
            prompt_source,
            prompt_target,
            noise_level,
            alpha, 
            opt_args,
            grad_args,
            output_args,
            tv_args,
            ini_velocity_args,
            analysis_args=None,
            use_lerp_cond=False,
            sphere_constraint=True):
        self.img0 = img0
        self.prompt_source = prompt_source
        self.prompt_target = prompt_target
        self.alpha = alpha
        self.test_name = test_name
        self.time_step = pipe.get_t(noise_level, return_single=True)
        self.sphere_constraint = sphere_constraint
        self.use_lerp_cond = use_lerp_cond
        Score_Distillation.__init__(self, pipe=pipe, time_step=self.time_step, **grad_args)
        IVP_Optimiser.__init__(self, **opt_args)
        TextInversion.__init__(self, pipe=pipe, **tv_args)
        IVP_IO.__init__(self, pipe=pipe, noise_level=noise_level, img0=img0, **output_args)
        IniVelocity.__init__(self, pipe=pipe, **ini_velocity_args)

        self.cur_iter = 0
        lat0 = self.pipe.img2latent(self.img0)
        if use_lerp_cond:    
            self.embed_condA = self.text_inversion_load(self.prompt_source, lat0, self.test_name, 'A')
            # self.embed_condB = self.text_inversion_load(self.prompt_target, lat0, self.test_name, 'B')
            self.embed_condB = self.pipe.prompt2embed(self.prompt_target)
        else:
            prompt = self.prompt_source 
            self.embed_cond = self.text_inversion_load(prompt, lat0, self.test_name, 'AB')

        lat = self.forward_single(lat0, self.embed_cond)
        self.x = lat.reshape(-1)
        self.radius = torch.norm(self.x).item()

        self.v = self.get_v0()
        self.v_size = torch.norm(self.v).item()
        self.v = o_project(self.v, self.x)
        self.v = norm_fix(self.v, self.v_size)
       

    def rk4_inter_step(self, xv):
        '''
        Runge-Kutta 4 intermidiate step for computing F1, F2, F3, F4
        input an xv = [gamma, gamma_dot]
        return the F
        '''
        x, v = xv
        lat = x.reshape(1, 4, 64, 64)
        if self.use_lerp_cond:
            t_ = self.start_t + self.cur_iter / self.iter_num 
            embed_merge = (1-t_)* self.embed_condA + t_ * self.embed_condB
            gradient = self.grad_compute(lat, embed_merge)
        else:
            gradient = self.grad_compute(lat, self.embed_cond)
        gradient = gradient.reshape(-1)
        gradient = gradient.to(torch.float32) # I shouldn't need to do this, very weird
        gradient = o_project(gradient, x)
        a = - self.alpha * torch.dot(v, v)* o_project(gradient, v)
        if self.sphere_constraint:
            a = o_project(a, x)
        return torch.stack([v, a], dim=0)
    
    def step(self):
        xv = torch.stack([self.x, self.v], dim=0)
        k1 = self.rk4_inter_step(xv)
        k2 = self.rk4_inter_step(xv + 0.5 * self.lr * k1)
        k3 = self.rk4_inter_step(xv + 0.5 * self.lr * k2)
        k4 = self.rk4_inter_step(xv + self.lr * k3)
        xv = xv + (self.lr / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.v = xv[1]
        if self.sphere_constraint:
            self.x = norm_fix(xv[0], self.radius)
            self.v = o_project(self.v, self.x) 
        self.v = norm_fix(self.v, self.v_size) 
        print('optimise iter:', self.cur_iter)  
        self.cur_iter += 1

    def solve(self):
        print(self.start_t, self.end_t, 'uuuuuuuuuuuu') #//////
        for i in range(self.iter_num):
            self.step()
            t = self.get_cur_t(self.cur_iter)
            if self.use_lerp_cond:
                embed_cond_args = {'embed_cond_A': self.embed_condA, 'embed_cond_B': self.embed_condB}
            else:
                embed_cond_args = {'embed_cond': self.embed_cond}
            self.save_ivp_sequence_if_need(self.x, self.cur_iter, t, **embed_cond_args)
        print(f'IVP solved with {self.iter_num} steps')



        