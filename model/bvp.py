from model.score import * 
import os
from model.scheduler import *
from model.bisection import *
from model.text_inv import *
from model.io import *
from model.spline import *


class BVP_Optimiser():
    def __init__(self, iter_num, lr, lr_scheduler='constant', lr_divide=True):
        self.iter_num = iter_num
        self.lr_ini = lr
        self.lr_scheduler = lr_scheduler
        self.lr_divide = lr_divide

    def get_learning_rate(self, cur_iter, t):
        if self.lr_scheduler == 'constant':
            return self.lr_ini
        scale = cur_iter / self.iter_num
        if self.lr_scheduler == 'linear':
            cur_lr =  self.lr_ini * (1 - scale)
        elif self.lr_scheduler == 'cosine':
            cur_lr = self.lr_ini * 0.5 *(1 + torch.cos(torch.pi * scale))
        elif self.lr_scheduler == 'polynomial':
            cur_lr = self.lr_ini * (1 - scale)**2
        else:
            raise ValueError('lr_scheduler not recognized')
        if self.lr_divide:
            cur_lr = cur_lr / len(t)
        return cur_lr 
    

class Geodesic_BVP(Score_Distillation, 
                   BVP_Optimiser,
                   Bisection_sampler, 
                   TextInversion, 
                   BVP_IO, 
                   Ouput_Grad):
    def __init__(self, 
            pipe: SimpleDiffusionPipeline, 
            test_name, 
            imgA, 
            imgB, 
            promptA,
            promptB, 
            noise_level, 
            alpha, 
            grad_args,
            bisect_args,
            output_args,
            tv_args,
            opt_args,
            analysis_args=None,
            spline_type='spherical_cubic', 
            grad_analysis_out=True, 
            use_lerp_cond =False, 
            sphere_constraint = True
            ):
        self.imgA = imgA
        self.imgB = imgB
        self.promptA = promptA
        self.promptB = promptB
        self.test_name = test_name
        self.alpha = alpha
        self.time_step = pipe.get_t(noise_level, return_single=True)
        self.sphere_constraint = sphere_constraint
        self.use_lerp_cond = use_lerp_cond
        
        Score_Distillation.__init__(self, pipe=pipe, time_step=self.time_step, **grad_args)
        BVP_Optimiser.__init__(self, **opt_args)
        Bisection_sampler.__init__(self, **bisect_args)
        TextInversion.__init__(self, pipe=pipe, **tv_args)
        BVP_IO.__init__(self, pipe=pipe, noise_level=noise_level, imgA=imgA, imgB=imgB, use_lerp_cond=use_lerp_cond, **output_args)
        Ouput_Grad.__init__(self, grad_analysis_out, os.path.join(self.out_dir, 'grad_check.txt'))

        # set up the optimization
        self.cur_iter = 0
        self.path = dict()

        # Init the conditional embedding
        latA0 = self.pipe.img2latent(self.imgA)
        latB0 = self.pipe.img2latent(self.imgB)
        if use_lerp_cond:    
            self.embed_condA = self.text_inversion_load(self.promptA, latA0, self.test_name, 'A')
            self.embed_condB = self.text_inversion_load(self.promptB, latB0, self.test_name, 'B')
        else:
            prompt = self.promptA + ' ' + self.promptB
            lat0 = torch.cat([latA0, latB0], dim=0)
            self.embed_condAB = self.text_inversion_load(prompt, lat0, self.test_name, 'AB')

        # Init the spline 
        if use_lerp_cond:
            latA = self.forward_single(latA0, self.embed_condA)
            latB = self.forward_single(latB0, self.embed_condB)
        else:
            latA = self.forward_single(latA0, self.embed_condAB)
            latB = self.forward_single(latB0, self.embed_condAB)
        pointA = latA.reshape(-1)
        pointB = latB.reshape(-1)
        if self.sphere_constraint:
            self.radius = 0.5 * (torch.norm(pointA) + torch.norm(pointB))
            pointA = norm_fix(pointA, self.radius)
            pointB = norm_fix(pointB, self.radius)
        self.spline = Spline(spline_type)
        self.spline.fit_spline(torch.tensor([0.0,1.0]).to(self.device), torch.stack([pointA, pointB], dim=0))
        self.path[0] = pointA
        self.path[1] = pointB
        

    def bvp_gradient(self, X, V, A, t):
        lats = X.reshape(-1, 4, 64, 64)
        if self.use_lerp_cond:
            embed_cond = lerp_cond_embed(t, self.embed_condA, self.embed_condB)
        else:
            embed_cond = self.embed_condAB.repeat(lats.shape[0], 1, 1)
        d_logps = self.grad_compute_batch(lats, embed_cond) 
        d_logps = d_logps.reshape(-1,16384)
        if self.sphere_constraint:
            d_logps = o_project_(d_logps, X) 
            A = o_project_(A, X)
        V_norm2 = torch.sum(V * V, dim=-1)
        A_scaled = A / V_norm2[:,None]
        term1 = o_project_(d_logps, V)
        term2 = o_project_(A_scaled, V) * (1/self.alpha)
        grad = -(term1 + term2)
        g_n, g1_n, g2_n, g_angle = self.grad_analysis(t, self.cur_iter, term1, term2, grad)

        if g1_n < g2_n:
            # This is a heuristic, if the acceleration term too big, it will go to the wrong direction
            # Setting the learning rate to be super small can avoid this issue
            return None, g_n, g_angle
        
        return grad, g_n, g_angle
    

    def step(self):
        t_opt = self.get_control_t()
        if t_opt is None:
            return True # means the optimisation finished, strength to max
        t_opt = t_opt.to(self.device)
        X_opt = self.spline(t_opt)
        V_opt = self.spline(t_opt, 1)
        A_opt = self.spline(t_opt, 2)
        grad, g_n, g_angle = self.bvp_gradient(X_opt, V_opt, A_opt, t_opt)
        if grad is None:
            self.add_strength(None)
            self.cur_iter += 1
            return False
        cur_lr = self.get_learning_rate(self.cur_iter, t_opt)
        control_t = t_opt.detach().cpu().numpy()
        if self.cur_iter % 5 == 0:
            print('optimise {} t={} iteration: {}, loss: {}, angle: {}'.format(
                        self.test_name, control_t, self.cur_iter, g_n, g_angle))
        
        X_opt = X_opt - cur_lr * grad 
        if self.sphere_constraint:
            X_opt = norm_fix_(X_opt, torch.tensor([self.radius]*X_opt.shape[0]).to(self.device))
        self.cur_iter += 1
        for i, t in enumerate(t_opt):
            self.path[t.item()] = X_opt[i]
        t_fit = torch.tensor(sorted(self.path.keys())).to(self.device)
        X_fit = torch.stack([self.path[t.item()] for t in t_fit], dim=0)
        self.spline.fit_spline(t_fit, X_fit)

        self.add_strength(self.cur_iter)
        return False
       
    def solve(self):
        if self.use_lerp_cond:
            embed_cond_args = {'embed_cond_A': self.embed_condA, 'embed_cond_B': self.embed_condB}
        else:
            embed_cond_args = {'embed_cond': self.embed_condAB}
        self.output_bvp_sequence_if_need(0, self.spline, 'start', **embed_cond_args)
        for i in range(self.iter_num):
            finish = self.step()
            if finish or i == self.iter_num - 1:
                self.output_bvp_sequence_if_need(self.cur_iter, self.spline, 'final', **embed_cond_args)
                break
            else:
                self.output_bvp_sequence_if_need(self.cur_iter, self.spline, str(self.cur_iter), **embed_cond_args)
            torch.cuda.empty_cache()
        self.save_opt_points_if_need(self.path)
        ts = torch.linspace(0, 1, 17, device=self.device) #/////
        torch.save(self.spline(ts,1), os.path.join(self.out_dir, 'final_vs.pt')) #/////