from model.pipeline import * 

class Score_Distillation():
    def __init__(self, 
            pipe: SimpleDiffusionPipeline,
            time_step=0, 
            grad_sample_range=50, 
            grad_weight_type='uniform', 
            grad_guidance_0= 1, 
            grad_guidance_1= 1, 
            grad_sample_type='ori_step', 
            grad_batch_size=10):
        self.pipe = pipe
        self.device = str(pipe.device)
        self.grad_sample_range = grad_sample_range
        self.grad_weight_type = grad_weight_type
        self.time_step = time_step
        self.grad_guidance_0 = grad_guidance_0
        self.grad_guidance_1 = grad_guidance_1
        self.grad_sample_type = grad_sample_type
        self.grad_batch_size = grad_batch_size
        self.embed_uncond = pipe.prompt2embed('')
        self.embed_neg = pipe.prompt2embed('A doubling image, unrealistic, artifacts, distortions, unnatural blending, ghosting effects,\
            overlapping edges, harsh transitions, motion blur, poor resolution, low detail')
    
    # def reset_args(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)

    def grad_weight(self, t):
        if self.grad_weight_type == 'uniform':
            return 1
        if self.grad_weight_type == 'increase':
            return 1 - self.pipe.scheduler.alphas_cumprod[t]
        if self.grad_weight_type == 'decrease':
            return 1 / (1 - self.pipe.scheduler.alphas_cumprod[t])**0.5
    
    def grad_prepare(self, latent):
        if self.grad_sample_type == 'ori_step':
            return latent, self.time_step
        if self.grad_sample_type == 'forward_sample':
            if self.grad_sample_range == -1:
                max_t = self.pipe.scheduler.config.num_train_timesteps 
            else:
                max_t = self.time_step + self.grad_sample_range
            tt = torch.randint(self.time_step, max_t, (1,), device=self.device)
        elif self.grad_sample_type == 'back_n_forward_sample':
            if self.grad_sample_range == -1:
                max_t = self.pipe.scheduler.config.num_train_timesteps - 50 
                min_t = 50
                range_t = min(max_t - self.time_step, self.time_step - min_t)
            else:
                range_t = self.grad_sample_range
            max_t = self.time_step + range_t
            min_t = self.time_step - range_t
            tt = torch.randint(min_t, max_t, (1,), device=self.device)
        elif isinstance(self.grad_sample_type, tuple):
            min_t, max_t = self.grad_sample_type
            tt = torch.randint(min_t, max_t, (1,), device=self.device)
        else:
            raise ValueError('sample_type not recognized')
        # DDIM ODE
        alpha_t_ = self.pipe.scheduler.alphas_cumprod[self.time_step]
        ep0 = torch.randn_like(latent)
        if self.time_step==0:
            latent_ori = latent
        else:
            latent_ori = (latent - (1-alpha_t_)**0.5 * ep0) / (alpha_t_**0.5)
        latent_tt = self.pipe.scheduler.add_noise(latent_ori, ep0, tt)
        return latent_tt, tt
    
    def grad_compute(self, latent, embed_cond):
        assert latent.shape[0] == embed_cond.shape[0]
        b = latent.shape[0] 
        embed_uncond = self.embed_uncond.repeat(b, 1, 1)
        embed_neg = self.embed_neg.repeat(b, 1, 1)
        latent, t = self.grad_prepare(latent)
        grad_c, grad_d = 0, 0
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            if self.grad_guidance_0 == self.grad_guidance_1: # just a trick to save some computation
                grad_c = -self.pipe.noise_pred(latent, t, embed_cond)
                grad_d = self.pipe.noise_pred(latent, t, embed_neg)
            else:
                ep_uncond = self.pipe.noise_pred(latent, t, embed_uncond)
                if self.grad_guidance_0 > 0:
                    ep_cond = self.pipe.noise_pred(latent, t, embed_cond)
                    grad_c = ep_uncond - ep_cond
                if self.grad_guidance_1 > 0:
                    ep_neg = self.pipe.noise_pred(latent, t, embed_neg)
                    grad_d = ep_neg - ep_uncond 
        w = self.grad_weight(t)
        normalise = 1/(abs(self.grad_guidance_0)+ abs(self.grad_guidance_1)) #/////
        grad = w * normalise * (self.grad_guidance_0*grad_c + self.grad_guidance_1*grad_d)
        return grad 
    
    def grad_compute_batch(self, latents, embed_cond):
        assert latents.shape[0] == embed_cond.shape[0]
        n = latents.shape[0]
        grad_out = None
        j = 0 
        while n > 0:
            b = min(n, self.grad_batch_size)
            lats = latents[j:j+b, :,:,:]
            grad = self.grad_compute(lats, embed_cond[j:j+b,:,:])
            grad_out = grad if grad_out is None else torch.cat([grad_out, grad])
            n -= b
            j += b
        return grad_out 

