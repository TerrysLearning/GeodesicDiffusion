from model.utils import *
from model.scheduler import Scheduler
from torchvision import transforms
import os

'This file implements the BVP Input and Output'

class IO():
    def __init__(self,
            pipe,
            cfg_sample=0.5,
            noise_level=0.0,
            eta=0.0,
            use_neg_prompt_io = False
           ):
        self.pipe = pipe
        self.device = str(pipe.device)
        self.noise_level = noise_level
        self.cfg_sample = cfg_sample
        self.eta = eta
        self.embed_uncond = pipe.prompt2embed('')
        self.use_neg_prompt_io = use_neg_prompt_io
        self.embed_neg = pipe.prompt2embed('A doubling image, unrealistic, artifacts, distortions, unnatural blending, ghosting effects,\
            overlapping edges, harsh transitions, motion blur, poor resolution, low detail')
        
    def forward_single(self, input, embed_cond):
        # input a single image or a tensor of size (1,4,64,64)
        if isinstance(input, torch.Tensor) and input.shape == (1,4,64,64):
            lat = input
        else:
            lat = self.pipe.img2latent(input)
        if self.noise_level > 0:
            if self.cfg_sample>0:
                prompt_cfg = torch.cat([self.embed_uncond, embed_cond])
                if self.use_neg_prompt_io:
                    prompt_cfg = torch.cat([self.embed_uncond, embed_cond-self.embed_neg])
                lat = self.pipe.latent_forward_ode(lat, prompt_cfg, self.noise_level, guidance_scale=self.cfg_sample)
            else:
                lat = self.pipe.latent_forward_ode(lat, self.embed_uncond, self.noise_level, guidance_scale=0)
        return lat
    
    def backward_single(self, lat, embed_cond):
        if self.noise_level > 0:
            if self.cfg_sample>0:
                prompt_cfg = torch.cat([self.embed_uncond, embed_cond])
                if self.use_neg_prompt_io:
                    prompt_cfg = torch.cat([self.embed_uncond, embed_cond-self.embed_neg])
                lat = self.pipe.latent_backward(lat, prompt_cfg, self.noise_level, guidance_scale=self.cfg_sample, eta=self.eta)
            else:
                lat = self.pipe.latent_backward(lat, self.embed_uncond, self.noise_level, guidance_scale=0, eta=self.eta)
        img = self.pipe.latent2img(lat)
        return img
    
    def backward_multi(self, X, embed_cond):
        assert X.shape[0] == embed_cond.shape[0]
        imgs = []
        for i in range(X.shape[0]):
            img = self.backward_single(X[i].reshape(1,4,64,64), embed_cond[i:i+1,:,:])
            imgs.append(img)
        return imgs
    
    def foward_multi(self, imgs, embed_cond):
        assert len(imgs) == embed_cond.shape[0]
        lats = []
        for img in imgs:
            lat = self.forward_single(img, embed_cond)
            lats.append(lat)
        return torch.cat(lats, dim=0)


class BVP_IO(IO):
    def __init__(self,
            pipe,
            noise_level=0.0,
            cfg_sample=0.5,
            eta=0.0,
            out_dir='./',
            output_psample=True,
            output_image_num=15,
            output_start_images=False,
            output_opt_points=False,
            output_reconstruct_end=False,
            output_separate_images=True,
            use_neg_prompt_io = False,
            out_interval=-1, 
            use_lerp_cond = False,
            imgA=None,
            imgB=None):
        super().__init__(pipe, cfg_sample, noise_level, eta, use_neg_prompt_io)
        self.out_dir = out_dir
        self.output_psample = output_psample
        self.output_image_num = output_image_num
        self.output_start_images = output_start_images
        self.output_opt_points = output_opt_points
        self.output_reconstruct_end = output_reconstruct_end
        self.output_separate_images = output_separate_images
        self.out_interval = out_interval
        self.imgA = imgA
        self.imgB = imgB
        self.use_lerp_cond = use_lerp_cond
        self.out_t = torch.linspace(0, 1, self.output_image_num).to(self.device)
        assert self.output_image_num < 100
        if self.output_start_images:
            os.makedirs(os.path.join(self.out_dir, 'start_imgs'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'out_imgs'), exist_ok=True)

    def backward_bvp(self, spline, embed_cond):
        # output the images from spline and given t, embed_cond
        # consider if we want to reconstruct the end-point images
        if not self.output_reconstruct_end:
            assert self.imgA is not None and self.imgB is not None
            imgs = self.backward_multi(spline(self.out_t[1:-1]), embed_cond[1:-1,:,:])
            imgs = [self.imgA] + imgs + [self.imgB]
        else:
            imgs = self.backward_multi(spline(self.out_t), embed_cond)
        return imgs
    
    def save_bvp_sequence(self, spline, out_name, **embed_cond_args):
        # output the images from spline and given t, embed_cond, and save them
        # consider if we want to do perceptual uniform sampling, and save the images separately
        if self.use_lerp_cond:
            embed_cond = lerp_cond_embed(self.out_t, **embed_cond_args)
        else:
            embed_cond = embed_cond_args['embed_cond'].repeat(self.out_t.shape[0], 1, 1)
        imgs = self.backward_bvp(spline, embed_cond)
        if self.output_psample and out_name == 'final':
            print('perceptual uniform sampling ...')
            scheduler = Scheduler(self.pipe.device)
            imgs_pt = [transforms.ToTensor()(img).unsqueeze(0)
                             for img in imgs]
            imgs_pt = [img.to(self.device) for img in imgs_pt]
            scheduler.from_imgs(imgs_pt)
            out_t = scheduler.get_list() # start from 0 to 1
            print('p-sampled t: ',list(out_t))
            if self.use_lerp_cond:
                embed_cond = lerp_cond_embed(out_t, **embed_cond_args)[1:-1]
            else:
                embed_cond = embed_cond[1:-1,:,:]
            out_t_tensor = torch.tensor(out_t).to(self.device).clip(0,1)
            X = spline(out_t_tensor)   
            imgs[1:-1] = self.backward_multi(X[1:-1], embed_cond)
        if self.output_separate_images:
            for i, img in enumerate(imgs):
                if out_name == 'start':
                    img.save(os.path.join(self.out_dir, 'start_imgs',  f'{i:02d}.png'))
                else:
                    img.save(os.path.join(self.out_dir, 'out_imgs',  f'{i:02d}.png'))
        img_long = display_alongside(imgs)
        img_long.save(os.path.join(self.out_dir, f'long_{out_name}.png'))
        print(f'Image sequence saved to {self.out_dir}/long_{out_name}.png')
        return imgs

    def output_bvp_sequence_if_need(self, iter, spline, out_name, **embed_cond_args):
        # decide whether to save the image squence given the iteration number
        check1 = iter == 0 and self.output_start_images
        check2 = iter % self.out_interval == 0 and self.out_interval > 0 and iter > 0
        check3 = out_name == 'final'
        if check1 or check2 or check3:
            return self.save_bvp_sequence(spline, out_name=out_name, **embed_cond_args)
        return None 

    def save_opt_points_if_need(self, path):
        if self.output_opt_points:
            torch.save(path, os.path.join(self.out_dir, 'opt_points.pth'))
            print(f'Optimisation points saved to {self.out_dir}/opt_points.pth')


class Ouput_Grad():
    def __init__(self, grad_analysis_out, grad_out_txt):
        self.grad_analysis_out = grad_analysis_out
        self.grad_out_txt = grad_out_txt
        
    def grad_analysis(self, t, cur_iter, grad_term1, grad_term2, grad_all):
        # def check_grad(self, k, grad_term1, grad_term2, grad_all):
        # norms of the grad terms
        grad_term1 = grad_term1.reshape(-1, 16384)
        grad_term2 = grad_term2.reshape(-1, 16384)
        grad_all = grad_all.reshape(-1, 16384)

        cos_theta = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        cos = cos_theta(grad_term1, grad_term2)
        angle = torch.arccos(cos) * 180 / torch.pi
    
        g_norm1 = torch.norm(grad_term1, dim=-1)
        g_norm2 = torch.norm(grad_term2, dim=-1)
        g_norm_all = torch.norm(grad_all, dim=-1)
        
        g_n = torch.mean(g_norm_all).item()
        g1_n = torch.mean(g_norm1).item()
        g2_n = torch.mean(g_norm2).item()
        g_angle = torch.mean(angle).item()
        
        if not self.grad_analysis_out:
            return g_n, g1_n, g2_n, g_angle
        
        angle = [round(angle[i].item(),4) for i in range(angle.shape[0])]
        n1 = [round(g_norm1[i].item(),4) for i in range(g_norm1.shape[0])]
        n2 = [round(g_norm2[i].item(),4) for i in range(g_norm2.shape[0])]
        n_all = [round(g_norm_all[i].item(),4) for i in range(g_norm_all.shape[0])]
        t_write = [round(t[i].item(),4) for i in range(t.shape[0])]
        
        txt_file = open(self.grad_out_txt, 'a')
        txt_file.write('iter:{}\n'.format(cur_iter))
        txt_file.write('t:{}\n'.format(t_write))
        txt_file.write('g_t1:{}\n'.format(n1))
        txt_file.write('g_t2:{}\n'.format(n2))
        txt_file.write('g_al:{}\n'.format(n_all))
        txt_file.write('g_an:{}\n'.format(angle))
        txt_file.write('\n')
        txt_file.close()
        return g_n, g1_n, g2_n, g_angle



class IVP_IO(IO):
    def __init__(self,
            pipe,
            noise_level=0.0,
            cfg_sample=0.5,
            eta=0.0,
            out_dir='./',
            output_interval=10,
            output_opt_points=False,
            output_separate_images=True,
            use_lerp_cond = False,
            output_reconstruct_start=False,
            img0=None):
        super().__init__(pipe, cfg_sample, noise_level, eta)
        self.out_dir = out_dir
        self.output_interval = output_interval
        self.output_opt_points = output_opt_points
        self.output_separate_images = output_separate_images
        self.use_lerp_cond = use_lerp_cond
        self.output_reconstruct_start = output_reconstruct_start
        self.img0 = img0
        self.imgs = []
        if not self.output_reconstruct_start:
            self.imgs.append(self.img0)
            img0.save(os.path.join(self.out_dir, '0000.png')) 

    def save_ivp_sequence_if_need(self, x, cur_iter, t, **embed_cond_args):
        if self.output_reconstruct_start and cur_iter == 0:
            img = self.backward_single(x.reshape(1,4,64,64), embed_cond_args['embed_cond'])
            self.imgs.append(img)
        elif cur_iter % self.output_interval == 0 and cur_iter > 0:
            if self.use_lerp_cond:
                t = torch.tensor([t]).to(self.pipe.device)
                embed_cond = lerp_cond_embed(t, **embed_cond_args)
            else:
                embed_cond = embed_cond_args['embed_cond']
            img = self.backward_single(x.reshape(1,4,64,64), embed_cond)
            self.imgs.append(img)
        else:
            return 
        if self.output_separate_images:
            img.save(os.path.join(self.out_dir, f'{cur_iter:04d}.png'))
            print(f'Image saved to {self.out_dir}/{cur_iter:04d}.png')
        img_long = display_alongside(self.imgs)
        img_long.save(os.path.join(self.out_dir, 'all.png'))
        print(f'Image sequence saved to {self.out_dir}/all.png')