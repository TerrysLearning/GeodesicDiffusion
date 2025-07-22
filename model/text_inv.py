from model.utils import *
from model.pipeline import *
from accelerate import Accelerator
import torch.nn.functional as F
import os 

'This file do the text inversion process'

class TextInversion():
    def __init__(self, pipe, tv_lr, tv_steps, tv_batch_size=2, tv_ckpt_folder='ckpt'):
        self.pipe = pipe
        self.tv_lr = tv_lr
        self.tv_steps = tv_steps
        self.tv_batch_size = tv_batch_size
        os.makedirs(tv_ckpt_folder, exist_ok=True)
        self.tv_ckpt_folder = tv_ckpt_folder

    def text_inversion(self, prompt, latent):
        # Do the text inversion on a single latent code
        print('optimize text embed...')
        embed_cond = self.pipe.prompt2embed(prompt)
        latent.requires_grad_(False)
        embed_cond_opt = embed_cond.clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([embed_cond_opt], lr=self.tv_lr)
        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')
        unet = accelerator.prepare_model(self.pipe.unet)
        optimizer = accelerator.prepare_optimizer(optimizer)
        if latent.shape[0]< self.tv_batch_size:
            latent = latent.repeat(self.tv_batch_size,1,1,1)
        for i in tqdm(range(self.tv_steps)):
            optimizer.zero_grad()
            indices = torch.randperm(latent.shape[0])[:self.tv_batch_size]
            lat = latent[indices]
            noise = torch.randn_like(lat)
            t = torch.randint(self.pipe.scheduler.config.num_train_timesteps, (self.tv_batch_size,), device=self.pipe.device)
            lat_t = self.pipe.scheduler.add_noise(lat, noise, t)
            model_pred = unet(lat_t, t, encoder_hidden_states=embed_cond_opt.repeat(self.tv_batch_size, 1,1)).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction='mean')
            accelerator.backward(loss)
            optimizer.step()
        embed_cond_opt.requires_grad_(False)
        return embed_cond_opt
    
    def text_inversion_load(self, prompt, latent, prefix, postfix=''):
        # Load the text inversion model
        text_ckpt_name = '{}_{}_{}_{}.pt'.format(prefix, self.tv_steps, str(self.tv_lr).replace('.',''), postfix)
        text_ckpt_path = os.path.join(self.tv_ckpt_folder, text_ckpt_name)
        print('text_ckpt_path:', text_ckpt_path)
        if os.path.exists(text_ckpt_path):
            embed_cond = torch.load(text_ckpt_path, weights_only=True).to(self.pipe.device)
        else:
            embed_cond = self.text_inversion(prompt, latent)
            torch.save(embed_cond, text_ckpt_path)
        return embed_cond
   

