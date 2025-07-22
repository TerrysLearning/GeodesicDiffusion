from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPFeatureExtractor, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, DDIMScheduler
from model.utils import * 
from tqdm import tqdm


class SimpleDiffusionPipeline(StableDiffusionPipeline):
    '''
    A simple version of implementation, assume batch size = 1, gnerator = None, may change this latter 
    '''
    def __init__(self, 
            vae: AutoencoderKL, 
            text_encoder: CLIPTextModel, 
            tokenizer: CLIPTokenizer, 
            unet: UNet2DConditionModel, 
            scheduler: DDIMScheduler, 
            safety_checker: StableDiffusionSafetyChecker, 
            feature_extractor: CLIPImageProcessor, 
            image_encoder: CLIPVisionModelWithProjection = None, 
            requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder, requires_safety_checker)
        self.generator = torch.Generator(device=self.device)

    def set_seed(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.generator = generator

    # --- VAE encode decode, using batch operation will cause a little bit error, so not using it
    def img2latent(self, image):
        img_tensor = load_image(image, self.device)
        latent = self.vae.config.scaling_factor * self.vae.encode(img_tensor)['latent_dist'].mean
        return latent

    def latent2img(self, latent):
        latent = 1 / self.vae.config.scaling_factor * latent
        img_tensor = self.vae.decode(latent)['sample']    
        img = output_image(img_tensor)
        return img
    
    def img2latent_batch(self, images):
        img_tensor = load_image_batch(images, self.device)
        latents = []
        for i in range(img_tensor.shape[0]):
            latents.append(self.vae.encode(img_tensor[i:i+1,:,:,:])['latent_dist'].mean)
        return self.vae.config.scaling_factor * torch.cat(latents, dim=0)
    
    def latent2img_batch(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        img_tensors = []
        for i in range(latents.shape[0]):
            img_tensor = self.vae.decode(latents[i:i+1,:,:,:])['sample']
            img_tensors.append(img_tensor)
        img_tensors = torch.cat(img_tensors, dim=0)
        imgs = output_image_batch(img_tensors)
        return imgs
    
    # --- get prompt embedding from text
    def prompt2embed(self, prompt_text):
        token = self.tokenizer(prompt_text, max_length=self.tokenizer.model_max_length,
            return_tensors="pt", padding="max_length",truncation=True).input_ids
        prompt_embed = self.text_encoder(token.to(self.device))[0]
        return prompt_embed
    
    # --- latent forward and backward process
    def latent_forward(self, latent, noise_level=1, ep0=None):
        t = self.get_t(noise_level, return_single=True)
        if t.item() == 0:
            return latent
        if ep0 is not None:
            noise = ep0
        else:
            noise = torch.randn(latent.shape, device=self.device, generator=self.generator)
        latent = self.scheduler.add_noise(latent, noise, t)
        return latent

    def latent_backward(self, latent, prompt_embed, noise_level=1, guidance_scale=0, eta=0.0):
        assert (guidance_scale > 0 and prompt_embed.shape[0] == 2*latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed.shape[0] == latent.shape[0])
        time_steps = self.get_t(noise_level)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta=eta)
        for t in tqdm(time_steps):          
            if guidance_scale > 0:       
                noise_pred = self.noise_pred_cfg(latent, t, prompt_embed, guidance_scale)
            else: # if no classifier free guidance, save half of the computation 
                noise_pred = self.noise_pred(latent, t, prompt_embed)
            latent = self.scheduler.step(noise_pred, t, latent, **extra_step_kwargs).prev_sample 
        return latent 

    def latent_forward_ode(self, latent, prompt_embed, noise_level=1, guidance_scale=0): 
        assert (guidance_scale > 0 and prompt_embed.shape[0] == 2*latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed.shape[0] == latent.shape[0])
        time_steps = self.get_t(noise_level)
        time_steps = reversed(time_steps)
        for t in tqdm(time_steps): # [1, 21, 41, 61...]
            t_prev = max((t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps), 0)
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            if guidance_scale > 0:
                epsilon = self.noise_pred_cfg(latent, t, prompt_embed, guidance_scale=guidance_scale)
            else:  # if no classifier free guidance, save half of the computation 
                epsilon = self.noise_pred(latent, t, prompt_embed)
            x0 = (latent -  (1 - alpha_prod_t_prev)**0.5 * epsilon) / (alpha_prod_t_prev**0.5)
            latent = alpha_prod_t**0.5 * x0 + (1- alpha_prod_t)**0.5*epsilon
        return latent
    
    def batch_operation(self, func, inputs, batch_size=5):
        # The inputs if of size (b,x,x,x)
        outputs = None
        b = inputs.shape[0]
        j = 0 
        while b > 0:
            bb = min(b, batch_size)
            out = func(inputs[j:j+bb,:,:,:], bb)
            outputs = out if outputs is None else torch.cat([outputs, out], dim=0)
            j += bb
            b -= bb
        return outputs

    def latent_backward_batch(self, latents, prompt_embed, noise_level=1, guidance_scale=0, eta=0.0, batch_size=5):
        def func(x, y):
            prompt_embed_  = torch.repeat_interleave(prompt_embed, y, dim=0)
            return self.latent_backward(x, prompt_embed_, noise_level, guidance_scale, eta)
        latents = self.batch_operation(func, latents, batch_size)
        return latents
    
    def latent_forward_ode_batch(self, latents, prompt_embed, noise_level=1, guidance_scale=0, batch_size=5):
        def func(x, y):
            prompt_embed_ = torch.repeat_interleave(prompt_embed, y, dim=0)
            return self.latent_forward_ode(x, prompt_embed_, noise_level, guidance_scale)
        latents = self.batch_operation(func, latents, batch_size)
        return latents
    


    # --- noise prediction    
    def noise_pred_cfg(self, latent, t, prompt_embed, guidance_scale=0.5):
        # model prediction with classifier free guidance
        latent_model_input = torch.cat([latent] * 2) 
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)  
        save_to_npy(noise_pred_cond - noise_pred_uncond) #/////
        return noise_pred

    def noise_pred(self, latent, t, prompt_embed):
        return self.unet(latent, t, encoder_hidden_states=prompt_embed).sample
    
    # ---
    def get_t(self, noise_level, return_single=False):
        # return the time step or steps for the given noise level
        if noise_level == 0:
            if return_single:
                return torch.tensor(0, device=self.device)
            return torch.tensor([], device=self.device)
        time_steps = self.scheduler.timesteps
        time_stamp = max(int(len(time_steps) * noise_level), 1)
        t = time_steps[-time_stamp]    
        if return_single:
            return t
        return time_steps[-time_stamp:]
        

def save_to_npy(vector):
    filename = 'temp_data0.npy'
    data = np.load(filename, allow_pickle=True).tolist()
    data.append(vector.reshape(-1).cpu().numpy())
    data = np.array(data)
    np.save(filename, data)


def load_pipe(device='cuda'):
    # model_id = 'CompVis/stable-diffusion-v1-4'
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    # seed = 2000
    timesteps_inference = 50 #/////
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = SimpleDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    pipe.scheduler.set_timesteps(timesteps_inference)
    pipe.to(device)
    # pipe.set_seed(seed)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe

