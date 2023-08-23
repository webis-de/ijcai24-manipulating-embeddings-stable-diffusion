from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as tfms


class StableDiffusion:

    def __init__(self, device="cuda"):
        self.device = device
        self.dtype = torch.float16
        if self.device == "cpu": self.dtype = torch.float
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                              num_train_timesteps=1000)
        self.scheduler.set_timesteps(51)
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                                         torch_dtype=self.dtype).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                                 torch_dtype=self.dtype).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                          torch_dtype=self.dtype).to(self.device)
        self.initial_latents = None

    def text_enc(self, prompts, maxlen=None):
        '''
        A function to take a texual prompt and convert it into embeddings
        '''
        if maxlen is None: maxlen = self.tokenizer.model_max_length
        inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        text_encoded = self.text_encoder(inp.input_ids.to(self.device))[0]
        if self.device == "cpu":
            return text_encoded.float()
        else:
            return text_encoded.half()

    def latents_to_image(self, latents, return_pil = True):
        '''
        Function to convert latents to images
        '''
        latents = (1 / 0.18215) * latents
        #with torch.no_grad():
        image = self.vae.decode(latents).sample
        if not return_pil:
            return torch.clamp(image, -1.0, 1.0)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def pil_to_latents(self, image):
        '''
        Function to convert image to latents
        '''
        init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
        init_image = init_image.to(device=self.device, dtype=self.dtype)
        init_latent_dist = self.vae.encode(init_image).latent_dist.sample() * 0.18215
        return init_latent_dist

    def load_image(self, p):
        '''
        Function to load images from a defined path
        '''
        return Image.open(p).convert('RGB').resize((512, 512))

    def get_embedding(self, prompts):
        embedding_list = list()
        for text in prompts:
            text_encoded = self.text_enc([text])

            uncond = self.text_enc([""], text_encoded.shape[1])
            emb = torch.cat([uncond, text_encoded])
            embedding_list.append(emb)

        return embedding_list

    def get_cov(self, X, Y):
        mean_X = torch.mean(X)
        mean_Y = torch.mean(Y)

        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)

        covariance = sum((X_flat - mean_X) * (Y_flat - mean_Y)) / (len(X_flat) - 1)

        return covariance

    def combine_embeddings(self, embedding1, embedding2, noise):
        X = embedding1 * (1 - noise)
        Y = embedding2 * noise

        cov = self.get_cov(X, Y)

        Z = (X + Y)
        Z = Z * torch.sqrt(torch.std(embedding1) * torch.std(embedding2)) \
            / (torch.sqrt(torch.std(X) ** 2 + torch.std(Y) ** 2 + 2 * cov) + 1e-14)
        return Z


    def lerp(self, embedding1, embedding2, noise):
        X = embedding1 * (1 - noise)
        Y = embedding2 * noise

        Z = (X + Y)
        return Z


    def slerp(self, embedding1, embedding2, val):
        embedding1 = embedding1[0]
        embedding2 = embedding2[0]
        low_norm = embedding1 / torch.norm(embedding1, dim=1, keepdim=True)
        high_norm = embedding2 / torch.norm(embedding2, dim=1, keepdim=True)
        dot = (low_norm * high_norm).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        faktor1 = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1).unsqueeze(0)
        mask = torch.isnan(faktor1)
        mean = torch.mean(faktor1[~mask])
        faktor1[mask] = mean
        faktor2 = (torch.sin(val * omega) / so).unsqueeze(1).unsqueeze(0)
        mask = torch.isnan(faktor2)
        mean = torch.mean(faktor2[~mask])
        faktor2[mask] = mean
        res = faktor1 * embedding1 + faktor2 * embedding2
        return res

    def set_initial_latents(self, dim):
        latents = torch.randn((1, self.unet.in_channels, dim // 8, dim // 8))

        return latents

    def embedding_2_img(self, emb, keep_init_latents=True, return_latents=False, return_latents_step = 0, return_pil=True, dim=512, g=7.5, seed=61582, steps=70):
        """
        Diffusion process to convert input to image
        """

        if seed: torch.manual_seed(seed)

        # Setting number of steps in scheduler
        self.scheduler.set_timesteps(steps)

        if not keep_init_latents: self.initial_latents = None

        if self.initial_latents is None: self.initial_latents = self.set_initial_latents(dim=dim)
        latents = torch.clone(self.initial_latents)

        # Adding noise to the latents
        if self.device == "cpu":
            latents = latents.to(self.device).float() * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(self.device).half() * self.scheduler.init_noise_sigma

        # Iterating through defined steps
        for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
            # We need to scale the i/p latents to match the variance
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            # Predicting noise residual using U-Net
            #if i < steps-1:
            if i != return_latents_step:
                with torch.no_grad():
                    u, t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
            else:
                u, t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

            # Performing Guidance
            pred = u + g * (t - u)

            # Conditioning  the latents
            latents = self.scheduler.step(pred, ts, latents).prev_sample
            #if return_latents and i == steps-1:
            if return_latents and i == return_latents_step:
                return latents

        if not return_pil: return latents

        pil_image = self.latents_to_image(latents)[0]
        return pil_image