import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
import torch

device = 'cuda'
seed = 824331

ldm = StableDiffusion(device=device)

prompt1 = "beautiful mountain landscape, lake, snow, oil painting 8 k hd"
prompt2 = "a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic " \
          "scale, insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert " \
          "bierstadt, james gurney, brian froud,"


if __name__ == '__main__':

    prompt = prompt1[0:15] + '_' + prompt2[0:15]
    prompt = prompt[0:30]

    condition1 = ldm.text_enc([prompt1], maxlen=77)
    condition2 = ldm.text_enc([prompt2], maxlen=77)
    uncond = ldm.text_enc([""], condition1.shape[1])

    cond_row = condition1[:, 0, :]


    for i in range(51):
        alpha = (i+1) * 0.02

        interpolated_cond = ldm.lerp(condition1[:, 1:, :], condition2[:, 1:, :], alpha)
        cond = torch.cat((cond_row.unsqueeze(dim=1), interpolated_cond), dim=1)
        emb = torch.cat([uncond, cond])

        pil_image = ldm.embedding_2_img(emb, seed=seed, return_pil=True)
        pil_image.save(f'output/{i + 1}_{prompt}.jpg')







