import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
import os
import random
seed = 417016
seed2 = 683395


seed_list = [683395, 297009, 23916, 417016]

target_seed = 510675
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)

prompt = "Glass cube, sharp focus, highly detailed, 3 d, rendered, octane render"
prompt = "Single Color Ball"
#prompt = "super detailed color art, a sinthwave northern sunset with rocks on front, lake in the middle of perspective " \
#         "and mountains at background, unreal engine, retrowave color palette, 3d render, lowpoly, colorful, digital art"


def get_random_seeds(num_seeds):
    seeds = list()
    while len(seeds) < num_seeds:
        seed = random.randint(1000, 1000000)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def compute_dist_metric(target_latents, latents):
    score = 10000 * torch.nn.functional.cosine_similarity(
        target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
        latents.flatten(start_dim=1, end_dim=-1).to(torch.float64))
    return score


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_latents, target_init_latents, val):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, -1, :])
        self.target_init_latents = target_init_latents
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.latents = None
        self.target_latents = target_latents
        self.default_std = torch.std(condition[:, 1:, :])
        self.default_mean = torch.mean(condition[:, 1:, :])
        self.val = val


    def get_text_embedding(self):
        condition = self.condition.repeat(1, 76, 1)
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), condition), dim=1)
        return torch.cat([self.uncondition, cond])

    def forward(self, lat_idx, seed_batch):
        score = 0
        for j in range(len(seed_batch)):
            ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim, seed=seed_batch[j], return_latents=True,
                                keep_init_latents=False, return_latents_step=lat_idx)

            ldm.initial_latents = ldm.slerp(self.target_init_latents, ldm.initial_latents, self.val)

            latents = ldm.embedding_2_img(self.get_text_embedding(), dim=dim,
                                          return_pil=False,
                                          return_latents=True, keep_init_latents=True, return_latents_step=lat_idx)

            score = compute_dist_metric(self.target_latents, latents) + score

        return score

    def get_optimizer(self, eta):
        return AdamOnLion(
            params=self.parameters(),
            lr=eta,
            eps=0.001,
        )


if __name__ == '__main__':

    with torch.no_grad():
        target_latents = ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim,
                                             seed=target_seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)


    for eta in [0.01]:
        os.makedirs(f'./output/universal_embeddings/{prompt[0:45].strip()}/{eta}', exist_ok=True)
        val = 0.01

        gd = GradientDescent(
            ldm.text_enc([prompt]),
            target_latents,
            target_init_latents,
            val
        )

        optimizer = gd.get_optimizer(eta)
        lat_idx = 0
        for i in range(120):
            if i % 40 == 0 and i > 0:
                lat_idx += 1
                val = 0.01
                with torch.no_grad():
                    del gd.target_latents
                    gd.target_latents = ldm.embedding_2_img(
                        ldm.get_embedding([prompt])[0],
                        seed=target_seed,
                        return_pil=False,
                        return_latents_step=lat_idx,
                        return_latents=True,
                        keep_init_latents=False
                    )

            if i == 50:
                for seed in seed_list:
                    pil_img = ldm.embedding_2_img(
                        gd.get_text_embedding(),
                        seed=seed,
                        return_pil=True,
                        return_latents=False,
                        keep_init_latents=False
                    )
                    pil_img.save(f'output/universal_embeddings/{prompt[0:45].strip()}/{eta}/{seed}_50_{prompt[0:25]}_{round(score.item(), 3)}_{round(val, 2)}.jpg')


            seed_batch = get_random_seeds(3)
            optimizer.zero_grad()

            score = gd.forward(lat_idx, seed_batch)
            loss = -score
            loss.backward(retain_graph=True)
            optimizer.step()

            val = val + 0.00495
            print('update initial latents')
            print(val)
            gd.val = val

            pil_img = ldm.embedding_2_img(
                gd.get_text_embedding(),
                return_pil=True,
                return_latents=False,
                keep_init_latents=False,
                seed=417016
            )

            pil_img.save(f'output/universal_embeddings/{prompt[0:45].strip()}/{eta}/417016_{i}_{prompt[0:25]}_{round(score.item(), 3)}_{round(val, 2)}.jpg')
            del pil_img

    with torch.no_grad():
        torch.save(gd.get_text_embedding(), f'output/universal_embeddings/{prompt[0:45].strip()}/{eta}/tensor.pt')
        for seed in seed_list:
            pil_img = ldm.embedding_2_img(
                gd.get_text_embedding(),
                seed=seed,
                return_pil=True,
                return_latents=False,
                keep_init_latents=False
            )
            pil_img.save(f'output/universal_embeddings/{prompt[0:45].strip()}/{eta}/{seed}_{prompt[0:25]}_{round(score.item(), 3)}_{round(val, 2)}.jpg')