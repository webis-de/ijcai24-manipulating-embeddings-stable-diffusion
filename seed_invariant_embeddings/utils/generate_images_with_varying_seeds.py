import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
ldm = StableDiffusion(device="cuda")

prompt = "Single Color Ball"


seed_list = [683395, 417016, 23916, 871288, 383124]


for seed in seed_list:
    embedding = ldm.get_embedding(prompts=[prompt])[0]
    pil_img = ldm.embedding_2_img(embedding,  seed=seed, return_pil=True, keep_init_latents=False)
    pil_img.save('./output/universal/embeddings/' + prompt[0:50].strip().replace(' |', '') + str(seed) + '.jpg')


