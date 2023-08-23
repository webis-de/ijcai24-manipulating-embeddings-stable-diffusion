import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
import os
import re
import numpy as np
import matplotlib.pyplot as plt


seeds = [
    952012, 456825, 15513, 514917, 313354, 919728, 915611, 953840, 978214, 688244,
    952561, 437443, 850810, 710085, 279155, 784669, 2258, 360058, 970741, 126198,
    562885, 896353, 724092, 401237, 134930, 944704, 707118, 723123, 510649, 92071,
    34158, 937241, 9330, 550112, 588423, 995257, 942594, 900060, 186981, 607337,
    289969, 658329, 174702, 101057, 958738, 504677, 202246, 266928, 944759, 135069,
    100, 1332, 222261, 871288, 370813, 752801, 23916, 935806, 354007, 662243,
    543920, 205620, 635868, 329084, 683395
]

prompt = 'highly detailed photoreal eldritch biomechanical rock monoliths, stone obelisks, aurora borealis, psychedelic'

dim = 512
device = 'cuda'
ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb

def embeddings_to_images():
    for seed in seeds:
        if not os.path.exists(f'output/metric_generalization/{prompt[0:45].strip()}/image_{seed}'):
            os.makedirs(f'output/metric_generalization/{prompt[0:45].strip()}/image_{seed}')
        for i in range(300):
            embedding = torch.load(f'./output/metric_generalization/{prompt[0:45].strip()}/embeddings/{i}_{prompt[0:45].strip()}.pt')

            latents = ldm.embedding_2_img('', embedding, dim=dim, seed=seed, return_pil=False, keep_init_latents=False)
            image = ldm.latents_to_image(latents, return_pil=False)

            image = preprocess(image)
            image_embedding = aesthetic_predictor.clip.encode_image(image).float()
            image_embedding = aesthetic_predictor.get_features(image_embedding, image_input=False)
            score = aesthetic_predictor.mlp(image_embedding).squeeze()
            pil_image = ldm.latents_to_image(latents)[0]
            pil_image.save(
                f'output/metric_based/{prompt[0:45].strip()}/image_{seed}/{i}_{prompt[0:45].strip()}_{round(score.item(), 4)}.jpg')



def create_confidence_interval_plot(base_dir: str) -> dict:
    # Regular expression pattern to match the directory name structure
    dir_pattern = re.compile(r'image_(\d+)')

    results = {}
    results_smoothed = {}
    interval_dict = {}

    dirs = os.listdir(base_dir)

    # Iterate over items in the base directory
    for item in dirs:
        full_path = os.path.join(base_dir, item)

        # Check if it's a directory and if its name matches the directory structure
        if os.path.isdir(full_path) and dir_pattern.match(item):
            dir_number = int(dir_pattern.match(item).group(1))

            results[dir_number] = []

            # Iterate over items inside the directory
            files = os.listdir(full_path)
            sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]))
            for sub_item in sorted_files:
                file_number = float(sub_item.split('_')[-1].split('.jpg')[0])
                results[dir_number].append(file_number)

            lines = results[dir_number]
            results_smoothed[dir_number] = lines
            interval_dict = {}

    min_values = []
    max_values = []
    for i in range(len(lines)):
        interval_dict[i] = list()
        for key in results_smoothed:
            try:
                interval_dict[i].append(results_smoothed[key][i])
            except: continue

    means = [np.mean(interval_dict[key]) for key in interval_dict]
    std_devs = [np.std(interval_dict[key]) for key in interval_dict]

    means_minus_std_dev = [mean - std_dev for mean, std_dev in zip(means, std_devs)]
    means_plus_std_dev = [mean + std_dev for mean, std_dev in zip(means, std_devs)]

    keys = list(interval_dict.keys())

    for i in range(len(keys)):
        max_values.append(np.max(interval_dict[keys[i]]))
        min_values.append(np.min(interval_dict[keys[i]]))

    x_values = range(len(means))

    plt.figure()
    plt.plot(x_values, means, color='blue', label='Mean')
    plt.fill_between(x_values, means_minus_std_dev, means_plus_std_dev, color='blue', alpha=0.2,
                     label='Confidence Interval')

    plt.xlabel("Iterations")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{prompt[0:45].strip()}.pdf', format='pdf')


def condition_to_image(path):
    condition = torch.load(path)
    uncond = ldm.text_enc([""])
    embedding = torch.cat([uncond, condition])
    pil_img = ldm.embedding_2_img('', embedding, dim=dim, seed=1332, return_pil=True, keep_init_latents=False)
    pil_img.save(
        f'output/metric_generalization/highly detailed photoreal eldritch biomechani.jpg')


embeddings_to_images()

base_directory = './output/metric_generalization/highly detailed photoreal eldritch biomechani'
create_confidence_interval_plot(base_directory)


