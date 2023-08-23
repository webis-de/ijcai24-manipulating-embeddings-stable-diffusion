import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
import matplotlib.pyplot as plt
import os

seed = 417016

target_seed = 683395
dim = 512

device = 'cuda'
ldm = StableDiffusion(device=device)


prompt = 'Single Color Ball'
prompt2 = 'Blue Single Color Ball'

def plot_scores(scores, file_path, y_label='Score', x_label='Image'):
    """
    Create a line plot of aesthetic scores and save the image to the specified directory.

    Parameters:
    scores (list): A list of aesthetic scores to plot.
    save_dir (str): The directory to save the image to.
    """
    plt.plot(scores)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_path)


def compute_dist_metric(target_latents, latents):
    score = 10000 * torch.nn.functional.cosine_similarity(
        target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
        latents.flatten(start_dim=1, end_dim=-1).to(torch.float64), dim=-1)
    return score

class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_condition, target_latents, target_init_latents, val):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = condition[:, 1:, :]
        self.target_init_latents = target_init_latents
        self.target_condition = target_condition[:, 1:, :]
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.latents = None
        self.target_latents = target_latents
        self.alpha = torch.nn.Parameter(torch.tensor(-5.))
        self.default_std = torch.std(condition[:, 1:, :])
        self.default_mean = torch.mean(condition[:, 1:, :])
        self.val = val

    def get_text_embedding(self):
        condition = ldm.slerp(self.condition, self.target_condition, torch.sigmoid(self.alpha))
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), condition), dim=1)
        return torch.cat([self.uncondition, cond])

    def forward(self):
        ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim, seed=seed, return_latents=True,
                            keep_init_latents=False)

        ldm.initial_latents = ldm.slerp(self.target_init_latents, ldm.initial_latents, self.val)



        latents = ldm.embedding_2_img(self.get_text_embedding(),  dim=dim,
                                      return_pil=False,
                                      return_latents=True, keep_init_latents=True)

        score = compute_dist_metric(self.target_latents, latents)

        return score

    def get_optimizer(self, eta):
        return AdamOnLion(
            params=self.parameters(),
            lr=eta,
            eps=0.001,
        )


if __name__ == '__main__':

    with torch.no_grad():
        latents_list = list()
        target_latents = ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim,
                                             seed=target_seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)



    for eta in [0.1]:
        os.makedirs(f'./output/universal_embeddings/interpolation/{eta}', exist_ok=True)
        val = 0.01



        gd = GradientDescent(
            ldm.text_enc([prompt]),
            ldm.text_enc([prompt2]),
            target_latents,
            target_init_latents,
            val
        )

        optimizer = gd.get_optimizer(eta)

        interpolation_value = [-5.]
        values = []

        for i in range(100):
            optimizer.zero_grad()
            score = gd.forward()
            loss = -score
            loss.backward(retain_graph=True)
            optimizer.step()
            interpolation_value.append(gd.alpha.item())
            values.append([val, gd.alpha.item()])

            val = val + 0.0099
            print('update initial latents')
            print(val)
            gd.val = val

            #pil_img = ldm.embedding_2_img('', gd.get_text_embedding(), save_img=False,
            #                                                            dim=dim, return_pil=True,
            #                                                            return_latents=False,
            #                              keep_init_latents=False,
            #                              seed=seed)
#
            #pil_img.save(f'output/interpolation/{eta}/{dir_num}/A_{i}_{prompt[0:25]}_{round(score.item(), 3)}_{round(val, 2)}.jpg')
#

        plot_scores(interpolation_value, f'output/universal_embeddings/interpolation/{eta}/interpolation_values.jpg',
                    x_label='Iterations',
                    y_label='alpha')
        plt.clf()
        print(values)