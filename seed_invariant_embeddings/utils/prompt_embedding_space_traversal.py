import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

from ldm.stable_diffusion import StableDiffusion
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import ScalarFormatter


seed = 417016
target_seed = 683395

ldm = StableDiffusion(device='cuda')

def construct_image(image_list):
    """Combine a list of images into one image."""

    # Get the dimensions of the input images
    width, height = image_list[0][0].size

    # Calculate the dimensions of the combined image
    combined_width = width * len(image_list[0])
    combined_height = height * len(image_list)

    # Create a new blank image with the combined dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the combined image
    for i, row in enumerate(image_list):
        for j, image in enumerate(row):
            combined_image.paste(image, (j * width, i * height))

    return combined_image


def load_ldm_conditions(prompt, prompt2):
    initial_condition = ldm.text_enc([prompt])[:, 1:, :]
    target_condition = ldm.text_enc([prompt2])[:, 1:, :]
    uncondition = ldm.text_enc([""])
    condition_row = uncondition[:, 0, :]
    return initial_condition, target_condition, condition_row, uncondition

def extract_latents(prompt):
    with torch.no_grad():
        ldm.embedding_2_img(ldm.get_embedding([prompt])[0],
                            seed=target_seed, return_pil=False,
                            return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)
        ldm.embedding_2_img(ldm.get_embedding([prompt])[0], seed=seed, return_latents=True,
                            keep_init_latents=False)
        latents = torch.clone(ldm.initial_latents)
    return latents, target_init_latents


def create_interpolated_image(prompt, prompt2):
    latents, target_init_latents = extract_latents(prompt)
    initial_condition, target_condition, condition_row, uncondition = load_ldm_conditions(prompt, prompt2)

    image_list = list()
    for alpha in range(-8, 9):
        image_list_row = []
        condition = ldm.slerp(target_condition, initial_condition, torch.sigmoid(torch.tensor(alpha)))
        condition = torch.cat((condition_row.unsqueeze(dim=1), condition), dim=1)
        embedding = torch.cat([uncondition, condition])
        for i in range(21):
            beta = i * 0.05
            print(beta)
            initial_latents = ldm.slerp(latents, target_init_latents, beta)
            ldm.initial_latents = initial_latents
            pil_image = ldm.embedding_2_img(embedding,
                                            return_pil=True,
                                            keep_init_latents=True
                                            )
            image_list_row.append(pil_image)
        image_list.append(list(reversed(image_list_row)))
    combined_image = construct_image(image_list)
    combined_image.save(f'./output/universal_embeddings/universal_embeddings/{seed}_{target_seed}.png')


def save_embedding_path_traversal(values):
    prompt = 'Single Color Ball'
    prompt2 = 'Blue Single Color Ball'
    create_interpolated_image(prompt, prompt2)

    # Extract x and y values from the provided data
    x_values = [item[0] for item in values]
    y_values = [item[1] for item in values]

    # Load your combined image
    constructed_image = Image.open(f'./output/universal_embeddings/{seed}_{target_seed}.png')

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(constructed_image, extent=[-0.01, 1., -8.5, 8.5])  # Using extent to define axis bounds

    plt.autoscale(enable=False)
    # Plotting the provided values as a solid line
    ax.plot(x_values, y_values, color="#e64727", linewidth=2)

    ax.axis('tight')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Configure the plot appearance
    ax.set_xlim(-0.02, 1.)
    ax.set_ylim(-8.6, 8.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")

    # After configuring the plot appearance, set the x-axis formatter:
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    fig.tight_layout()
    plt.savefig('./output/universal_embeddings/embedding_space_traversal.pdf', format="pdf")

def main():
    prompt = 'Single Color Ball'
    prompt2 = 'Blue Single Color Ball'
    create_interpolated_image(prompt, prompt2)

    values = [[0.01, -4.998259544372559], [0.0199, -4.9939470291137695], [0.0298, -4.9878950119018555], [0.0397, -4.979873180389404], [0.0496, -4.969803333282471], [0.0595, -4.958005428314209], [0.0694, -4.944397926330566], [0.07930000000000001, -4.928825378417969], [0.08920000000000002, -4.911246299743652], [0.09910000000000002, -4.89127779006958], [0.10900000000000003, -4.868973731994629], [0.11890000000000003, -4.844175338745117], [0.12880000000000003, -4.816736698150635], [0.13870000000000002, -4.786481857299805], [0.1486, -4.753331184387207], [0.1585, -4.717158794403076], [0.1684, -4.677825450897217], [0.1783, -4.635352611541748], [0.18819999999999998, -4.589503765106201], [0.19809999999999997, -4.540133476257324], [0.20799999999999996, -4.48714017868042], [0.21789999999999995, -4.430442810058594], [0.22779999999999995, -4.369941234588623], [0.23769999999999994, -4.3055243492126465], [0.24759999999999993, -4.237123012542725], [0.25749999999999995, -4.164639949798584], [0.26739999999999997, -4.0880208015441895], [0.2773, -4.0071516036987305], [0.2872, -3.922016143798828], [0.29710000000000003, -3.8326122760772705], [0.30700000000000005, -3.7389519214630127], [0.31690000000000007, -3.64106822013855], [0.3268000000000001, -3.539008617401123], [0.3367000000000001, -3.4328553676605225], [0.34660000000000013, -3.3226943016052246], [0.35650000000000015, -3.2086379528045654], [0.36640000000000017, -3.0908141136169434], [0.3763000000000002, -2.9693679809570312], [0.3862000000000002, -2.8444530963897705], [0.39610000000000023, -2.7162270545959473], [0.40600000000000025, -2.584850788116455], [0.41590000000000027, -2.4504826068878174], [0.4258000000000003, -2.3132755756378174], [0.4357000000000003, -2.173373222351074], [0.44560000000000033, -2.030880928039551], [0.45550000000000035, -1.8859360218048096], [0.46540000000000037, -1.7386285066604614], [0.4753000000000004, -1.5890835523605347], [0.4852000000000004, -1.4373724460601807], [0.49510000000000043, -1.2835447788238525], [0.5050000000000004, -1.1276252269744873], [0.5149000000000005, -0.9696260094642639], [0.5248000000000005, -0.8094947934150696], [0.5347000000000005, -0.6472261548042297], [0.5446000000000005, -0.48286759853363037], [0.5545000000000005, -0.31661149859428406], [0.5644000000000006, -0.14882931113243103], [0.5743000000000006, 0.019892653450369835], [0.5842000000000006, 0.1887640357017517], [0.5941000000000006, 0.35686415433883667], [0.6040000000000006, 0.523158848285675], [0.6139000000000007, 0.6866908073425293], [0.6238000000000007, 0.8465102314949036], [0.6337000000000007, 1.0016977787017822], [0.6436000000000007, 1.1513161659240723], [0.6535000000000007, 1.294480800628662], [0.6634000000000008, 1.4304698705673218], [0.6733000000000008, 1.5587588548660278], [0.6832000000000008, 1.67903470993042], [0.6931000000000008, 1.7912193536758423], [0.7030000000000008, 1.895399570465088], [0.7129000000000009, 1.9918113946914673], [0.7228000000000009, 2.080800771713257], [0.7327000000000009, 2.1627402305603027], [0.7426000000000009, 2.2380733489990234], [0.752500000000001, 2.307276964187622], [0.762400000000001, 2.370816946029663], [0.772300000000001, 2.429154872894287], [0.782200000000001, 2.482741355895996], [0.792100000000001, 2.5319976806640625], [0.802000000000001, 2.577338218688965], [0.8119000000000011, 2.6191458702087402], [0.8218000000000011, 2.657803535461426], [0.8317000000000011, 2.6936357021331787], [0.8416000000000011, 2.726985454559326], [0.8515000000000011, 2.758152484893799], [0.8614000000000012, 2.787405014038086], [0.8713000000000012, 2.81499981880188], [0.8812000000000012, 2.8411779403686523], [0.8911000000000012, 2.8661606311798096], [0.9010000000000012, 2.8901479244232178], [0.9109000000000013, 2.913316011428833], [0.9208000000000013, 2.935824394226074], [0.9307000000000013, 2.9578230381011963], [0.9406000000000013, 2.979447603225708], [0.9505000000000013, 3.000797986984253], [0.9604000000000014, 3.021986484527588], [0.9703000000000014, 3.043086051940918], [0.9802000000000014, 3.064178705215454], [0.9901000000000014, 3.085325002670288]]
    save_embedding_path_traversal(values)




if __name__ == "__main__":
    main()