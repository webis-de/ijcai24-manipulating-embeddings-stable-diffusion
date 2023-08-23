import sys
import platform
if platform.system() == "Linux":
    sys.path.append('/workspace')

import gradio as gr
import os
from ldm.stable_diffusion import StableDiffusion
import json
import torch


ldm = StableDiffusion(device='cuda')
img_dir = ''
no_of_images = 0
global_seed = None
current_image = None
current_prompt = None

image_list = [None] * 5
prompt_list = [None] * 5


def set_img_directory(prompt):
    global img_dir
    base_path = "./output/prompt_engineering/"

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    prompt = prompt.strip()  # Trim the prompt
    dirs = os.listdir(base_path)

    # Find the existing directories with the same prompt and get the maximum number used
    max_number = 0
    for directory in dirs:
        if directory.endswith(prompt):
            number = int(directory.split('_')[0])
            if number > max_number:
                max_number = number

    new_dir_name = f"{max_number + 1}_{prompt}"

    # Create the directory
    img_dir = os.path.join(base_path, new_dir_name)
    os.makedirs(os.path.join(img_dir, 'cond_binary'))


def generate_image(curr_prompt):
    embedding = ldm.get_embedding([curr_prompt])[0]
    current_image = ldm.embedding_2_img(
        '',
        embedding,
        seed=global_seed,
        return_pil=True,
        save_img=False,
        keep_init_latents=False
    )
    current_image.save(os.path.join(img_dir, f'{no_of_images}.jpg'))
    torch.save(embedding[1, :, :].unsqueeze(0), os.path.join(img_dir, 'cond_binary', f'{no_of_images}_tensor.pt'))

    return current_image


def init_pipeline_params(init_prompt, seed):
    global global_seed, current_image, current_prompt, no_of_images

    no_of_images = 0
    global_seed = int(seed)
    current_prompt = init_prompt

    set_img_directory(init_prompt)
    current_image = generate_image(init_prompt)

    text = 'Initialization completed. Switch to Image Generation Tab.'

    with open(os.path.join(img_dir, 'prompts.json'), 'w') as file:
        json.dump({0: init_prompt}, file)

    return current_image, init_prompt, text


def update_history():
    global image_list, prompt_list, no_images_list

    for i in range(len(image_list) - 1):
        image_list[len(image_list) - 1 - i] = image_list[len(image_list) - 2 - i]
        prompt_list[len(prompt_list) - 1 - i] = prompt_list[len(prompt_list) - 2 - i]
    image_list[0] = current_image
    prompt_list[0] = current_prompt
    print(image_list)
    print(prompt_list)


def update_image(prompt):
    global no_of_images, current_image, current_prompt

    no_of_images += 1
    update_history()
    no_images_list = ['# <p style="text-align: center;">' + str(no_of_images - i) + '</p>'
                      if no_of_images - i > 0 else None for i in range(5)]

    current_image = generate_image(prompt)
    current_prompt = prompt

    with open(os.path.join(img_dir, 'prompts.json'), 'r') as file:
        prompts_json = json.load(file)

    prompts_json[no_of_images] = prompt

    with open(os.path.join(img_dir, 'prompts.json'), 'w') as file:
        json.dump(prompts_json, file, indent=4)

    return current_image, \
        no_images_list[0], image_list[0], prompt_list[0], \
        no_images_list[1], image_list[1], prompt_list[1], \
        no_images_list[2], image_list[2], prompt_list[2], \
        no_images_list[3], image_list[3], prompt_list[3], \
        no_images_list[4], image_list[4], prompt_list[4]


css = '''
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #img_id [data-testid="image"] > div{min-height: 400px}
'''

with gr.Blocks(css=css) as demo:

    with gr.Tab("1. Initialization"):
        with gr.Row():
            seed = gr.Number(label="Seed", value=1332, visible=False)
            initial_prompt = gr.Textbox(label="Prompt")
        with gr.Row():
            btn_init = gr.Button("Initialize")
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():
                text = gr.Textbox(label=" ")
            with gr.Column():
                pass

    with gr.Tab("2. Image Generation"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            gr_image = gr.Image(label="Image", interactive=True, elem_id="img_id", type="pil").style(height=400)
        with gr.Row():
            btn_generate = gr.Button("Generate")


    with gr.Tab("3. History"):
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                gr.Markdown(
                    """
                    # <p style="text-align: center;">Iteration</p>
                    """)
            with gr.Column():
                gr.Markdown(
                    """
                    # <p style="text-align: center;">Image</p>
                    """)
            with gr.Column():
                gr.Markdown(
                    """
                    # <p style="text-align: center;">Prompt</p>
                    """)
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                iteration1 = gr.Markdown()
            image1 = gr.Image(interactive=True)
            prompt1 = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                iteration2 = gr.Markdown()
            image2 = gr.Image(interactive=True)
            prompt2 = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                iteration3 = gr.Markdown()
            image3 = gr.Image(interactive=True)
            prompt3 = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                iteration4 = gr.Markdown()
            image4 = gr.Image(interactive=True)
            prompt4 = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=0, min_width=100):
                iteration5 = gr.Markdown()
            image5 = gr.Image(interactive=True)
            prompt5 = gr.Markdown()

    btn_init.click(
        init_pipeline_params,
        inputs=[initial_prompt, seed],
        outputs=[gr_image, prompt, text]
    )

    btn_generate.click(
        update_image,
        inputs=[prompt],
        outputs=[
            gr_image,
            iteration1, image1, prompt1,
            iteration2, image2, prompt2,
            iteration3, image3, prompt3,
            iteration4, image4, prompt4,
            iteration5, image5, prompt5
        ]
    )


demo.launch(share=True)
