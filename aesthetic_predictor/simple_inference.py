from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json
from torch.autograd import grad
import torch.nn.functional as F
from warnings import filterwarnings

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm


import clip

from PIL import Image, ImageFile

#####  This script will img_predict the aesthetic score for this image file:

img_path = "test.jpg"



# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, return_l2=False, axis=-1, order=2):
    l2 = torch.norm(a, dim=axis, p=order, keepdim=True)
    l2_zeros = torch.zeros_like(l2)
    l2 = torch.where(l2 == 0, l2_zeros + 1, l2)
    if return_l2:
        return l2
    return a / l2


class AestheticPredictor:

    def __init__(self, device='cuda'):
        self.embedding_dim = 768  # CLIP embedding dim is 768 for CLIP ViT L 14
        self.device = device
        self.mlp = self.initialize_mlp()
        self.clip, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def initialize_mlp(self):
        mlp = MLP(self.embedding_dim)  #
        print(os.path.abspath('./aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth'))
        # load the mlp you trained previously or the mlp available in this repo
        s = torch.load("./aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth")
        mlp.load_state_dict(s)
        mlp.to(self.device)
        mlp.eval()
        return mlp

    def encode_input(self, input, image_input = True):
        if image_input:
            input = self.preprocess(input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if not image_input:
                input = clip.tokenize([input]).to(self.device)
                return self.clip.encode_text(input)
            else:
                return self.clip.encode_image(input)

    def get_features(self, input, text_input = False, image_input = True, normalize = True):
        if text_input or image_input:
            features = self.encode_input(input, image_input)
        else:
            features = input
        if normalize:
            #features = normalized(features.cpu().detach().numpy())
            features = normalized(features)
            #features = torch.from_numpy(features).to(self.device)
            if self.device == 'cuda': features = features.type(torch.cuda.FloatTensor)
            else: features = features.type(torch.FloatTensor)
        return features

    def predict_aesthetic_score(self, input, image_input = True):
        text_input = not image_input
        features = self.get_features(input, text_input=text_input, image_input=image_input)
        prediction = self.mlp(features)

        print("Aesthetic score predicted by the mlp:")
        print(prediction)
        return prediction.item()

    def get_model_params(self):
        return self.mlp.parameters()

    def get_optimizer(self):
        return self.mlp.configure_optimizers()
