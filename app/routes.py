from app import app
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import os
from flask import jsonify, make_response

dirname = os.path.dirname(__file__)


class BraitCnn(nn.Module):
    def __init__(self):
        super(BraitCnn, self).__init__()
        self.brait1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.01)
        )
        self.brait2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.01)
        )
        self.brait3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.Linear(100, 26)
        )

    def forward(self, x):
        y = F.relu(self.brait1(x))
        y = F.relu(self.brait2(y))

        # flatten
        y = y.view(-1, 32 * 7 * 7)
        y = F.relu(self.brait3(y))

        return y


def BraitPrediction(img_path):
    model = BraitCnn()
    model_path = filename = os.path.join(dirname, 'ml-model/BRAIT_PYTORCH.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # prepocess image inputan convert jadi RGB
    image = Image.open(img_path)
    image = image.convert('RGB')

    # segmentasi image inputan
    width, height = image.size  # mengambil ukuran size image
    jumlah_segment = round(width / height / 0.78)  # menentukan jumlah segment huruf braille
    print(jumlah_segment)
    segment = width / jumlah_segment
    print(segment)

    tamp = []
    for i in range(0, jumlah_segment):
        cropped = image.crop((i * segment, 0, (i + 1) * segment, height))
        cropped = np.array(cropped)
        cropped = cv2.resize(cropped, (28, 28))
        cropped = cropped.astype(np.float32) / 255.0
        cropped = torch.from_numpy(cropped[None, :, :, :])
        cropped = cropped.permute(0, 3, 1, 2)
        predicted_tensor = model(cropped)
        _, predicted_letter = torch.max(predicted_tensor, 1)
        tamp.append(chr(97 + predicted_letter))

    return tamp


@app.route('/')
@app.route('/index')
def index():
    filename = os.path.join(dirname, 'image/imageTest1.jpeg')
    brait_prediction_result = BraitPrediction(filename)
    text = ''.join(brait_prediction_result)
    return make_response(jsonify(
        {
            "succes": True,
            "message": "succes translate braile image",
            "data": {
                "text": text
            }
        }
    ), 200)
