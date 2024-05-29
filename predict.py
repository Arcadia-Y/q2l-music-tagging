import sys
import tempfile
from pathlib import Path
import os
import torch
import librosa
import numpy as np
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

sys.path.insert(0, "training")
from training.query2label import build_q2l

SAMPLE_RATE = 16000
DATASET = "mtat"


class Predictor(object):
    def __init__(self, model_load_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        mod = build_q2l(50, 512).to(self.device)
        self.input_length = 59049

        filename = model_load_path
        state_dict = torch.load(filename, map_location=self.device)
        if "spec.mel_scale.fb" in state_dict.keys():
            mod.spec.mel_scale.fb = state_dict["spec.mel_scale.fb"]
        mod.load_state_dict(state_dict)

        self.tags = np.load("split/mtat/tags.npy")
        self.model = mod.eval()

    def predict(self, input, output_format):
        model = self.model
        input_length = self.input_length
        signal, _ = librosa.core.load(str(input), sr=SAMPLE_RATE)
        length = len(signal)
        hop = length // 2 - input_length // 2
        print("length, input_length", length, input_length)
        x = torch.zeros(1, input_length)
        x[0] = torch.Tensor(signal[hop : hop + input_length]).unsqueeze(0)
        x = Variable(x.to(self.device))
        print("x.max(), x.min(), x.mean()", x.max(), x.min(), x.mean())
        # asdf()
        out = model(x)
        result = dict(zip(self.tags, out[0].cpu().detach().numpy().tolist()))

        if output_format == "JSON":
            print(result)
            return result

        result_list = list(sorted(result.items(), key=lambda x: x[1]))
        plt.figure(figsize=[5, 10])
        plt.barh(
            np.arange(len(result_list)), [r[1] for r in result_list], align="center"
        )
        plt.yticks(np.arange(len(result_list)), [r[0] for r in result_list])
        plt.tight_layout()

        out_path = "./result.png"
        plt.savefig(out_path)
        print("The result is stroed in", out_path)
        return out_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default="./in.mp3")
    parser.add_argument('--output_format',  type=str, default='image')
    parser.add_argument('--model_load_path',  type=str, default='./pretrained/q2l.pth')
    config = parser.parse_args()
    predictor = Predictor(config.model_load_path)
    predictor.predict(config.input, config.output_format)
