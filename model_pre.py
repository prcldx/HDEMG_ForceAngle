import torch
import numpy as np
from module import LSTM, LSTMF

def model_load(params_path_a, params_path_f):
    model_a = LSTM(20, 1, 32, 1, 14)
    model_f = LSTMF(20, 1, 32, 1, 14)
    model_a.load_state_dict(torch.load(params_path_a, map_location=torch.device('cpu')))
    model_f.load_state_dict(torch.load(params_path_f, map_location=torch.device('cpu')))
    model_a.eval()
    model_f.eval()
    return model_a, model_f
