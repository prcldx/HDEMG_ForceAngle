import torch
import numpy as np

def prediction(model_a, model_f, data):
    data = torch.tensor(data, dtype=torch.float32)#转为tensor
    with torch.no_grad():#不计算梯度，因为是预测阶段
        # if idx==1:
        #     model.begin_state(init_method='zero')
        # else:
        #     for i in range(len(model.state)):
        #         model.state[i].detach_()
        pred_a = model_a(data)  # 预测
        pred_f = model_f(data)
    pred_a = pred_a.detach().cpu().numpy()
    pred_f = pred_f.detach().cpu().numpy()
    return pred_a, pred_f
