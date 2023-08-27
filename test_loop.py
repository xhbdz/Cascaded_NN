import torch
import numpy as np
def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    x_sum , y_sum= 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            x_sum += ((pred-y)**2).sum()
            y_sum += (y**2).sum()


    NMSE = 10 * np.log10(x_sum / y_sum)
    return NMSE