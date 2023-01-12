import numpy as np
import matplotlib.pyplot as plt
import torch

xs = torch.linspace(-16, 16, 200)
ys = torch.sigmoid(xs)
plt.plot(xs, ys)
plt.show()