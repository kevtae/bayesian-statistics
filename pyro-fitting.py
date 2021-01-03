import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
torch.manual_seed(0)
n = 100

# Generate data from actual distribution
data = dist.Bernoulli(0.4).sample(sample_shape=(n, 1))

print(data)

# calculating prior distrubution
prior = dist.Beta(10, 10)

# for i in range(10):
#     print(prior.sample())

plt.figure(num=None, figsize=(10, 6), dpi=80)
x_range = np.linspace(0, 1, num=100)

print(x_range)

# Analytical posterior
posterior = dist.Beta(
    prior.concentration0 + data.sum(),
    prior.concentration1 + len(data) - data.sum(),
)

# print(data)
# print(data.sum())
# print(prior.concentration0, prior.concentration1)
# print(posterior.concentration0, posterior.concentration1)

y_values = torch.exp(posterior.log_prob(torch.tensor(x_range)))
plt.plot(x_range, y_values, label="posterior")

y_values = torch.exp(prior.log_prob(torch.tensor(x_range)))
plt.plot(x_range, y_values, label="prior")

plt.title("Prior belief")
plt.legend()
plt.show()
