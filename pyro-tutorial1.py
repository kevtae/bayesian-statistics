import torch
import pyro

# # mean zero
# loc = 0.
# # unit variance
# scale = 1.
# normal = torch.distributions.Bernoulli(loc, scale)
# x = normal.sample()  # draw a sample from N(0,1)
# y = torch.sample('head', torch.distributions.Bernoulli(0.5))
# print("sample", y)
# print("log prob", normal.log_prob(x))


def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample(
        'temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()


for _ in range(3):
    print(weather())
