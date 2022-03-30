from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats


class ArmDist(ABC):

    @abstractmethod
    def sample(self):
        return NotImplementedError

    @abstractmethod
    def get_mean(self):
        return NotImplementedError


class BernoulliDist(ArmDist):
    NAME = "Bernoulli"

    def __init__(self, theta):
        super(BernoulliDist, self).__init__()

        assert 1 >= theta >= 0
        self.theta = theta

    def sample(self):
        return np.random.binomial(n=1, p=self.theta)

    def get_mean(self):
        return self.theta


class GaussianDist(ArmDist):
    NAME = "Gaussian"

    def __init__(self, mu):
        super(GaussianDist, self).__init__()
        self.mu = mu

    def sample(self):
        return np.random.normal(loc=self.mu, scale=1)

    def get_mean(self):
        return self.mu


class YahtzeeDist(ArmDist):
    NAME = "Yahtzee"

    def __init__(self, mu):
        super(YahtzeeDist, self).__init__()

        assert 1 >= mu >= 0
        self.mu = mu
        self._noise_dist = stats.truncnorm(-0.08 / 0.05, 0.08 / 0.05, loc=0.0, scale=0.05)

    def sample(self):
        return self.mu + self._noise_dist.rvs(1)[0]

    def get_mean(self):
        return self.mu


class DistributionFactory:
    name_to_dist = {YahtzeeDist.NAME: YahtzeeDist, BernoulliDist.NAME: BernoulliDist, GaussianDist.NAME: GaussianDist}

    @staticmethod
    def get_dist(name: str, param):
        return DistributionFactory.name_to_dist[name](param)
