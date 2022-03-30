from typing import Dict, Tuple, List

import numpy as np

from environments.dist import ArmDist, DistributionFactory
from utils.math import hoeffding_anytime


class MultiFidelityEnvironmentConfiguration:
    __slots__ = [
        "n_arms",
        "m_fidelity",
        "max_fidelity",
        "order_fidelity",
        "costs",
        "theta",
        "fidelity",
        "arms",
        "distribution_type",
        "variance_proxy",
        "options"
    ]

    def __init__(self,
                 n_arms: int,
                 m_fidelity: int,
                 costs: List[float],
                 theta: List[List[float]],
                 distribution_type: str,
                 variance_proxy: float,
                 max_fidelity=None,
                 order_fidelity=None,
                 options=None
                 ):
        # General parameters
        self.n_arms = n_arms
        self.m_fidelity = m_fidelity
        self.costs = costs
        self.theta = theta
        self.max_fidelity = max_fidelity
        self.order_fidelity = order_fidelity
        self.fidelity = self.max_fidelity if self.max_fidelity is not None else self.order_fidelity
        self.distribution_type = distribution_type
        self.variance_proxy = variance_proxy
        self.options = options

        # Check that everything looks fine
        self._verify()
        self.arms = self._build_distributions()

    def _verify(self):
        # Type of assumptions employed by the algorithm
        if self.max_fidelity is not None:
            assert self.order_fidelity is None
        if self.order_fidelity is not None:
            assert self.max_fidelity is None

        # Dimensionality check
        assert len(self.costs) == self.m_fidelity
        assert len(self.fidelity) == self.m_fidelity
        assert len(self.theta) == self.m_fidelity
        for elem in self.theta:
            assert len(elem) == self.n_arms

        # Check that orders are fine
        assert self.costs == sorted(self.costs)
        assert self.fidelity == sorted(self.fidelity)[::-1]

    def _build_distributions(self) -> Dict[Tuple[int, int], ArmDist]:
        d = {}
        for m, arm_at_curr_fid in enumerate(self.theta):
            for arm, theta_arm in enumerate(arm_at_curr_fid):
                d[arm, m] = DistributionFactory.get_dist(self.distribution_type, theta_arm)
        return d

    def compute_cost(self, arm_count: List[np.ndarray]) -> List[np.ndarray]:
        """
        :param arm_count: for each fidelity, the count of each arm up so far
        :return: for each fidelity, the cost for each arm so far
        """
        return [arm_count[m] * self.costs[m] for m in range(self.m_fidelity)]

    def get_best_arm(self) -> int:
        """
        :return: idx of the best arm
        """
        m = 0
        val_m = self.arms[0, self.m_fidelity - 1].get_mean()
        for i in range(1, self.n_arms):
            if self.arms[i, self.m_fidelity - 1].get_mean() > val_m:
                val_m = self.arms[i, self.m_fidelity - 1].get_mean()
                m = i
        return m

    def get_delta(self) -> List[float]:
        mu_star = self.arms[self.get_best_arm(), self.m_fidelity - 1].get_mean()
        return [mu_star - self.arms[arm, self.m_fidelity - 1].get_mean() for arm in range(self.n_arms)]

    def get_option(self, m, th, active_arms, conf):
        assert self.max_fidelity is not None
        assert m < self.m_fidelity
        if self.options is None:
            return False, None
        if (m, th) not in self.options:
            return False, None

        for i in active_arms:
            for j in active_arms:
                if self.arms[i, m].get_mean() > self.arms[j, m].get_mean() > 0:
                    if self.arms[i, m].get_mean() - self.arms[j, m].get_mean() > 2 * self.max_fidelity[m]:
                        return False, None

        assert 4 * hoeffding_anytime(self.options[(m, th)],
                                     conf/(self.n_arms * self.m_fidelity),
                                     self.variance_proxy) <= th * self.max_fidelity[m]
        assert 4 * hoeffding_anytime(self.options[(m, th)] - 1,
                                     conf/(self.n_arms * self.m_fidelity),
                                     self.variance_proxy) > th * self.max_fidelity[m]
        return True, self.options[(m, th)]


class MultiFidelityEnvironment:

    def __init__(self, multi_fidelity_cfg: MultiFidelityEnvironmentConfiguration):
        self.cfg = multi_fidelity_cfg

    def step(self, arm_idx: List[int], fidelity: int):
        """
        :param arm_idx: list of the indexes of the arms to be pulled
        :param fidelity: fidelity at which arms will be pulled
        :return: (rewards list, total cost of interaction)
        """
        for idx in arm_idx:
            assert self.cfg.n_arms > idx >= 0
        assert self.cfg.m_fidelity > fidelity >= 0, f"Attempting to play fidelity {fidelity}"

        return [self.cfg.arms[idx, fidelity].sample() for idx in arm_idx], self.cfg.costs[fidelity] * len(arm_idx)
