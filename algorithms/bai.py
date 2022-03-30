from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np

from environments.multi_fidelity_env import MultiFidelityEnvironmentConfiguration
from utils.math import update_mean_online, hoeffding_anytime


class BaiAlgorithm(ABC):

    def __init__(self, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
                 variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
        assert len(precisions) == m_fidelity

        self._n_arms = n_arms
        self._m_fidelity = m_fidelity
        self._active_set = set([i for i in range(self._n_arms)])
        self._delta = delta
        self._precisions = precisions
        self._costs = costs
        self._variance_proxy = variance_proxy
        self._discard_all_th = discard_all_th
        self._cfg = cfg

        self._mean_hat = [np.zeros(self._n_arms) for _ in range(m_fidelity)]
        self._arm_count = [np.zeros(self._n_arms) for _ in range(m_fidelity)]

    @abstractmethod
    def stopping_condition(self) -> bool:
        """
        :return: True if the algorithm needs to stop, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def pull_arm(self) -> ((List[int], int), int):
        """
        Returns which arm to pull at which fidelity level

        :return: (list of arm_idx to be pulled, fidelity)
        """
        raise NotImplementedError

    @abstractmethod
    def recommendation(self) -> int:
        """
        :return: which is is the best arm
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, arm_idxes: List[int], fidelity: int, rewards: List):
        raise NotImplementedError

    def get_active_set(self) -> Set:
        return self._active_set.copy()

    def get_arm_count(self) -> List:
        """
        :return: number of pulls for each arm
        """
        return self._arm_count


class SuccessiveElimination(BaiAlgorithm):
    NAME = "SuccessiveElimination"

    def __init__(self, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
                 variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
        super(SuccessiveElimination, self).__init__(n_arms, m_fidelity, delta, precisions, costs, variance_proxy,
                                                    discard_all_th, cfg)

    def stopping_condition(self) -> bool:
        # Terminates when there is a single arm within the optimal set
        if len(self.get_active_set()) == 1:
            return True
        return False

    def pull_arm(self) -> (List[int], int):
        # Pull all the active arms at the best fidelity
        return list(self.get_active_set()), self._m_fidelity - 1

    def recommendation(self) -> int:
        # Best arm identified. The method is available only when we have stopped
        assert self.stopping_condition()
        return list(self.get_active_set())[0]

    def update(self, arm_idxes: List[int], fidelity: int, rewards: List):
        # Mean and confidence intervals
        self._mean_hat[self._m_fidelity - 1][arm_idxes] = update_mean_online(
            self._arm_count[self._m_fidelity - 1][arm_idxes],
            self._mean_hat[self._m_fidelity - 1][arm_idxes],
            np.array(rewards))
        self._arm_count[self._m_fidelity - 1][arm_idxes] += 1
        conf = hoeffding_anytime(self._arm_count[self._m_fidelity - 1][arm_idxes],
                                 self._delta / self._n_arms,
                                 self._variance_proxy)

        # Eliminate arms
        max_lb = np.max(self._mean_hat[self._m_fidelity - 1][arm_idxes] - conf)
        eliminated_arms = np.where(max_lb >= self._mean_hat[self._m_fidelity - 1][arm_idxes] + conf)[0]
        eliminated_arms = [arm_idxes[e] for e in eliminated_arms]

        for a in eliminated_arms:
            self._active_set.remove(a)


class MultiFidelitySuccessiveElimination(BaiAlgorithm, ABC):

    def __init__(self, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
                 variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
        super(MultiFidelitySuccessiveElimination, self).__init__(n_arms, m_fidelity, delta, precisions, costs,
                                                                 variance_proxy, discard_all_th)
        self._curr_phase = 0

    def stopping_condition(self) -> bool:
        if len(self.get_active_set()) == 1:
            return True
        return False

    def pull_arm(self) -> ((List[int], int), int):
        # Pull all the active arms at the best fidelity
        return list(self.get_active_set()), self._curr_phase

    def recommendation(self) -> int:
        # Best arm identified. The method is available only when we have stopped
        assert self.stopping_condition()
        return list(self.get_active_set())[0]

    @abstractmethod
    def update(self, arm_idxes: List[int], fidelity: int, rewards: List):
        # Mean and confidence intervals
        self._mean_hat[self._curr_phase][arm_idxes] = update_mean_online(
            self._arm_count[self._curr_phase][arm_idxes],
            self._mean_hat[self._curr_phase][arm_idxes],
            np.array(rewards))
        self._arm_count[self._curr_phase][arm_idxes] += 1

    @abstractmethod
    def switch_phase(self):
        raise NotImplementedError

    @abstractmethod
    def compute_thresholds(self):
        raise NotImplementedError


class MultiFidelitySuccessiveEliminationOrder(MultiFidelitySuccessiveElimination):
    NAME = "MFSuccessiveEliminationOrder"

    def __init__(self, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
                 variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
        super(MultiFidelitySuccessiveElimination, self).__init__(n_arms, m_fidelity, delta, precisions, costs,
                                                                 variance_proxy, discard_all_th, cfg)
        self._curr_phase = 0
        self._thresholds = self.compute_thresholds()
        self._init_curr_phase()

    def update(self, arm_idxes: List[int], fidelity: int, rewards: List):
        super(MultiFidelitySuccessiveEliminationOrder, self).update(arm_idxes, fidelity, rewards)

        # Compute confidence intervals
        conf = hoeffding_anytime(self._arm_count[self._curr_phase][arm_idxes],
                                 self._delta / (self._n_arms * self._m_fidelity),
                                 self._variance_proxy)

        # Eliminate arms
        max_lb = np.max(self._mean_hat[self._curr_phase][arm_idxes] - conf)
        eliminated_arms = \
            np.where(max_lb >= self._mean_hat[self._curr_phase][arm_idxes] + conf + self._precisions[self._curr_phase])[
                0]
        eliminated_arms = [arm_idxes[e] for e in eliminated_arms]
        for a in eliminated_arms:
            self._active_set.remove(a)

        # Switch phase
        self.switch_phase()

    def switch_phase(self):
        curr_iter_in_phase = np.max(self._arm_count[self._curr_phase])
        conf = hoeffding_anytime(curr_iter_in_phase,
                                 self._delta / (self._n_arms * self._m_fidelity),
                                 self._variance_proxy)
        if self._thresholds[self._curr_phase] - 2 * self._precisions[self._curr_phase] >= 4 * conf:
            self._curr_phase += 1
            self._init_curr_phase()

    def compute_thresholds(self) -> List[float]:
        th = []
        for m in range(self._m_fidelity - 1):
            all_vals = [2 * self._precisions[m] + 2 * (self._precisions[m] - self._precisions[k]) * np.sqrt(
                self._costs[m]) / (
                                np.sqrt(self._costs[k]) - np.sqrt(self._costs[m]))
                        for k in range(m + 1, self._m_fidelity)]
            curr_th = max(all_vals)
            assert curr_th > 0
            th.append(curr_th)
        th.append(0)

        return th

    def _init_curr_phase(self):
        while self._curr_phase < self._m_fidelity and self._thresholds[self._curr_phase] == np.inf:
            self._curr_phase += 1


class MultiFidelitySuccessiveEliminationMaxBound(MultiFidelitySuccessiveElimination):
    NAME = "MFSuccessiveEliminationMaxBound"

    def __init__(self, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
                 variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
        super(MultiFidelitySuccessiveElimination, self).__init__(n_arms, m_fidelity, delta, precisions, costs,
                                                                 variance_proxy, discard_all_th, cfg)
        self._curr_phase = 0
        self._thresholds = self.compute_thresholds()
        self._init_curr_phase()

    def update(self, arm_idxes: List[int], fidelity: int, rewards: List):
        super(MultiFidelitySuccessiveEliminationMaxBound, self).update(arm_idxes, fidelity, rewards)

        # Compute confidence intervals
        conf = hoeffding_anytime(self._arm_count[self._curr_phase][arm_idxes],
                                 self._delta / (self._n_arms * self._m_fidelity),
                                 self._variance_proxy)

        # Eliminate arms
        max_lb = np.max(self._mean_hat[self._curr_phase][arm_idxes] - conf - self._precisions[self._curr_phase])
        eliminated_arms = \
            np.where(max_lb >= self._mean_hat[self._curr_phase][arm_idxes] + conf + self._precisions[self._curr_phase])[
                0]
        eliminated_arms = [arm_idxes[e] for e in eliminated_arms]
        for a in eliminated_arms:
            self._active_set.remove(a)

        # Switch phase
        self.switch_phase()

    def switch_phase(self):
        curr_iter_in_phase = np.max(self._arm_count[self._curr_phase])
        conf = hoeffding_anytime(curr_iter_in_phase,
                                 self._delta / (self._n_arms * self._m_fidelity),
                                 self._variance_proxy)
        if self._thresholds[self._curr_phase] - 4 * self._precisions[self._curr_phase] >= 4 * conf:
            self._curr_phase += 1
            self._init_curr_phase()

    def compute_thresholds(self) -> List[float]:
        th = []
        for m in range(self._m_fidelity - 1):
            all_vals = [4 * self._precisions[m] + 4 * (self._precisions[m] - self._precisions[k]) * np.sqrt(
                self._costs[m]) / (
                                np.sqrt(self._costs[k]) - np.sqrt(self._costs[m]))
                        for k in range(m + 1, self._m_fidelity)]
            curr_th = max(all_vals)
            assert curr_th > 0
            th.append(curr_th)
        th.append(0)

        return th

    def _init_curr_phase(self):
        while self._curr_phase < self._m_fidelity and self._thresholds[self._curr_phase] == np.inf:
            self._curr_phase += 1


class MultiFidelitySuccessiveEliminationMaxBoundDiscardAll(MultiFidelitySuccessiveEliminationMaxBound):
    NAME = "MFSuccessiveEliminationMaxBoundDiscardAll"

    def compute_thresholds(self) -> List[float]:
        th = []
        for m in range(self._m_fidelity - 1):
            # th.append(4 * self._precisions[m] + 1e-1 * self._precisions[m])
            th.append(4 * self._precisions[m] + self._discard_all_th * self._precisions[m])
        th.append(0)

        return th

    def _init_curr_phase(self):
        while self._curr_phase < self._m_fidelity and self._thresholds[self._curr_phase] == np.inf:
            self._curr_phase += 1

        while self._curr_phase < self._m_fidelity:
            applicable, time = self._cfg.get_option(self._curr_phase,
                                                    self._discard_all_th,
                                                    self.get_active_set(),
                                                    self._delta)
            if applicable:
                self._arm_count[self._curr_phase][list(self.get_active_set())] += time
                self._curr_phase += 1
            else:
                break
