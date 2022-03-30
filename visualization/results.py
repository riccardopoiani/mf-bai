from typing import List
import numpy as np

from environments.multi_fidelity_env import MultiFidelityEnvironmentConfiguration


class ResultItem:
    __slots__ = ["best_arm",
                 "cost_complexity",
                 "cost_per_arm_and_fidelity"]

    def __init__(self,
                 best_arm,
                 cost_complexity,
                 cost_per_arm_and_fidelity):
        self.best_arm = best_arm
        self.cost_complexity = cost_complexity
        self.cost_per_arm_and_fidelity = cost_per_arm_and_fidelity


class ResultSummary:

    def __init__(self, res_list: List[ResultItem], multi_env_cfg: MultiFidelityEnvironmentConfiguration):
        self.env_cfg = multi_env_cfg
        self._res_list = res_list
        self._num_res = len(self._res_list)

    @property
    def num_run(self):
        return self._num_res

    def best_arm_stats(self):
        """
        :return: (percentage of right identifications)
        """
        true_best_arm = self.env_cfg.get_best_arm()
        count = 0
        for res in self._res_list:
            if res.best_arm == true_best_arm:
                count += 1
        return count / self._num_res * 100

    def cost_complexity_stats(self):
        """
        :return: (mean, std, all_vals) of cost complexity required to identify the best arm
        """
        all_vals = np.array([res.cost_complexity for res in self._res_list])
        return all_vals.mean(), all_vals.std(), all_vals

    def cost_complexity_detail_stats(self):
        """
        :return: (mean, std, all_vals). Mean and std are (num_fidelity, num_arms).
                  All runs is (num_runs, num_fid, num_arms)
        """
        all_vals = np.zeros((self._num_res, self.env_cfg.m_fidelity, self.env_cfg.n_arms))
        for i, res in enumerate(self._res_list):
            t = res.cost_per_arm_and_fidelity
            for m, elem in enumerate(t):
                all_vals[i, m, :] = elem

        return all_vals.mean(0), all_vals.std(0), all_vals

    def get_cost_complexity_by_arm_gap(self):
        """
        :return: dictionary containing for each arm (gap, tot_cost_mean, tot_cost_std)
        """
        _, _, all_vals = self.cost_complexity_detail_stats()
        sum_cmplx = all_vals.sum(1)
        deltas = self.env_cfg.get_delta()
        d = {}
        for i in range(self.env_cfg.n_arms):
            d[i] = (deltas[i], sum_cmplx.mean(0)[i], sum_cmplx.std(0)[i])
        return d

    def get_p_active_by_budget(self, max_budget: int):
        _, _, all_vals = self.cost_complexity_detail_stats()
        sum_cmplx = all_vals.sum(1)

        activity = np.zeros((self.num_run, self.env_cfg.n_arms, max_budget))
        for run_id in range(sum_cmplx.shape[0]):
            for arm in range(self.env_cfg.n_arms):
                budget_spent = int(sum_cmplx[run_id, arm])
                activity[run_id, arm, 0:budget_spent] = 1

        return activity.mean(0), activity.std(0), activity
