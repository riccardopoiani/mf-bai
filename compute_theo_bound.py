import pickle
import numpy as np

from environments.multi_fidelity_env import MultiFidelityEnvironmentConfiguration
from utils.math import conf_interval
from utils.run_utils import read_cfg
from visualization.results import ResultItem


def get_complexity(sample: ResultItem):
    return sum([arr.sum() for arr in sample.cost_per_arm_and_fidelity])


def get_complexity_by_phase(sample: ResultItem, phase: int):
    return sample.cost_per_arm_and_fidelity[phase].sum()


def compute_theoretical_bound(config):
    cfg = read_cfg(config)
    cfg = MultiFidelityEnvironmentConfiguration(**cfg)

    delta = cfg.get_delta()

    bound_val = 0
    type_b = 4
    c_mul = 128

    for d in delta:
        if d != 0:
            fid_mi = None
            for m in range(cfg.m_fidelity):
                all_vals = [type_b * cfg.fidelity[m] + type_b * (cfg.fidelity[m] - cfg.fidelity[k]) * np.sqrt(
                    cfg.costs[m]) / (
                                    np.sqrt(cfg.costs[k]) - np.sqrt(cfg.costs[m]))
                            for k in range(m + 1, cfg.m_fidelity)]
                curr_th = max(all_vals)

                if d >= 4 * cfg.fidelity[m] + curr_th:
                    fid_mi = m
                    break

            # Main term
            main_term = c_mul * cfg.costs[fid_mi] / ((d - type_b * cfg.fidelity[fid_mi]) ** 2)

            # Log term
            log_term = np.log(
                (c_mul * cfg.n_arms * cfg.m_fidelity) / ((d - type_b * cfg.fidelity[fid_mi]) ** 2 * 0.001))

            # Overhead term
            ovh_term = 0
            for m in range(fid_mi):
                all_vals = [type_b * cfg.fidelity[m] + type_b * (cfg.fidelity[m] - cfg.fidelity[k]) * np.sqrt(
                    cfg.costs[m]) / (
                                    np.sqrt(cfg.costs[k]) - np.sqrt(cfg.costs[m]))
                            for k in range(m + 1, cfg.m_fidelity)]
                curr_th = max(all_vals)

                ovh_main_term = c_mul * cfg.costs[m] / (curr_th ** 2)
                ovh_log_term = np.log((c_mul * cfg.n_arms * cfg.m_fidelity) / (curr_th ** 2 * 0.001))
                ovh_term += ovh_main_term * ovh_log_term

            bound_val += main_term * log_term + ovh_term

    return bound_val


if __name__ == "__main__":
    exp_name = "ya"
    # cfg_name = "configs/simulation/ablation/setting_1_Gaussian_order.yml"
    cfg_name = "configs/yahtzee/max_bound.yml"

    if exp_name == "Ab1":
        iise = "results/final/setting1/iise/results.pkl"
        discard_all = "results/final/setting1/mfe/results.pkl"
        se = "results/final/setting1/se/results.pkl"
        iise_order = "results/final/setting1/iiseorder/results.pkl"
    elif exp_name == "Ab2":
        iise = "results/final/setting2/iise/results.pkl"
        discard_all = "results/final/setting2/mfe/results.pkl"
        se = "results/final/setting2/se/results.pkl"
        iise_order = "results/final/setting2/iiseorder/results.pkl"
    elif exp_name == "ya":
        iise = "results/final/ya/iise/results.pkl"
        discard_all = "results/final/ya/mfe/results.pkl"
        se = "results/final/ya/se/results.pkl"
        iise_order = "results/final/ya/iiseorder/results.pkl"
    else:
        raise RuntimeError

    with open(iise, 'rb') as f:
        iise_x = pickle.load(f)
    with open(iise_order, 'rb') as f:
        iise_order_x = pickle.load(f)
    with open(se, 'rb') as f:
        se_x = pickle.load(f)
    with open(discard_all, 'rb') as f:
        mfe_x = pickle.load(f)

    # Standard cost complexity
    iise_cost = []
    iise_order_cost = []

    for iise_sample, iise_order_sample, se_sample, mfe_sample in zip(iise_x, iise_order_x, se_x, mfe_x):
        iise_cost.append(get_complexity(iise_sample))
        iise_order_cost.append(get_complexity(iise_order_sample))

    iise_cost = np.array(iise_cost)
    iise_order_cost = np.array(iise_order_cost)

    print(f"IISE COST: {iise_cost.mean()} +- {conf_interval(iise_cost.std(), 100)}")
    # print(f"IISE ORDER COST: {iise_order_cost.mean()} +- {conf_interval(iise_order_gain.std(), 100)}")
    print()

    # Theoretical bound
    theo_bound = compute_theoretical_bound(cfg_name)

    print(theo_bound)
    print((iise_cost.mean() + conf_interval(iise_cost.std(), 100)) / theo_bound * 100)
