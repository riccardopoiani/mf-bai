from typing import List

import yaml
import os
import random
import numpy as np

from algorithms.bai import SuccessiveElimination, MultiFidelitySuccessiveEliminationOrder, \
    MultiFidelitySuccessiveEliminationMaxBound, MultiFidelitySuccessiveEliminationMaxBoundDiscardAll
from algorithms.learn import learn
from environments.multi_fidelity_env import MultiFidelityEnvironmentConfiguration, MultiFidelityEnvironment
from visualization.results import ResultItem

algorithm_map = {SuccessiveElimination.NAME: SuccessiveElimination,
                 MultiFidelitySuccessiveEliminationOrder.NAME: MultiFidelitySuccessiveEliminationOrder,
                 MultiFidelitySuccessiveEliminationMaxBound.NAME: MultiFidelitySuccessiveEliminationMaxBound,
                 MultiFidelitySuccessiveEliminationMaxBoundDiscardAll.NAME: MultiFidelitySuccessiveEliminationMaxBoundDiscardAll}


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fix_seed(seed_val):
    if seed_val is not None:
        os.environ["PYTHONHASHSEED"] = str(seed_val)

        random.seed(seed_val)
        np.random.seed(seed_val)


def read_cfg(env_cfg_path: str):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return env_cfg


def get_algo(algo_name: str, n_arms: int, m_fidelity: int, delta: float, precisions: List[float], costs: List[float],
             variance_proxy: float, discard_all_th: float, cfg: MultiFidelityEnvironmentConfiguration):
    return algorithm_map[algo_name](n_arms, m_fidelity, delta, precisions, costs, variance_proxy, discard_all_th, cfg)


def run(run_id, seed, env_cfg, algo_name, delta, discard_all_th):
    print(f"Run {run_id} started.")

    # Fix seed
    fix_seed(seed)

    # Instantiate env and agents
    multi_fidelity_cfg = MultiFidelityEnvironmentConfiguration(**env_cfg)
    env = MultiFidelityEnvironment(multi_fidelity_cfg)
    algo = get_algo(algo_name,
                    multi_fidelity_cfg.n_arms,
                    multi_fidelity_cfg.m_fidelity,
                    delta,
                    multi_fidelity_cfg.fidelity,
                    multi_fidelity_cfg.costs,
                    multi_fidelity_cfg.variance_proxy,
                    discard_all_th,
                    multi_fidelity_cfg)

    # Learn
    best_arm = learn(algo, env)

    print(f"Run {run_id} completed.")

    # Prepare results
    return ResultItem(best_arm,
                      sum(arr.sum() for arr in multi_fidelity_cfg.compute_cost(algo.get_arm_count())),
                      multi_fidelity_cfg.compute_cost(algo.get_arm_count()))
