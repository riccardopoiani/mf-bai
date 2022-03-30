import argparse
import os
import pickle

import numpy as np
import yaml
from joblib import Parallel, delayed

from environments.multi_fidelity_env import MultiFidelityEnvironmentConfiguration
from utils.math import conf_interval
from utils.run_utils import read_cfg, run, mkdir_if_not_exist
from visualization.results import ResultSummary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", type=str, required=True, choices=["SuccessiveElimination",
                                                    "MFSuccessiveEliminationOrder",
                                                    "MFSuccessiveEliminationMaxBound",
                                                    "MFSuccessiveEliminationMaxBoundDiscardAll"]
    )
    parser.add_argument("--env-cfg", type=str, required=True)
    parser.add_argument("--dump-dir", type=str, default="results/")
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--discard-all-th", type=float, default=None)

    # Reading common arguments and read environment configuration
    args = parser.parse_args()
    if args.discard_all_th is not None and args.algo != "MFSuccessiveEliminationMaxBoundDiscardAll":
        raise ValueError
    env_cfg = read_cfg(args.env_cfg)

    # Launch pure-exploration
    seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
    if args.n_jobs == 1:
        results = [run(run_id=id, seed=seed, env_cfg=env_cfg,
                       algo_name=args.algo, delta=args.delta, discard_all_th=args.discard_all_th)
                   for id, seed in zip(range(args.n_runs), seeds)]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(run_id=id, seed=seed, env_cfg=env_cfg,
                         algo_name=args.algo, delta=args.delta, discard_all_th=args.discard_all_th)
            for id, seed in zip(range(args.n_runs), seeds))

    # Dump results on file
    mkdir_if_not_exist(args.dump_dir)
    with open(os.path.join(args.dump_dir, "results.pkl"), "wb") as output:
        pickle.dump(results, output)

    # Dump configuration on file
    summary = ResultSummary(results, MultiFidelityEnvironmentConfiguration(**env_cfg))
    env_cfg["run_setting"] = {}
    env_cfg["run_setting"]["name"] = args.algo
    env_cfg["run_setting"]["delta"] = args.delta
    env_cfg["run_setting"]["n_runs"] = args.n_runs
    env_cfg["run_setting"]["discard_all_th"] = args.discard_all_th

    # Add summary results to config so that it can be retrieved one-shot
    env_cfg["results"] = {}

    env_cfg["results"]["correctness"] = summary.best_arm_stats()

    tot_cost_mean, tot_cost_std, _ = summary.cost_complexity_stats()
    env_cfg["results"]["cost_complexity"] = {}
    env_cfg["results"]["cost_complexity"]["mean"] = float(tot_cost_mean)
    env_cfg["results"]["cost_complexity"]["ci"] = float(conf_interval(tot_cost_std, summary.num_run))

    cost_by_gap = summary.get_cost_complexity_by_arm_gap()
    for k, v in cost_by_gap.items():
        cost_by_gap[k] = (float(cost_by_gap[k][0]), float(cost_by_gap[k][1]), float(conf_interval(cost_by_gap[k][2], summary.num_run)))
    env_cfg["results"]["cost_by_gap"] = cost_by_gap

    detailed_cost = summary.cost_complexity_detail_stats()
    env_cfg["results"]["detailed_cost"] = {}
    for arm in range(summary.env_cfg.n_arms):
        env_cfg["results"]["detailed_cost"][arm] = {}
        for fidelity in range(summary.env_cfg.m_fidelity):
            env_cfg["results"]["detailed_cost"][arm][fidelity] = {}
            env_cfg["results"]["detailed_cost"][arm][fidelity]["mean"] = float(detailed_cost[0][fidelity, arm])
            env_cfg["results"]["detailed_cost"][arm][fidelity]["ci"] = float(
                conf_interval(detailed_cost[1][fidelity, arm],
                              summary.num_run))

    with open(os.path.join(args.dump_dir, "config.yml"), 'w') as outfile:
        yaml.dump(env_cfg, outfile, default_flow_style=False)
