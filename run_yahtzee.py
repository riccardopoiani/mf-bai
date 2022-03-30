import argparse
import os

import gym
import yaml

from utils.run_utils import mkdir_if_not_exist
from yahtzee.gym_yahtzee.envs.yahtzee_env import YahtzeeSingleEnv
from yahtzee.planner import SimplePlanner, get_precision_bound, get_action_expected_values

if __name__ == "__main__":
    # Reading command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump-dir", type=str, default="results/")
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--tot-horizon", type=int, default=13)
    parser.add_argument("--r-max", type=int, default=50)

    args = parser.parse_args()

    r_max = args.r_max
    gamma = args.gamma
    tot_horizon = args.tot_horizon

    # Fix horizon
    if tot_horizon > 13:
        all_h = [h for h in range(4, 13+1)]
    else:
        all_h = [tot_horizon]

    # Run computations
    for tot_horizon in all_h:
        # Initial conditions
        init_action_list = [31 + i for i in range(12 - tot_horizon)]
        init_dice_list = [[1, 2, 3, 4, 5] for _ in range(12 - tot_horizon)]
        init_dice_set = [1, 1, 1, 1, 1]

        # Planning (compute arm means)
        results = {}
        for horizon in range(1, tot_horizon):
            print(f"Horizon {horizon} for Tot horizon {tot_horizon}")

            # Compute action expectation
            clean_environment: YahtzeeSingleEnv = gym.make('yahtzee-single-v0', **{"init_dices_list": [],
                                                                                   "init_action_list": []})
            clean_environment.reset()
            action_expectation = get_action_expected_values(clean_environment, gamma, tot_horizon, r_max=r_max)

            env: YahtzeeSingleEnv = gym.make('yahtzee-single-v0', **{"init_dices_list": init_dice_list,
                                                                     "init_action_list": init_action_list})
            env.reset()
            env.reset_internal_state([], [])
            planner = SimplePlanner(env)
            arm_means = planner.compute_arm_means(init_dice_set, gamma, tot_horizon, horizon, r_max, action_expectation)
            results[horizon] = arm_means

        # Dump results on file
        mkdir_if_not_exist(args.dump_dir)
        with open(os.path.join(args.dump_dir, f"arm_mean_gamma_{gamma}_th_{tot_horizon}_rmax_{r_max}.yml"), 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

        # Verify bounds
        true_means = results[tot_horizon - 1]
        for h in range(1, tot_horizon):
            approximations = results[h]
            actions = list(results[h].keys())
            curr_bound = get_precision_bound(tot_horizon, h, gamma)
            for a in actions:
                assert true_means[a] - approximations[a] <= curr_bound

