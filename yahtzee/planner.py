from typing import Dict, List

from pyhtzee.utils import CATEGORY_ACTION_OFFSET

from yahtzee.gym_yahtzee.envs.yahtzee_env import YahtzeeSingleEnv


def get_costs(n_actions: int, M: int):
    return [n_actions ** h for h in range(1, M + 1)]


def get_precision_bound(tot_horizon: int, planning_steps: int, gamma: float):
    return ((gamma ** (planning_steps + 1) - gamma ** tot_horizon) / (1 - gamma)) * \
           ((1-gamma) / (1 - gamma ** tot_horizon))


def normalization(reward, gamma, max_horizon, r_max=50):
    """
    Rescale rewards

    :param reward: reward to be rescaled
    :param gamma: discount factor
    :param max_horizon: maximum horizon
    :param r_max: maximum reward
    :return: normalized reward
    """
    return reward / (r_max * ((1 - gamma ** max_horizon) / (1 - gamma)))


def get_action_expected_values(env: YahtzeeSingleEnv, gamma: float, max_horizon: int, r_max=50) -> Dict[int, float]:
    """
    Compute expectation of a given action

    :param env: environment
    :param gamma: discount factor (used to normalize rewards)
    :param max_horizon: maximum horizon (used to normalize rewards)
    :param r_max: maximum reward (used to normalize rewards)
    :return: dictionary that maps action to their expected values prior to the realization of a dice roll
    """
    expectation = {}
    for a in env.get_available_actions():
        curr_count = 0
        curr_s = 0
        if a >= CATEGORY_ACTION_OFFSET:
            for d1 in range(1, 6 + 1):
                for d2 in range(1, 6 + 1):
                    for d3 in range(1, 6 + 1):
                        for d4 in range(1, 6 + 1):
                            for d5 in range(1, 6 + 1):
                                dices = [d1, d2, d3, d4, d5]
                                env.reset_internal_state([], [])
                                env.set_dice(dices.copy())
                                _, rew, done, _ = env.step(a)
                                rew = normalization(rew, gamma, max_horizon, r_max=r_max)
                                assert not done
                                curr_s += rew
                                curr_count += 1
            expectation[a] = curr_s / curr_count
    return expectation


def get_action_value_given_dice_set(env: YahtzeeSingleEnv,
                                    action_history: List,
                                    dice_history: List,
                                    gamma: float,
                                    max_horizon: int,
                                    r_max: int,
                                    init_dice_set: List) -> Dict[int, float]:
    """
    Given an initial position, compute the value of each possible combination

    :param env: environment
    :param action_history: history of actions that lead to a given initial state
    :param dice_history: history of dices that lead to a given initial state
    :param gamma: discount factor (for reward normalization)
    :param max_horizon: maximum planning horizon (for reward normalization)
    :param r_max: maximum reward (i.e., 50)
    :param init_dice_set: a fixed dice combination
    :return: value for each of the available actions
    """
    env.reset_internal_state(dice_history, action_history)
    env.set_dice(init_dice_set)
    res = {}
    for a in env.get_available_actions():
        if a >= CATEGORY_ACTION_OFFSET:
            env.reset_internal_state(dice_history, action_history)
            env.set_dice(init_dice_set)
            _, rew, _, _ = env.step(a)
            rew = normalization(rew, gamma, max_horizon, r_max=r_max)
            res[a] = rew

    return res


class SimplePlanner:

    def __init__(self, env: YahtzeeSingleEnv):
        # Environments
        self.env = env

        # Parameters for backtracking
        self._plan_result = None
        self._action_expectation = None

    def compute_arm_means(self,
                          init_dice_set: List,
                          gamma: float,
                          tot_horizon: int,
                          horizon: int,
                          r_max: int,
                          action_expected_values: Dict[int, float]):
        self.env.reset_internal_state([], [])

        # Compute action expectation
        self._action_expectation = action_expected_values

        # Compute available action immediate values
        self.env.reset_internal_state([], [])
        self.env.set_dice(init_dice_set.copy())
        immediate_step_values = get_action_value_given_dice_set(self.env, [], [], gamma, tot_horizon, r_max,
                                                                init_dice_set)

        # Start the search
        self.env.reset_internal_state([], [])
        self.env.set_dice(init_dice_set.copy())
        available_actions = self.env.get_available_actions()
        available_actions = [a for a in available_actions if a >= CATEGORY_ACTION_OFFSET]
        self._plan_result = {}

        # Fidelity 0: immediate values (cost = number of initial actions)
        for k, v in immediate_step_values.items():
            self._plan_result[k] = v

        # Compute results
        for a in available_actions:
            curr_avail = [b for b in available_actions if b != a]
            curr_res = self.run_dfs(horizon, curr_avail, gamma)
            self._plan_result[a] += curr_res

        return self._plan_result

    def run_dfs(self, depth: int, remaining_actions: List, gamma: float):
        if depth == 0:  # Planning is over, the environment is terminated this far
            return

        rewards = []
        for a in remaining_actions:
            curr_reward = self._action_expectation[a]
            if depth > 1:
                curr_reward += self.run_dfs(depth - 1, [b for b in remaining_actions if b != a], gamma)
            rewards.append(curr_reward)

        return gamma * max(rewards)
