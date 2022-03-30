from algorithms.bai import BaiAlgorithm
from environments.multi_fidelity_env import MultiFidelityEnvironment


def learn(algo: BaiAlgorithm, env: MultiFidelityEnvironment) -> (int, int):
    """
    :param algo: algorithm to be used
    :param env: env in which agent will run
    :return: (best arm, cost complexity)
    """
    while not algo.stopping_condition():
        curr_arms, fidelity = algo.pull_arm()
        rewards, cost = env.step(curr_arms, fidelity)
        algo.update(curr_arms, fidelity, rewards)

    return algo.recommendation()
