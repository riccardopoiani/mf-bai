import yaml
import numpy as np
import argparse


def get_setting_1():
    curr_n_arms = 2000
    curr_m_fidelity = 4
    curr_distribution = "Gaussian"
    curr_costs = [1, 10, 100, 1000]
    curr_bias_term = [1.0, 0.2, 0.01, 0]
    curr_order = [0.3, 0.05, 0.001, 0]
    return curr_n_arms, curr_m_fidelity, curr_distribution, curr_costs, curr_bias_term, curr_order


def get_setting_2():
    curr_n_arms = 1000
    curr_m_fidelity = 5
    curr_distribution = "Gaussian"
    curr_costs = [16, 64, 256, 1024, 4096]
    curr_bias_term = [1.0, 0.4, 0.1, 0.01, 0]
    curr_order = [0.3, 0.1, 0.01, 0.001, 0]
    return curr_n_arms, curr_m_fidelity, curr_distribution, curr_costs, curr_bias_term, curr_order


def sample_max_fidelity_mean(dist_type="Bernoulli"):
    if dist_type == "Bernoulli":
        return np.random.uniform(low=0.5, high=0.9)
    if dist_type == "Gaussian":
        return np.random.normal(loc=0.0, scale=1.0)
    else:
        raise NotImplementedError


def sample_lower_fidelity_mean(gamma):
    return np.random.uniform(low=-gamma / 2, high=gamma / 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, default=1)
    args = parser.parse_args()
    write_file = True

    if args.setting == 1:
        n_arms, m_fidelity, distribution, costs, bias_term, order = get_setting_1()
        output_file = f"configs/simulation/ablation/setting_1_{distribution}"
    elif args.setting == 2:
        n_arms, m_fidelity, distribution, costs, bias_term, order = get_setting_2()
        output_file = f"configs/simulation/ablation/setting_2_{distribution}"
    else:
        raise RuntimeError

    max_fidelity = []
    for b, o in zip(bias_term, order):
        max_fidelity.append(b + (o / 2))

    variance_proxy = 0.25 if distribution == "Bernoulli" else 1

    theta = [[] for _ in range(m_fidelity)]
    theta[-1] = [sample_max_fidelity_mean(distribution) for _ in range(n_arms)]
    for i, g in enumerate(order):
        if g != 0:
            init_arms = [theta[-1][0] - g / 2 - bias_term[i],
                         theta[-1][0] + g / 2 - bias_term[i]]
            temp = [true_arm - b + sample_lower_fidelity_mean(g) for true_arm in theta[-1][2:]]
            theta[i] = init_arms + temp

    cfg_max_fidelity = {"n_arms": n_arms,
                        "m_fidelity": m_fidelity,
                        "distribution_type": distribution,
                        "variance_proxy": variance_proxy,
                        "costs": costs,
                        "max_fidelity": max_fidelity,
                        "theta": theta}

    cfg_order = {"n_arms": n_arms,
                 "m_fidelity": m_fidelity,
                 "distribution_type": distribution,
                 "variance_proxy": variance_proxy,
                 "costs": costs,
                 "order_fidelity": order,
                 "theta": theta}

    if write_file:
        with open(f"{output_file}_maxbound.yml", 'w') as outfile:
            yaml.dump(cfg_max_fidelity, outfile, default_flow_style=False)
        with open(f"{output_file}_order.yml", 'w') as outfile:
            yaml.dump(cfg_order, outfile, default_flow_style=False)
