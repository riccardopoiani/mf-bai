import pickle
import numpy as np

from utils.math import conf_interval
from visualization.results import ResultItem


def get_complexity(sample: ResultItem):
    return sum([arr.sum() for arr in sample.cost_per_arm_and_fidelity])


def get_complexity_by_phase(sample: ResultItem, phase: int):
    return sample.cost_per_arm_and_fidelity[phase].sum()


if __name__ == "__main__":
    exp_name = "ya"

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
    iise_gain = []
    iise_order_gain = []
    mfe_gain = []
    for iise_sample, iise_order_sample, se_sample, mfe_sample in zip(iise_x, iise_order_x, se_x, mfe_x):
        iise_gain.append(get_complexity(iise_sample) / get_complexity(se_sample) * 100)
        iise_order_gain.append(get_complexity(iise_order_sample) / get_complexity(se_sample) * 100)
        mfe_gain.append((get_complexity(mfe_sample)) / get_complexity(se_sample) * 100)

    iise_gain = np.array(iise_gain)
    iise_order_gain = np.array(iise_order_gain)
    mfe_gain = np.array(mfe_gain)

    print(f"IISE GAIN: {iise_gain.mean()} +- {conf_interval(iise_gain.std(), 100)}")
    print(f"IISE ORDER GAIN: {iise_order_gain.mean()} +- {conf_interval(iise_order_gain.std(), 100)}")
    print(f"MFE GAIN: {mfe_gain.mean()} +- {conf_interval(mfe_gain.std(), 100)}")
    print()

    # Total cost invested by each algorithm in different fidelity
    M = len(iise_x[0].cost_per_arm_and_fidelity)

    iise_by_fidelity = [[] for _ in range(M)]
    iise_order_by_fidelity = [[] for _ in range(M)]
    se_by_fidelity = [[] for _ in range(M)]
    mfe_by_fidelity = [[] for _ in range(M)]

    for iise_sample, iise_order_sample, se_sample, mfe_sample in zip(iise_x, iise_order_x, se_x, mfe_x):
        for m in range(M):
            iise_by_fidelity[m].append(get_complexity_by_phase(iise_sample, m))
            iise_order_by_fidelity[m].append(get_complexity_by_phase(iise_order_sample, m))
            se_by_fidelity[m].append(get_complexity_by_phase(se_sample, m))
            mfe_by_fidelity[m].append(get_complexity_by_phase(mfe_sample, m))

    iise_by_fidelity = np.array(iise_by_fidelity)
    iise_order_by_fidelity = np.array(iise_order_by_fidelity)
    mfe_by_fidelity = np.array(mfe_by_fidelity)
    se_by_fidelity = np.array(se_by_fidelity)

    print(se_by_fidelity.mean(1))