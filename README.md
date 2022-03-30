# Code for the paper "Multi-Fidelity Best-Arm Identification" 
This repository contains the code to reproduce the experiments for the "Multi-Fidelity Best-Arm Identification" paper.
In particular, we provide 3 main python scripts:
- `run.py`
- `run_yahtzee.py`
- `generate_config.py`

The script `run.py` is used to run a given experiments. More specifically the typical command is given by 
`run.py --algo AlgoName --env-cfg CfgName --dump-dir results/exp_name --delta 0.001 --n-jobs 10 --n-runs 100`,
where `algo` is the name of the algorithm that will be used (see full list within the script), `env-cfg` is 
the configuration of the environment (e.g., the synthetic domains or the yahtzee one), `delta` is confidence, 
`n-jobs` is the number of parallel runs and `n-runs` is the number of total runs. Finally, the additional 
`--discard-all-th` can be used to specify thresholds of the MFE algorithm. 
The run script will dump on file the results of the run(s) in `dump-dir/result.pkl`. 
These results can be read with and visualized with any tool of choice.

For what concerns `generate_config.py` and `run_yahtzee.py`, we describe their behavior below. They are python scripts 
that helps in generating configurations that will be used as arguments in the `run.py` script.


### Synthetic domains
Configurations of Synthetic A and Synthetic B have been generated using `generate_config.py`.
In particular, `python generate_config.py --setting 1` generates Synthetic A, 
and `python generate_config.py --setting 2` generates Synthetic B. 

In `configs/simulation/ablation/`, you can find the configurations that we used in the paper.
Suffix `order` stands for the configuration for IISE-gamma, while suffix `maxbound` stands 
for the configuration of IISE and MFE. SE can be run with any configuration file.

These configurations can be used in `run.py` to run our algorithms on these domains.


### Yahtzee
First of all, we mention that our implementation relies on a python implementation of the Yahtzee game. 
More specifically, we use version `1.2.7` of `pyhtzee` (https://pypi.org/project/pyhtzee/).

Using `run_yathzee.py` we generate arm distributions that can be used within `run.py` with
any algorithm of our choice. In `configs/yahtzee` we processed the results in configuration files that can be
used directly within the `run.py` script. Suffix `order` stands for the configuration for IISE-gamma, while suffix `maxbound` stands 
for the configuration of IISE and MFE. SE can be run with any configuration file.