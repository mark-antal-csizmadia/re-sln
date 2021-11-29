#  re-sln

Re-implementation of the paper titled "Noise against noise: stochastic label noise helps combat inherent label noise" from ICLR 2021.

## Setup

Make a virtual env and isntall dependencies from the ```environment.yml``` file.

## Run

Run the ```main.ipynb``` notebook.

## Logs

Tensorboard is used for logging. Share your logs as shown below (from [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#share-tensorboard-dashboards)):
```
tensorboard dev upload --logdir runs --name "re-sln results" --description "By Mark"
```

## Experiments

Available models: CE, SLN, SLN+MO, SLN+MO+LC
Available noisy data sets for CIFAR-10 (p=0.4): sym (paper, mine), asym (paper, mine), dependent (paper), openset (paper, mine)

| `model` / `noise` | sym |  | asym |  | dependent |  | openset |  |
| - | - | - | - | - | - | - | - | - |
|  | paper | custom | paper | custom | paper | custom | paper | custom |
| CE | exp_2021-11-25 13:17:26.851200 | exp_2021-11-25 20:30:28.794160 | exp_2021-11-26 09:21:19.524188 | exp_2021-11-26 14:03:18.975684 | exp_2021-11-26 20:14:40.983299 | x | x | exp_2021-11-27 13:35:23.026659 |
| SLN | exp_2021-11-25 15:38:09.361059 | exp_2021-11-25 20:31:37.546765 | exp_2021-11-26 11:41:56.758060 | exp_2021-11-26 16:11:00.844488 | exp_2021-11-27 11:07:55.847340 | x | x | exp_2021-11-27 13:44:37.885816 |
|  SLN+MO | exp_2021-11-25 16:46:29.066838 | exp_2021-11-26 09:18:14.291265 | exp_2021-11-26 11:44:27.727904 | exp_2021-11-26 16:14:06.628600 | exp_2021-11-27 11:11:07.020347 | x | x | exp_2021-11-27 13:46:43.777573 |
| SLN+MO+LC | exp_2021-11-26 11:18:36.051172 | exp_2021-11-26 13:51:03.590616 | exp_2021-11-26 13:57:45.567433 | exp_2021-11-26 16:16:06.031597 | exp_2021-11-27 11:14:24.120092 | x | x | exp_2021-11-28 16:34:38.935269 |

Training Times per Computational Resources
- exp_2021-11-25 13:17:26.851200: 1 h 31 m with 1 x V100
get times and final test accs from runs/
