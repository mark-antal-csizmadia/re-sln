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