# OccuPYcy

OccuPYcy is a package for <u>**occu**</u>pan<u>**cy**</u> modeling in <u>**PY**</u>thon. It provides tools for fitting occupancy models using maximum likelihood estimation through (stochastic) gradient descent optimization. 


## Local developpement commands

The following commands will set up the development environment, install the package in editable mode, and run the tests (in quiet mode):

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
```

If you don't want to use uv, you can use the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

## TODO

### First minimal version
- [x] Loss function
- [x] Data class
- [x] Constant model
- [x] Fit function for lbfgs

### SGD-like optimizers
- [x] Dataloaders
- [x] Fit function for sgd like optimizers

### Data
- [x] Filtering OccuData object on a subset of sites

### Models
- [x] Linear model
- [x] MLP model
- [x] Model with MLP + Linear

### Automation of training
- [x] Validation for sgd like fit
- [x] Early stopping for sgd like fit
- [x] LR scheduler for sgd like fit
- [ ] Go back to best model checkpoint for sgd like fit (when early stopping for instance)
- [ ] Checkpointing for sgd like fit
- [ ] Cross-validation for hyper parameter tuning

### Penalisation
- [x] L2/L1 penalisation on parameters
- [x] L2/L1 penalisation on logits
- [x] L2/L1 penalisation on logits with naive estimator

### Remaining tests
#### Data
- [ ] Dataloaders
- [ ] Splitting
- [ ] Filtering
#### Tests SGD-like fit
- [ ] Validation
- [ ] Early stopping
- [ ] LR scheduler
- [ ] Penalizations
- [ ] Checkpoints
#### Models
- [ ] Constant
- [ ] Linear
- [ ] MLP
- [ ] Linear + MLP

### Further features
- [x] Allow user-defined split
- [ ] Filtering OccuData object on a subset of species
- [ ] Models with qualitative variables

### Cleaning
- [ ] Delete dependencies: matplotlib, ipykernel

### Documentation
- [ ] Docstrings for all functions and classes
- [ ] Full documentation
- [ ] Detailed examples for some use cases


## License

OccuPYcy is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
