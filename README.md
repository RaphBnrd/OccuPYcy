# OccuPYcy

OccuPYcy is a package for <u>**occu**</u>pan<u>**cy**</u> modeling in <u>**PY**</u>thon. It provides tools for fitting occupancy models using maximum likelihood estimation through (stochastic) gradient descent optimization. 


## Local developpement commands

The following commnads will set up the development environment, install the package in editable mode, and run the tests (in quiet mode):

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
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
### Models
- [x] Linear model
- [x] MLP model
- [x] Model with MLP + Linear
- [ ] Models with qualitative variables
### Automation of training
- [ ] Validation for sgd like fit
- [ ] Early stoppintg for sgd like fit
- [ ] LR scheduler for sgd like fit
- [ ] Cross-validation for hyper parameter tuning
### Penalisation
- [ ] L2 penalisation
- [ ] L2 with naive estimator
- [ ] L1 penalisation
### Remaining tests
- [ ] Tests SGD-like fit
- [ ] Tests models
### Further features
- [ ] Allow user-defined split
### Cleaning
- [ ] Delete matplotlib dependency
### Documentation
- [ ] Docstrings for all functions and classes
- [ ] Full documentation
- [ ] Detailed examples for some use cases


## License

OccuPYcy is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
