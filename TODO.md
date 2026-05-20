If you identify any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on the GitHub repository. 

Contributions are always welcome! 😉

# Most pressing tasks

## Remaining tests
### Data
- [ ] Dataloaders
- [ ] Splitting
- [ ] Filtering
### Tests SGD-like fit
- [ ] Validation
- [ ] Early stopping
- [ ] LR scheduler
- [ ] Penalizations
- [ ] Checkpoints
### Models
- [ ] Constant
- [ ] Linear
- [ ] MLP
- [ ] Linear + MLP

## Further features
- [ ] Resume training from a checkpoint

## Cleaning
- [ ] Delete dependencies: matplotlib, ipykernel

## Documentation
- [ ] Full documentation
- [ ] Detailed examples for some use cases (custom model, different models pre-defined, cross-validation, penalization, etc.)

# Less pressing tasks

## Further features
- [ ] WandB integration for training monitoring and hyperparameter tuning
- [ ] Filtering OccuData object on a subset of species
- [ ] Models with qualitative variables

## Cleaning
- [ ] Choose between pandas and polars


# DONE

## First minimal version
- [x] Loss function
- [x] Data class
- [x] Constant model
- [x] Fit function for lbfgs

## SGD-like optimizers
- [x] Dataloaders
- [x] Fit function for sgd like optimizers

## Data
- [x] Filtering OccuData object on a subset of sites

## Models
- [x] Linear model
- [x] MLP model
- [x] Model with MLP + Linear

## Automation of training
- [x] Validation for sgd like fit
- [x] Early stopping for sgd like fit
- [x] LR scheduler for sgd like fit
- [x] Go back to best model checkpoint for sgd like fit (when early stopping for instance)
- [x] Cross-validation for hyper parameter tuning

## Penalisation
- [x] L2/L1 penalisation on parameters
- [x] L2/L1 penalisation on logits
- [x] L2/L1 penalisation on logits with naive estimator

## Further features
- [x] Allow user-defined split
- [x] Formula similar to R's formula for model specification (almost linear models)
- [x] Utils to convert input data (e.g. csv of pandas DataFrame) to OccuData object
- [x] Save checkpoints along the training for sgd like fit

## Documentation
- [x] Docstrings for all functions and classes