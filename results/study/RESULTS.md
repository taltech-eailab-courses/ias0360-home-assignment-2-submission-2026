# Home Assignment 2 experiment results

Optimization objective: validation accuracy. Each selected checkpoint maximizes validation accuracy within its run.

| Experiment | Architecture | Epochs | Own-digit avg ms | Parameters / weight KiB / model H5 KiB | Validation accuracy | Own-digit accuracy |
|---|---|---:|---:|---|---:|---:|
| baseline | custom_cnn | 5 | 0.169 | 14,410 / 56.3 / 77.3 | 98.10% | 80% |
| more_epochs | custom_cnn | 10 | 0.167 | 14,410 / 56.3 / 77.3 | 98.47% | 80% |
| lower_learning_rate | custom_cnn | 5 | 0.156 | 14,410 / 56.3 / 77.3 | 97.12% | 70% |
| dropout | custom_cnn | 5 | 0.167 | 14,410 / 56.3 / 78.4 | 98.10% | 80% |
| wider_cnn | custom_cnn | 5 | 0.225 | 56,714 / 221.5 / 243.3 | 98.38% | 80% |
| sgd_momentum | custom_cnn | 5 | 0.158 | 14,410 / 56.3 / 77.3 | 98.13% | 70% |
| lenet | lenet | 5 | 0.230 | 61,706 / 241.0 / 269.3 | 98.05% | 90% |
| mlp | mlp | 5 | 0.156 | 109,386 / 427.3 / 444.9 | 97.13% | 80% |

Own-data status: available. Missing values are not zero accuracies.
Weight bytes and H5 bytes are separate measures: the serialized model already contains weights, so do not add them.
Model summaries are stored in each experiment directory. This does not measure peak runtime RAM.
The separate MNIST-reference timing column in results.csv uses ten validation examples, not personal data.
Own-data timing uses total warmed prediction time for ten digits divided by ten, excluding preprocessing.
With only one example per digit, own-data accuracy changes in 10 percentage-point steps.

Best validation result: more_epochs (98.47%).
Observed improvement over baseline: 0.367 percentage points.
The final summary identifies the best candidate within each architecture before testing the reserved MNIST test set.

## Architecture comparison on the reserved test set

| Architecture | Validation-selected experiment | Test accuracy |
|---|---|---:|
| custom_cnn | more_epochs | 98.56% |
| lenet | lenet | 98.21% |
| mlp | mlp | 97.44% |

These are results for one fixed split and seed. The custom CNN had more hyperparameter trials than the alternatives; this is not a matched-search-budget proof of architectural superiority.
The real touchscreen/SD-card evaluation remains required when own-data status is pending.
