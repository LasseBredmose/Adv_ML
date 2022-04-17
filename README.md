# Adv_ML
Repository for the course 02460 Advanced Machine Learning at DTU

## Predict
```
python main.py predict models/STATEtrained_model_epocs100_16_04_22_trans_1_layers_5.pt
```

## Running tests on Laplace and their Hessians
### Diag Hessian
```
python main.py laplace models/STATEtrained_model_epocs100_16_04_22_trans_1_layers_5.pt diag
```

### Kron Hessian
```
python main.py laplace models/STATEtrained_model_epocs100_16_04_22_trans_1_layers_5.pt kron
```

### Full Hessian
```
python main.py laplace models/STATEtrained_model_epocs100_16_04_22_trans_1_layers_5.pt full
```
