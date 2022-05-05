# Adv_ML
Repository for the course 02460 Advanced Machine Learning at DTU.

To run our code, you can either train a model from scratch, but also use some of our already pretrained models! 😄 

The pretrained model that we recommend to use is stored in `models/deep_ensemble/moment/STATEtrained_model_LAST_epochs100_29_04_22_14_04_trans_1_mp_1_arr_11.pt`

## Train
Train on the full training set:
```
python3 main.py train 0 1 5 0
```

Train on a small (sub)set
```
python3 main.py train 1 1 5 0
```

## Predict
Make predictions using our model and store probabilities (will be used for Deep Ensemble)
```
python main.py predict models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt
```

## Running tests on Laplace and their Hessians
We use Laplace approximation on the last layer to create a BNN
### Diag Hessian
```
python3 main.py laplace models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt diag
```

### Kron Hessian
```
python3 main.py laplace models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt kron
```

### Full Hessian
```
python3 main.py laplace models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt full
```

## Sample from posterior
Make samples from the posterior distribution. The third argument is the method (`average`, `intersect` or `union`).
```
python3 main.py sample_la models/laplace_diag_22-04-2022_00.pkl <METHOD>
```

## Class Activation Maps
We can make CAMs on CNNs and BNNs. Follow the overall structure:
```
python3 main.py cam <IMAGE PATH> <MODEL_PATH>
```

Make CAMs from CNN
```
python3 main.py cam data/MURA-v1.1/valid/XR_SHOULDER/patient11723/study1_positive/image3.png models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt
```

Make CAMs from BNN
```
python3 main.py cam data/MURA-v1.1/valid/XR_SHOULDER/patient11723/study1_positive/image3.png models/BNN_diag_average_23-04-2022_10.pt
```