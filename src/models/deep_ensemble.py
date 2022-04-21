from os import listdir
from src.models.models import CNN
import torch

# models/STATEtrained_model_epocs2_21_04_15_trans_1_layers_5_arr_0_bnfirst.pt
def deep_ensemble(folder_path, num_models):
    # Loading cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = listdir(folder_path)[:num_models]

    model = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    model.eval()

    def model_loader(path):
        model.load_state_dict(
            torch.load(
                f'models/deep_ensemble/{path}',
                map_location=torch.device(device),
            )
        )
        return model
    
    models = [model_loader(p) for p in model_paths]

    weights = [m.l_out.weight.data for m in models]
    biases = [m.l_out.bias.data for m in models]

    mean_weights = torch.stack(weights).mean(axis=0)
    mean_biases = torch.stack(biases).mean(axis=0)

    # Load weights and biases into the model

    # Save the model



