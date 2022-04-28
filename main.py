import sys

from src.models.cam import cam
from src.models.laplace_model import laplace, laplace_eval, laplace_sample
from src.models.predict_model import predict
from src.models.train_model import train
from src.models.deep_ensemble import deep_ensemble

if __name__ == "__main__":
    """
    Main script for running what we have done
    Use: python main.py train
    """
    args = sys.argv
    # args are ['main.py', 'train', ...], so 1-indexed
    if args[1] == "train":
        train(
            small=int(args[2]),
            transf=int(args[3]),
            mp=int(args[4]),
            arr=int(args[5]),
        )

    elif args[1] == "predict":
        model_path = args[2]
        mp = int(args[3])
        # Good model path: ./models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt
        predict(model_path, mp)

    elif args[1] == "laplace":
        model_path = args[2]
        hessian = args[3]
        mp = int(args[4])
        laplace(model_path, hessian, mp)

    elif args[1] == "eval_la":
        la_path = args[2]
        laplace_eval(la_path)

    elif args[1] == "sample_la":
        # python3 main.py sample_la models/laplace.pkl average
        la_path = args[2]
        method = args[3]
        laplace_sample(la_path, 10, method)

    elif args[1] == "cam":
        # CNN:
        # python3 main.py cam data/MURA-v1.1/valid/XR_SHOULDER/patient11723/study1_positive/image3.png models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt

        # BNN:
        # python3 main.py cam data/MURA-v1.1/valid/XR_SHOULDER/patient11723/study1_positive/image3.png models/BNN_diag_average_23-04-2022_10.pt
        # python3 main.py cam data/MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png models/BNN_diag_average_23-04-2022_10.pt

        image_path = args[2]
        model_path = args[3]
        mp = int(args[4])  # whether there is max pooling
        cam(image_path, model_path, mp)

    elif args[1] == "deep":
        # python3 main.py deep models/deep_ensemble 1
        folder_path = args[2]
        num_models = int(args[3])
        deep_ensemble(folder_path, num_models)
