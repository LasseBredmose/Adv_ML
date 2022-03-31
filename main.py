from src.models.train_model import train
from src.models.predict_model import predict
from src.models.laplace_model import laplace, laplace_eval, laplace_sample
import sys

if __name__ == "__main__":
    """
    Main script for running what we have done
    Use: python main.py train
    """
    args = sys.argv
    # args are ['main.py', 'train', ...], so 1-indexed
    if args[1] == "train":
        train(small=1)

    elif args[1] == "predict":
        model_path = args[2]
        # Good model path: ./models/STATEtrained_model_epocs70_24-03-2022_22.pt
        predict(model_path)

    elif args[1] == "laplace":
        model_path = args[2]
        hessian = args[3]
        laplace(model_path, hessian)

    elif args[1] == "eval_la":
        la_path = args[2]
        laplace_eval(la_path)

    elif args[1] == "sample_la":
        la_path = args[2]
        laplace_sample(la_path, 10)
