import copy
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN_nomax, CNN
from src.models.Transformation import ChooseTrans
from src.models.utils import get_numpy, get_variable

warnings.filterwarnings("ignore")


def train(small, transf, mp):
    print(f'Small: {small}')
    print(f'Transform: {transf}')
    print(f'MaxPool: {mp}')
    # Cuda Stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    print("Running GPU.") if use_cuda else print("No GPU available.")

    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ChosenTrans = ["Blank", "FlipV", "FlipH", "Rotate90", "Rotate180", "Rotate270"]
    num_epochs = 10
    learning_rate = 0.001
    w_decay = 0.001
    # train_CNN = False
    batch_size = 32
    shuffle = True
    # pin_memory = True
    num_workers = 1

    dataset = MURADataset(
        "data",
        "MURA-v1.1/train_labeled_studies.csv",
        "MURA-v1.1/train_image_paths.csv",
        transform=transform,
    )

    test_set = MURADataset(
        "data",
        "MURA-v1.1/valid_labeled_studies.csv",
        "MURA-v1.1/valid_image_paths.csv",
        transform=transform,
    )

    if small == 1:
        train_set, validation_set = torch.utils.data.random_split(
            dataset, [len(dataset) - 250, 250]
        )
        train_set, dummy = torch.utils.data.random_split(
            train_set, [750, len(train_set) - 750]
        )
        num_epochs = 2
    else:
        train_set, validation_set = torch.utils.data.random_split(
            dataset, [25765, len(dataset) - 25765]
        )

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        dataset=validation_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if mp == 0:
        model = CNN_nomax(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)
    else:
        model = CNN(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=w_decay
    )
    # scheduler = ReduceLROnPlateau(optimizer, "min")
    criterion = nn.CrossEntropyLoss()

    train_loss_epoch, validation_loss_epoch = [], []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        training_loss = []
        running_loss = 0.0

        # Training the model
        model.train()
        for i, data in enumerate(train_loader):

            # get the inputs
            inputs, labels = data

            print(labels)

            # wrap them in Variable
            inputs, labels = Variable(get_variable(inputs)), Variable(
                get_variable(labels)
            )
            if transf == 1:
                trans = ChooseTrans(ChosenTrans)
                inputs = trans(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # output = softmax(model(inputs), dim=1)
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss.append(get_numpy(loss))
            # print statistics
            running_loss += get_numpy(loss)

            if i % 10 == 9:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        # Evaluating the model
        model.eval()

        validation_loss = []

        lowest_loss = 100000
        correct = 0
        total = 0
        for i, data in enumerate(validation_loader):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(get_variable(inputs)), Variable(
                get_variable(labels)
            )

            """if transf == 1:
                trans = ChooseTrans(ChosenTrans)
                inputs = trans(inputs)"""

            # forward + backward + optimize
            # output = softmax(model(inputs), dim=1)
            output = model(inputs)
            loss = criterion(output, labels)

            validation_loss.append(get_numpy(loss))

        print(f"VALIDATION loss: {np.mean(validation_loss)}")

        train_loss_epoch.append(np.mean(training_loss))
        validation_loss_epoch.append(np.mean(validation_loss))

        if np.mean(validation_loss) < lowest_loss:
            lowest_loss = np.mean(validation_loss)
            best_model = copy.deepcopy(model.state_dict())
    print("Finished !!")

    date_time = datetime.now().strftime("%d_%m_%H_%M_%S")

    # Storing the LAST model
    torch.save(
        model.state_dict(),
        f"./models/STATEtrained_model_LAST_epochs{num_epochs}_{date_time}_trans_{transf}_mp_{maxpool}.pt",
    )

    # Storing the BEST model
    model.load_state_dict(best_model)
    # torch.save(model, f'./models/trained_model_epocs{num_epochs}_{date_time}.pt')
    torch.save(
        model.state_dict(),
        f"./models/STATEtrained_model_BEST_epochs{num_epochs}_{date_time}_trans_{transf}_mp_{layers}.pt",
    )

    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(get_variable(inputs)), Variable(labels)
        output = softmax(model(inputs), dim=1)

        predicted = torch.max(output, 1)[1]
        total += labels.size(0)
        correct += (get_numpy(predicted) == labels.numpy()).sum()

    # print("Accuracy of the network: {:4.2f} %".format(100 * correct.true_divide(total)))
    print(
        "Accuracy of the network: {:4.2f} %".format(
            100 * np.true_divide(correct, total)
        )
    )
    plt.plot(
        range(num_epochs), train_loss_epoch, range(num_epochs), validation_loss_epoch
    )
    plt.legend(["Training data", "Validation data"])
    plt.savefig(
        f"./reports/learning_curve/epochs{num_epochs}_{date_time}_trans_{transf}_layers_{mp}"
    )
