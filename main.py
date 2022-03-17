import copy
import warnings

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # Cuda Stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    print("Running GPU.") if use_cuda else print("No GPU available.")

    def get_variable(x):
        # Converts tensors to cuda, if available.
        if use_cuda:
            return x.cuda()
        return x

    def get_numpy(x):
        # Get numpy array for both cuda and not. 
        if use_cuda:
            '''return x.cpu().item()
        return x.item()'''
            return x.cpu().data.numpy()
        return x.data.numpy()

    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    num_epochs = 70
    learning_rate = 0.001
    w_decay = 0.001
    train_CNN = False
    batch_size = 32
    shuffle = True
    pin_memory = True
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

    # train_set_dummy, validation_set = torch.utils.data.random_split(
    #     dataset, [len(dataset) - 250, 250]
    # )
    # train_set, dummy = torch.utils.data.random_split(
    #     train_set_dummy, [750, len(train_set_dummy) - 750]
    # )

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

    model = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=w_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, "min")
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

            # wrap them in Variable
            inputs, labels = Variable(get_variable(inputs)), Variable(get_variable(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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
            inputs, labels = Variable(get_variable(inputs)), Variable(get_variable(labels))

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)

            validation_loss.append(get_numpy(loss))

        train_loss_epoch.append(np.mean(training_loss))
        validation_loss_epoch.append(np.mean(validation_loss))

        if np.mean(validation_loss) < lowest_loss:
            lowest_loss = np.mean(validation_loss)
            best_model = copy.deepcopy(model.state_dict())
    print("Finished !!")

    # Storing the model
    date_time = datetime.now().strftime("%d-%m-%Y_%H")
    model.load_state_dict(best_model)
    # torch.save(model, f'./models/trained_model_epocs{num_epochs}_{date_time}.pt')
    torch.save(model.state_dict(), f'./models/STATEtrained_model_epocs{num_epochs}_{date_time}.pt')

    model.eval()
    correct = 0 
    total = 0  
    for data in test_loader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(get_variable(inputs)), Variable(labels)
        output = model(inputs)

        predicted = torch.max(output, 1)[1]
        total += labels.size(0)
        correct += (get_numpy(predicted) == labels.numpy()).sum()

    #print("Accuracy of the network: {:4.2f} %".format(100 * correct.true_divide(total)))
    print("Accuracy of the network: {:4.2f} %".format(100 * np.true_divide(correct,total)))
    plt.plot(
        range(num_epochs), train_loss_epoch, range(num_epochs), validation_loss_epoch
    )
    plt.legend(["Training data", "Validation data"])
    plt.savefig(f'./reports/epocs{num_epochs}_{date_time}')
