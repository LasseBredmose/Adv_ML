import sys

sys.path.append("./src/data/")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import CNN
from dataloader import MURADataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    num_epochs = 10
    learning_rate = 0.00001
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

    train_set, validation_set = torch.utils.data.random_split(
        dataset, [20000, len(dataset) - 20000]
    )

    train_loader = DataLoader(
        dataset=dataset,
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

    model = CNN(input_channels=3,input_height=256,input_width=256,num_classes=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=w_decay)
    scheduler = ReduceLROnPlateau(optimizer,'min')
    criterion = nn.CrossEntropyLoss()

    num_epoch = 10  # Your code here!

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')



