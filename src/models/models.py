import torch.nn as nn
from torch.nn import Conv2d, Dropout2d, Linear
from torch.nn.functional import relu

# Defining hyperparameters
# First convolutional layer
num_filters_conv1 = 16
kernel_size_conv1 = 3
stride_size_conv1 = 1
padding_size_conv1 = 1

# Second convolutional layer
num_filters_conv2 = 32
kernel_size_conv2 = 3
stride_size_conv2 = 1
padding_size_conv2 = 1

# Third convolutional layer
num_filters_conv3 = 64
kernel_size_conv3 = 3
stride_size_conv3 = 1
padding_size_conv3 = 1

# Fourth convolutional layer
num_filters_conv4 = 128
kernel_size_conv4 = 3
stride_size_conv4 = 1
padding_size_conv4 = 1

# Fifth convolutional layer
num_filters_conv5 = 256
kernel_size_conv5 = 3
stride_size_conv5 = 1
padding_size_conv5 = 1


def compute_conv_dim(dim_size, kernel_size, padding=0, stride=1):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


# Creating the model class
class CNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_classes):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = Conv2d(
            in_channels=input_channels,  # 3
            out_channels=num_filters_conv1,  # 16
            kernel_size=kernel_size_conv1,  # 3
            stride=stride_size_conv1,  # 2
            padding=padding_size_conv1,  # 1
        )

        self.bn1 = nn.BatchNorm2d(num_filters_conv1)

        # Second convolutional layer
        self.conv2 = Conv2d(
            in_channels=num_filters_conv1,
            out_channels=num_filters_conv2,
            kernel_size=kernel_size_conv2,
            stride=stride_size_conv2,
            padding=padding_size_conv2,
        )

        self.bn2 = nn.BatchNorm2d(num_filters_conv2)

        # Third convolutional layer
        self.conv3 = Conv2d(
            in_channels=num_filters_conv2,
            out_channels=num_filters_conv3,
            kernel_size=kernel_size_conv3,
            stride=stride_size_conv3,
            padding=padding_size_conv3,
        )

        self.bn3 = nn.BatchNorm2d(num_filters_conv3)

        # Fourth convolutional layer
        self.conv4 = Conv2d(
            in_channels=num_filters_conv3,
            out_channels=num_filters_conv4,
            kernel_size=kernel_size_conv4,
            stride=stride_size_conv4,
            padding=padding_size_conv4,
        )

        self.bn4 = nn.BatchNorm2d(num_filters_conv4)

        # Fourth convolutional layer
        self.conv5 = Conv2d(
            in_channels=num_filters_conv4,
            out_channels=num_filters_conv5,
            kernel_size=kernel_size_conv5,
            stride=stride_size_conv5,
            padding=padding_size_conv5,
        )

        self.bn5 = nn.BatchNorm2d(num_filters_conv5)

        self.mp = nn.MaxPool2d(2, 2)

        # add dropout to network
        self.dropout = Dropout2d(p=0.2)

        self.avgpool = nn.AvgPool2d(16)

        self.l_out = Linear(
            in_features=num_filters_conv5, out_features=num_classes, bias=True
        )

    def forward(self, x):
        x = self.bn1(self.mp(relu(self.conv1(x))))
        x = self.bn2(self.mp(relu(self.conv2(x))))
        x = self.bn3(self.mp(relu(self.conv3(x))))
        x = self.bn4(self.mp(relu(self.conv4(x))))
        x = self.bn5(relu(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(relu(x))
        x = self.dropout(x)
        # return softmax(self.l_out(x), dim=1)
        return self.l_out(x)


# Try a version without max pooling
class CNN_nomax(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_classes):
        super(CNN_nomax, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(16)

        # Second convolutional layer
        self.conv2 = Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.bn2 = nn.BatchNorm2d(32)

        # Third convolutional layer
        self.conv3 = Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.bn3 = nn.BatchNorm2d(64)

        # Fourth convolutional layer
        self.conv4 = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.bn4 = nn.BatchNorm2d(64)

        # Fourth convolutional layer
        self.conv5 = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.bn5 = nn.BatchNorm2d(64)

        # add dropout to network
        self.dropout = Dropout2d(p=0.2)

        self.avgpool = nn.AvgPool2d(256 // 8)

        self.l_out = Linear(in_features=64, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.dropout(self.bn1(relu(self.conv1(x))))
        x = self.bn2(relu(self.conv2(x)))
        x = self.bn3(relu(self.conv3(x)))
        x = self.bn4(relu(self.conv4(x)))
        x = self.bn5(relu(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        return self.l_out(x)


class CNN_3(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_classes):
        super(CNN_3, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = Conv2d(
            in_channels=input_channels,  # 3
            out_channels=num_filters_conv1,  # 16
            kernel_size=kernel_size_conv1,  # 3
            stride=stride_size_conv1,  # 2
            padding=padding_size_conv1,  # 1
        )

        self.bn1 = nn.BatchNorm2d(num_filters_conv1)

        # Second convolutional layer
        self.conv2 = Conv2d(
            in_channels=num_filters_conv1,
            out_channels=num_filters_conv2,
            kernel_size=kernel_size_conv2,
            stride=stride_size_conv2,
            padding=padding_size_conv2,
        )

        self.bn2 = nn.BatchNorm2d(num_filters_conv2)

        # Third convolutional layer
        self.conv3 = Conv2d(
            in_channels=num_filters_conv2,
            out_channels=num_filters_conv3,
            kernel_size=kernel_size_conv3,
            stride=stride_size_conv3,
            padding=padding_size_conv3,
        )

        self.bn3 = nn.BatchNorm2d(num_filters_conv3)

        self.mp = nn.MaxPool2d(2, 2)

        # add dropout to network
        self.dropout = Dropout2d(p=0.2)

        self.avgpool = nn.AvgPool2d(kernel_size=32)

        self.l_out = Linear(
            in_features=num_filters_conv3, out_features=num_classes, bias=True
        )

    def forward(self, x):
        x = self.bn1(self.mp(relu(self.conv1(x))))
        x = self.bn2(self.mp(relu(self.conv2(x))))
        x = self.bn3(self.mp(relu(self.conv3(x))))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(relu(x))
        x = self.dropout(x)
        # return softmax(self.l_out(x), dim=1)
        return self.l_out(x)
