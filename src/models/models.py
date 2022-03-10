import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d

# Defining hyperparameters
# First convolutional layer
num_filters_conv1 = 16
kernel_size_conv1 = 3
stride_size_conv1 = 2  # half the size
padding_size_conv1 = 1

# Second convolutional layer
num_filters_conv2 = 32
kernel_size_conv2 = 5
stride_size_conv2 = 2
padding_size_conv2 = 2

# Third convolutional layer
num_filters_conv3 = 64
kernel_size_conv3 = 5
stride_size_conv3 = 2
padding_size_conv3 = 2

num_l1 = 100


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

        # First convulutional layer
        self.conv1 = Conv2d(
            in_channels=input_channels,
            out_channels=num_filters_conv1,
            kernel_size=kernel_size_conv1,
            stride=stride_size_conv1,
            padding=padding_size_conv1,
        )

        self.conv_out_height = compute_conv_dim(
            input_height, kernel_size_conv1, padding=padding_size_conv1
        )
        self.conv_out_width = compute_conv_dim(
            input_width, kernel_size_conv1, padding=padding_size_conv1
        )

        # Second convulutional layer
        self.conv2 = Conv2d(
            in_channels=num_filters_conv1,
            out_channels=num_filters_conv2,
            kernel_size=kernel_size_conv2,
            stride=stride_size_conv2,
            padding=padding_size_conv2,
        )

        self.conv_out_height2 = compute_conv_dim(
            self.conv_out_height, kernel_size_conv2, padding=padding_size_conv2
        )
        self.conv_out_width2 = compute_conv_dim(
            self.conv_out_width, kernel_size_conv2, padding=padding_size_conv2
        )

        # Third convulutional layer
        self.conv3 = Conv2d(
            in_channels=num_filters_conv2,
            out_channels=num_filters_conv3,
            kernel_size=kernel_size_conv3,
            stride=stride_size_conv3,
            padding=padding_size_conv3,
        )

        self.conv_out_height3 = compute_conv_dim(
            self.conv_out_height2, kernel_size_conv3, padding=padding_size_conv3
        )
        self.conv_out_width3 = compute_conv_dim(
            self.conv_out_width2, kernel_size_conv3, padding=padding_size_conv3
        )

        # add dropout to network
        self.dropout = Dropout2d(p=0.2)
        self.l1_in_features = (
            num_filters_conv3 * self.conv_out_height3 * self.conv_out_width3
        )

        self.l_1 = Linear(
            in_features=self.l1_in_features // 256, out_features=num_l1, bias=True
        )
        self.l_out = Linear(in_features=num_l1, out_features=num_classes, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(num_filters_conv1)

        self.bn2 = nn.BatchNorm2d(num_filters_conv2)

        self.bn3 = nn.BatchNorm2d(num_filters_conv3)

    def forward(self, x):
        x = self.dropout(self.bn1(relu(self.conv1(x))))
        x = self.bn2(relu(self.conv2(x)))
        x = self.bn3(relu(self.conv3(x)))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(relu(self.l_1(x)))
        return softmax(self.l_out(x), dim=1)
