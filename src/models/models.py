import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d

# Defining hyperparameters
# First convolutional network
num_filters_conv1 = 16
kernel_size_conv1 = 3
stride_size_conv1 = 2 # half the size
padding_size_conv1 = 1

# First convolutional network
num_filters_conv2 = 32
kernel_size_conv2 = 5
stride_size_conv2 = 2
padding_size_conv2 = 2

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
            in_channels = input_channels, 
            out_channels = num_filters_conv1,
            kernel_size = kernel_size_conv1,
            stride = stride_size_conv1, 
            padding = padding_size_conv1
            )

        self.conv_out_height = compute_conv_dim(input_height, kernel_size_conv1, padding=padding_size_conv1)
        self.conv_out_width = compute_conv_dim(input_width, kernel_size_conv1, padding=padding_size_conv1)
                                                                                                                                                                 
        # Second convulutional layer
        self.conv2 = Conv2d(
            in_channels = num_filters_conv1,
            out_channels = num_filters_conv2,
            kernel_size = kernel_size_conv2,
            stride = stride_size_conv2,
            padding = padding_size_conv2
        )

        self.conv_out_height2 = compute_conv_dim(self.conv_out_height, kernel_size_conv2, padding=padding_size_conv2)
        self.conv_out_width2 = compute_conv_dim(self.conv_out_width, kernel_size_conv2, padding=padding_size_conv2)

        # add dropout to network
        self.dropout = Dropout2d(p=0.2)
        self.l1_in_features = num_filters_conv2 * self.conv_out_height2 * self.conv_out_width2

        self.l_1 = Linear(in_features=self.l1_in_features//64, 
                          out_features=num_l1,
                          bias=True)
        self.l_out = Linear(in_features=num_l1, 
                            out_features=num_classes,
                            bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)
        
        self.bn1 = nn.BatchNorm2d(num_filters_conv1)
        
        self.bn2 = nn.BatchNorm2d(num_filters_conv2)
        
    def forward(self, x):
        x = self.dropout(self.bn1(relu(self.conv1(x))))
        x = self.bn2(relu(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(relu(self.l_1(x)))
        return softmax(self.l_out(x), dim=1)

"""
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True) 
    ) 

self.conv_last = last_conv(2*self.num_chan,self.num_chan,1)

def __init__(self, channels):
        super().__init__()
           
        self.num_chan = channels
        
        self.dconv_down1 = double_conv(self.num_chan, 2*self.num_chan)
        self.dconv_down2 = double_conv(2*self.num_chan, 4*self.num_chan)
        self.dconv_down3 = double_conv(4*self.num_chan, 8*self.num_chan)
        self.dconv_down4 = double_conv(8*self.num_chan, 16*self.num_chan)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        
        # Layer 1 down
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        # Layer 2 down
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        # Layer 3 down
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        # Layer 4 (bottom layer of the U)
        x = self.dconv_down4(x)
"""