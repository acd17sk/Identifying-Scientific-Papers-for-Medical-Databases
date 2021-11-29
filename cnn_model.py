from torch import nn
import torch


class ConvNet(nn.Module):
    def __init__(self,
                 channels_in=1,
                 channels_out=50,
                 num_features=250,
                 kernel_size_conv=5,
                 dilation_conv=1,
                 padding_conv=0,
                 stride_conv=1,
                 kernel_size_mxp=5,
                 dilation_mxp=1,
                 padding_mxp=0,
                 stride_mxp=2):

        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1d(channels_in,
                               channels_out,
                               kernel_size_conv,
                               dilation=dilation_conv,
                               padding=padding_conv,
                               stride=stride_conv)

        out = int(((num_features + 2*padding_conv - dilation_conv * (kernel_size_conv - 1) - 1) / stride_conv) + 1)

        self.pool1 = nn.MaxPool1d(kernel_size_mxp,
                                  stride_mxp,
                                  dilation=dilation_mxp,
                                  padding=padding_mxp)

        out = int(((out + 2*padding_mxp - dilation_mxp * (kernel_size_mxp - 1) - 1) / stride_mxp) + 1)

        self.fc = nn.Linear(channels_out * out,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.pool1(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
