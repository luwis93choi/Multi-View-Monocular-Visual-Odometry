import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

from torch.nn.init import kaiming_normal_ # (Reference : https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py)

import numpy as np

class DeepVONet_without_Activation(nn.Module):

    # DeepVO NN Initialization
    # Overriding base class of neural network (nn.Module)
    def __init__(self, lstm_layer=2, lstm_hidden_size=1000):
        super(DeepVONet_without_Activation, self).__init__()

        self.use_cuda = False

        # CNN Layer 1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        # CNN Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        # CNN Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        # CNN Layer 3_1
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # CNN Layer 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # CNN Layer 4_1
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # CNN Layer 5
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # CNN Layer 5_1
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # CNN Layer 6
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # RNN Layer (Reference : https://github.com/thedavekwon/DeepVO)
        self.rnn = nn.LSTM(
            input_size = 6 * 20 * 1024,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_layer,
            batch_first = True
        )

        # RNN Dropout
        self.rnn_drop = nn.Dropout(0.5)

        # Linear Regression between RNN output features (1x500) and Absolute Scale between t-1 and t (1x1) (Absolute Scale)
        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=1)

        # RNN Learnable Variable Initilization (Reference : https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py)
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            
            elif isinstance(m, nn.LSTM):
                # RNN Layer 1 Init 
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # RNN Layer 2 Init
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # Foward pass of DeepVO NN
    def forward(self, x):
        
        # Forward pass through CNN Layer 1
        x = self.conv1(x)

        # Forward pass through CNN Layer 2
        x = self.conv2(x)

        # Forward pass through CNN Layer 3
        x = self.conv3(x)

        # Forward pass through CNN Layer 3_1
        x = self.conv3_1(x)

        # Forward pass through CNN Layer 4
        x = self.conv4(x)

        # Forward pass through CNN Layer 4_1
        x = self.conv4_1(x)

        # Forward pass through CNN Layer 5
        x = self.conv5(x)

        # Forward pass through CNN Layer 5_1
        x = self.conv5_1(x)

        # Foward pass through CNN Layer 6
        x = self.conv6(x)

        # Reshpae/Flatten the output of CNN in order to use it as the input of RNN
        x = x.view(x.size(0), x.size(0), -1)     # Input Shape : [Length of recurrent sequence, Size(0), Size(1)]

        # RNN Layer Forward Pass
        x, _ = self.rnn(x)

        # RNN Dropout
        x = self.rnn_drop(x)

        # Forward pass into Linear Regression in order to change output vectors into Pose vector
        x = self.fc(x)

        return x

    def get_pose_loss(self, _estimated_scale, _groundtruth_scale):

        # Custom Loss Function if necessary

        return ''