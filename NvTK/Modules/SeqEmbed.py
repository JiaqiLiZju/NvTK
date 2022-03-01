import logging

import torch
import torch.nn as nn

class BasicConvEmbed(nn.Module):
    '''
    Embed Sequence using Convolution Layer.
    '''
    def __init__(self, out_planes, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, 
                        **conv_args)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.pool = pool(**pool_args) if pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        logging.debug(x.shape)
        return x


class RevComp(nn.Module):
    def forward(self, x):
        return x.flip([1,2]) # (batchsize, 4, seqlen)

class RevCompConvEmbed(nn.Module):
    '''
    Embed Sequence and Reverse Complement Sequence using Convolution Layer.
    '''
    def __init__(self, out_planes, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.RevCompConvEmbed = BasicConvEmbed(out_planes, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3})
        self.RevComp = RevComp()

    def forward(self, x):
        fmap1 = self.RevCompConvEmbed(x)
        fmap2 = self.RevCompConvEmbed(self.RevComp(x))
        return fmap1 + fmap2


class CharConvModule(nn.Module):
    def __init__(self, numFiltersConv1=40, filterLenConv1=5,
                        numFiltersConv2=44, filterLenConv2=15,
                        numFiltersConv3=44, filterLenConv3=25):
        
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv1, 
                                kernel_size=filterLenConv1, padding=(filterLenConv1 - 1) // 2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv2, 
                                kernel_size=filterLenConv2, padding=(filterLenConv2 - 1) // 2)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv3, 
                                kernel_size=filterLenConv3, padding=(filterLenConv3 - 1) // 2)

    def forward(self, x):
        logging.debug(x.shape)

        out1 = self.conv1(x)
        logging.debug(out1.shape)
        
        out2 = self.conv2(x)
        logging.debug(out2.shape)

        out3 = self.conv3(x)
        logging.debug(out3.shape)
        
        out = torch.cat([out1, out2, out3], dim=1)
        logging.debug(out.shape)

        return out

