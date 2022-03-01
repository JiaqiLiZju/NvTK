''' 
    Basic Model in NvTK
    Code: jiaqili@zju.edu
'''

import logging

import torch
from torch import nn

from ..Modules import BasicModule

class BasicModel(BasicModule):
    '''
    Basis Model in NvTK
    contain Embedding, Encoder, Decoder, Predictor
    '''
    def __init__(self):
        super().__init__()
        self.Embedding = nn.Sequential()
        self.Encoder = nn.Sequential()
        self.Decoder = nn.Sequential()
        self.Predictor = nn.Sequential()

    def forward(self, x):
        embed = self.Embedding(x)
        logging.debug(embed.shape)

        fmap = self.Encoder(embed)
        logging.debug(fmap.shape)

        if len(fmap.shape) > 2:
            fmap = fmap.reshape((fmap.size(0), -1))
            logging.warning("fmap after Encoder reshaped as (batchsize, -1), \n \
                            Add Flatten module in Encoder to deprecate this warning")

        fmap = self.Decoder(fmap)
        logging.debug(fmap.shape)

        out = self.Predictor(fmap)
        logging.debug(out.shape)

        return out

