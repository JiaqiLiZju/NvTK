import torch
from torch import nn

from NvTK.Model import BasicModel
from NvTK.Modules import BasicConv1d, BasicConvEmbed, Flatten, BasicLinearModule, BasicPredictor
from NvTK.Modules import CBAM, CharConvModule

from collections import OrderedDict


class BaselineCNN(BasicModel):
    '''
    Baseline CNN (Default two-layer CNN; 
        Embedding: BasicConvEmbed(N128_L15_P15_Pa_ReLU_BN)
        Encoder: BasicConv1d(N256|CBAM) * n_deep_convlayers; GAP{64|Expr, 1|Epig}; Flatten
        Decoder: MTLnone
        Predictor: regression/binary_classification)
    BasicConvEmbedding could be tuned afterward.
    '''
    def __init__(self, output_size, 
                    emb_planes=128, kernel_size=15, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 15},
                    n_deep_convlayers=1, use_CBAM=False, GAP=64, tasktype='regression'):
        super().__init__()
        self.Embedding = BasicConvEmbed(out_planes=emb_planes, 
                    kernel_size=kernel_size, in_planes=in_planes, conv_args=conv_args, 
                    bn=bn, activation=activation, activation_args=activation_args, 
                    pool=pool, pool_args=pool_args)

        encoder_layers = OrderedDict([('Conv', BasicConv1d(in_planes=emb_planes, out_planes=256))])
        if use_CBAM:
            encoder_layers['CBAM'] = CBAM(256)
        if n_deep_convlayers > 1:
            for n in range(1, n_deep_convlayers):
                encoder_layers['Conv_'+n] = BasicConv1d(256, 256)
                if use_CBAM:
                    encoder_layers['CBAM_'+n] = CBAM(256)
        encoder_layers['GAP'] = nn.AdaptiveAvgPool1d(GAP) # (batch_size, 256, GAP)
        encoder_layers['Flatten'] = Flatten() # (batch_size, 256*GAP)
        self.Encoder = nn.Sequential(encoder_layers)

        self.Decoder = BasicLinearModule(256 * GAP, 256)
        self.Predictor = BasicPredictor(256, output_size, tasktype=tasktype)


def get_charCNN(baselinemodel):
    baselinemodel.Embedding = CharConvModule(numFiltersConv1=44, filterLenConv1=5,
                                    numFiltersConv2=40, filterLenConv2=15,
                                    numFiltersConv3=44, filterLenConv3=25)
    return baselinemodel


def get_resnet18(output_size):
    from NvTK import resnet18
    model = BaselineCNN(output_size)
    model = resnet18(output_size)
    return model


def get_resnet50(output_size):
    from NvTK import resnet50
    model = resnet50(output_size)
    return model


def get_resnet101(output_size):
    from NvTK import resnet101
    model = resnet101(output_size)
    return model



# test 
# for channel in (8, 32, 128, 512):
#     model = BaselineExprCNN(output_size=100, out_planes=channel)
#     print(model)

