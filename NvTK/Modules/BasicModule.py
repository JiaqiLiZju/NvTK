import random, time, logging
import numpy as np

import torch
from torch import nn

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    m.bias.data.zero_()
                logging.debug("init Conv param...")

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    m.bias.data.zero_()
                logging.debug("init Linear param...")

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                logging.debug("init BatchNorm param...")

            elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.all_weights[0][0])
                nn.init.orthogonal_(m.all_weights[0][1])
                nn.init.orthogonal_(m.all_weights[1][0])
                nn.init.orthogonal_(m.all_weights[1][1])
                logging.debug("init LSTM param...")

    def initialize_weights_from_pretrained(self, pretrained_net_fname):
        pretrained_dict = torch.load(pretrained_net_fname)
        net_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict)
        self.load_state_dict(net_state_dict)
        logging.debug("params loaded from: %s" % pretrained_net_fname)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save(self, fname=None):
        if fname is None:
            fname = time.strftime("model" + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), fname)
        return fname


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class EXP(nn.Module):
    '''Exp activation'''
    def forward(self, x):
        return x.exp()


class BasicConv1d(BasicModule):
    '''
    Basic Convolutional Module
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    dropout=True, dropout_args={'p':0.5},
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, 
                        **conv_args)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.dropout = nn.Dropout(**dropout_args) if dropout else None
        self.pool = pool(**pool_args) if pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)
        logging.debug(x.shape)
        return x


class BasicRNNModule(BasicModule):
    '''
    Basic RNN(LSTM) Module in batch-first style
    '''
    def __init__(self, LSTM_input_size=512, LSTM_hidden_size=512, LSTM_hidden_layes=2):
        super().__init__()
        self.rnn_hidden_state = None
        self.rnn = nn.LSTM(
            input_size=LSTM_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_hidden_layes,
            batch_first=True, # batch, seq, feature
            bidirectional=True,
        )
    def forward(self, input):
        output, self.rnn_hidden_state = self.rnn(input, None)
        logging.debug(output.shape)
        return output


class BasicLinearModule(BasicModule):
    '''
    Basic Linear Module
    '''
    def __init__(self, input_size, output_size, bias=False, 
                    bn=True, 
                    activation=nn.ReLU, activation_args={}, 
                    dropout=True, dropout_args={'p':0.5}):
        super().__init__()
        self.out_channels = output_size
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.bn = nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.dropout = nn.Dropout(**dropout_args) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        logging.debug(x.shape)
        return x


class BasicPredictor(nn.Module):
    '''
    BasicPredictor supprt tasks of 'none', 'binary_classification', 'classification', 'regression';
    'none' : nullify the BaiscPredictor with identity
    '''
    def __init__(self, input_size, output_size, tasktype='binary_classification'):
        super().__init__()
        self.supported_tasks = ['none', 'binary_classification', 'classification', 'regression']

        self.input_size = input_size
        self.tasktype = tasktype
        
        self.Map = nn.Linear(input_size, output_size, bias=True)
        self.switch_task(tasktype) # init self.Map

    def forward(self, x):
        return self.Pred(self.Map(x))

    def switch_task(self, tasktype):
        msg = 'tasktype: %s not supported, check the document' % tasktype
        assert tasktype in self.supported_tasks, msg

        if tasktype == 'none':
            self.Map = nn.Sequential()
            self.Pred = nn.Sequential()
        elif tasktype == 'classification':
            self.Pred = nn.Softmax(dim=1)
        elif tasktype == 'binary_classification':
            self.Pred = nn.Sigmoid()
        elif tasktype == 'regression':
            self.Pred = nn.Sequential()

        self.tasktype = tasktype
    
    # TODO
    # def finetune(self, output_size, tasktype='binary_classification'):
    #     self.Map = nn.Linear(input_size, output_size, bias=True)
    #     self.Pred = nn.Sequential()

    #     self.tasktype = tasktype
    #     self.switch_task(tasktype)

    def current_task(self):
        return self.tasktype

    def remove(self):
        '''
        Predictor.remove: replace predictor with null Sequential,
        same as switch_task('none')
        '''
        self.switch_task('none')

