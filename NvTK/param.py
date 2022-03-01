

def load_params():
    return params

def load_params_from_json(json):
    return params

def update_params(args):
    return params

def get_model_from_params(params):
    return model

def get_optimizer_from_params(params):
    return optimizer

def get_criterion_from_params(params):
    return criterion


PARAMS = {
    # datasets params
    'datasets':{
        'batchsize' : 128, 
        'leftpos' : 3500,
        'rightpos' : 13500,
    },

    # Embedding params
    'Embedding': {
        'module' : 'BasicConvEmbed',
        'out_planes' : 128,
        'kernel_size': 3,
        'bn':True,
        'activation':'ReLu',
    },

    # Encoder params
    'Encoder':{
        'module' : 'BasicConvEmbed',
        'in_planes' : 128,
        'out_planes' : 512,
        'kernel_size': 3,
        'conv_args' : {
            'stride':1, 
            'padding':0, 
            'dilation':1, 
            'groups':1, 
            'bias':False
        },
        'bn':True,
        'activation':'ReLu',
    }
    
    # conv params
    'pooltype' : 'avgpool',

    'numFiltersConv1' : 64,
    'filterLenConv1' : 7,
    # 'dilRate1' : hp.quniform('dilRate1', 1, 2, 1), # dilRate1 = 1
    'CBAM1': False,
    'reduction_ratio1' : 16,
    'maxPool1' : 7,
    'numconvlayers': {
        'numFiltersConv2' : 128,
        'filterLenConv2' : 7,
        'dilRate2' : 2,
        'CBAM2' : False,
        'reduction_ratio2': 16,
        'maxPool2' : 7,
        'numconvlayers1' : {
            'numFiltersConv3' : 512,
            'filterLenConv3' : 9,
            'dilRate3' : 3,
            'CBAM3' : False,
            'reduction_ratio3' : 16,
            'maxPool3' : 9,
            'numconvlayers2' : 'three'
        }
    },
    
    'globalpoolsize' : 8,
    
    # RNN params
    'numRNNlayers': {
            'numRNNlayers1' : 'zero'},
    
    # FC params
    'dense1' : 2048,
    'dropout1' : 0.7,
    'numdenselayers' : {
        'dense2' :2048, 
        'dropout2' : 0.5,
        'numdenselayers1' : {
            'dense3' : 2048,
            'dropout3' : 0.3,
            'numdenselayers2' : 'three' ,
        }
    },

    # TOWER params
    'tower_hidden': 32,
    'tower_drop' : 0.1,
}
