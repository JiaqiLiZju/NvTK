import sys
import torch
from .Config import *
from .Evaluator import calculate_roc, calculate_pr
from .Explainer import get_activate_W, meme_generate, save_activate_seqlets

class NvTKCLI(object):
    def __init__(self, config):
        super().__init__()

        if isinstance(config, 'str'):
            self.config = load_config_from_json(config)
        else:
            self.config = config
        self.modes = parse_modes_from_config(self.config)
        self.device = torch.device("cuda") # TODO

        self.train_loader, self.validate_loader, self.test_loader = generate_dataloader_from_config(self.config)

        self.model = get_model_from_config(self.config)
        self.optimizer = get_optimizer_from_config(self.config, self.model)
        self.criterion = get_criterion_from_config(self.config)

        trainer_args = parse_trainer_args(self.config)
        self.trainer = Trainer(self.model, 
                            self.criterion, 
                            self.optimizer, 
                            self.device, **trainer_args)

    def execute(self):
        '''execute nvtk'''
        for mode in self.modes:
            assert mode in ["hpo", "train", "evaluate", "explain"]
        for mode in self.modes:
            if mode == "hpo":
                self.execute_nvtk_hpo()
            if mode == "train":
                self.execute_nvtk_train()
            elif mode == "evaluate":
                self.execute_nvtk_evaluate()
            elif mode == "explain":
                self.execute_nvtk_explain()

    def execute_nvtk_hpo(self):
        pass

    def execute_nvtk_train(self):
        # train
        self.trainer.train_until_converge(self.train_loader, 
                                    self.validate_loader, 
                                    self.test_loader, EPOCH=500)

        # reload best model
        self.model = self.trainer.load_best_model()

    def execute_nvtk_evaluate(self):
        # reload best model
        self.model = self.trainer.load_best_model()

        # predict test-set
        _, _, test_predictions, test_targets = self.trainer.predict(self.test_loader)
        # metric test-set
        fpr, tpr, roc_auc = calculate_roc(test_targets, test_predictions)
        auroc = [roc_auc[k] for k in roc_auc.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+

        p, r, average_precision = calculate_pr(test_targets, test_predictions)
        aupr = [average_precision[k] for k in average_precision.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+

        pd.DataFrame({"auroc":auroc, "aupr":aupr}, index=task_name).T.to_csv("Metric.csv")

    def execute_nvtk_explain(self):
        # explain based on feature-map
        W = get_activate_W(self.model, model.Embedding.conv, self.test_loader, motif_width=3)
        meme_generate(W, output_file='meme.txt', prefix='Filter_')

        save_activate_seqlets(model, model.Embedding.conv, self.test_loader, threshold=0.999,
                                out_fname='seqlets.fasta', motif_width=3)


def execute_nvtk(config):
    '''execute nvtk'''
    if isinstance(config, 'str'):
        config = load_config_from_json(config)
    
    modes = parse_modes_from_config(config)
    for mode in modes:
        assert mode in ["hpo", "train", "evaluate", "explain"]
    for mode in modes:
        if mode == "hpo":
            execute_nvtk_hpo(config)
        if mode == "train":
            execute_nvtk_train(config)
        elif mode == "evaluate":
            execute_nvtk_evaluate(config)
        elif mode == "explain":
            execute_nvtk_explain(config)

def execute_nvtk_hpo(config):
    pass

def execute_nvtk_train(config):
    trainer_args = parse_trainer_args(config)
    trainer = Trainer(model, criterion, optimizer, device, **trainer_args)
    # train
    trainer.train_until_converge(train_loader, validate_loader, test_loader, EPOCH=500)
    # reload best model
    model = trainer.load_best_model()
    return model

def execute_nvtk_evaluate(config):
    pass


def execute_nvtk_explain(config):
    pass


def initialize_logger(output_path, verbosity=1):
    """
    Initializes the logger for NvTK.
    This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.

    Parameters
    ----------
    output_path : str
        The path to the output file where logs will be written.

    verbosity : int, {2, 1, 0}
        Default is 1. The level of logging verbosity to use.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    """
    logger = logging.getLogger("nvtk")
    # check if logger has already been initialized
    if len(logger.handlers):
        return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(output_path)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)

    stdout_formatter = logging.Formatter(
        "%(asctime)s - %(message)s")

    stdout_handle = logging.StreamHandler(sys.stdout)
    stdout_handle.setFormatter(stdout_formatter)
    stdout_handle.setLevel(logging.INFO)
    logger.addHandler(stdout_handle)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--config", dest="config", default=None, type=str)
    args = parser.parse_args()

    if args.config is not None:
        nvtk = NvTKCLI(args.config)
        nvtk.execute()


import os, time, shutil, logging, argparse
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from .Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--mode", dest="mode", default="train")
parser.add_argument("--gpu-device", dest="device_id", default="0")
parser.add_argument("--trails", dest="trails", default="explainable")
parser.add_argument("--subset_task_by", dest="subset_task_by", default=None, type=str)
parser.add_argument("--subset_task", dest="subset_task", default=None, type=str)
parser.add_argument("--sample_size", dest="sample_size", default=None, type=int)
parser.add_argument("--pos", dest="pos", default=None, type=int)
parser.add_argument("--left_pos", dest="left_pos", default=None, type=int)
parser.add_argument("--right_pos", dest="right_pos", default=None, type=int)
parser.add_argument("--numFiltersConv1", dest="numFiltersConv1", default=None, type=int)
parser.add_argument("--filterLenConv1", dest="filterLenConv1", default=None, type=int)
parser.add_argument("--pooltype", dest="pooltype", default=None, type=str)
parser.add_argument("--Pool1", dest="Pool1", default=None, type=int)
parser.add_argument("--activation", dest="activation1", default=None, type=str)
parser.add_argument("--globalpoolsize", dest="globalpoolsize", default=None, type=int)
parser.add_argument("--use_BN", dest="use_BN", action="store_true", default=False)
parser.add_argument("--use_CBAM", dest="use_CBAM", action="store_true", default=False)
parser.add_argument("--use_Transformer", dest="use_transformer", action="store_true", default=False)
parser.add_argument("--use_ResNet", dest="use_ResNet", default=None, type=int)
parser.add_argument("--use_DeepCNN", dest="use_DeepCNN", default=None, type=int)
parser.add_argument("--use_CharCNN", dest="use_CharCNN", action="store_true", default=False)
parser.add_argument("--tower_by", dest="tower_by", default="Celltype")
parser.add_argument("--tower_hidden", dest="tower_hidden", default=None, type=int)
parser.add_argument("--regression", dest="regression", action="store_true", default=False)
parser.add_argument("--use_data_rc_augment", dest="use_data_rc_augment", action="store_true", default=False)
parser.add_argument("--use_data_shift_augment", dest="use_data_shift_augment", action="store_true", default=False)
parser.add_argument("--use_hard_mining_augment", dest="use_hard_mining_augment", action="store_true", default=False)
parser.add_argument("--patience", dest="patience", default=10, type=int)
parser.add_argument("--lr", dest="lr", default=1e-5, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)
parser.add_argument("--EPOCH", dest="EPOCH", default=500, type=int)
parser.add_argument("--l1_weight", dest="l1_weight", default=0, type=float)
parser.add_argument("--use_focal_loss", dest="use_focal_loss", action="store_true", default=False)
parser.add_argument("--use_scMTLoss", dest="use_scMTLoss", action="store_true", default=False)
parser.add_argument("--scMTLoss_level", dest="scMTLoss_level", default="Celltype", type=str)
parser.add_argument("--metric_sample", dest="metric_sample", default=100, type=int)

args = parser.parse_args()
data = args.data
device_id, mode = args.device_id, args.mode
trails_fname = args.trails # default
subset_task, subset_task_by = args.subset_task, args.subset_task_by
sample_size = args.sample_size
pos, left_pos, right_pos = args.pos, args.left_pos, args.right_pos
numFiltersConv1, filterLenConv1 =  args.numFiltersConv1, args.filterLenConv1
activation1 = args.activation1
pooltype, Pool1, globalpoolsize = args.pooltype, args.Pool1, args.globalpoolsize
tower_by, tower_hidden = args.tower_by, args.tower_hidden
patience, lr, EPOCH = args.patience, args.lr, args.EPOCH
l1_weight, use_focal_loss, use_scMTLoss = args.l1_weight, args.use_focal_loss, args.use_scMTLoss
scMTLoss_level = args.scMTLoss_level
batch_size = args.batch_size
use_data_rc_augment, use_data_shift_augment = args.use_data_rc_augment, args.use_data_shift_augment
use_ResNet, use_cbam, use_transformer, use_CharCNN, use_DeepCNN = args.use_ResNet, args.use_CBAM, args.use_transformer, args.use_CharCNN, args.use_DeepCNN
pred_prob = not args.regression
metric_sample = args.metric_sample
use_BN = args.use_BN

os.makedirs('./Log', exist_ok=True)
os.makedirs('./Figures', exist_ok=True)

# set random_seed
set_random_seed()
set_torch_benchmark()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./Log/log_HyperBest.' + mode + '.%m%d.%H:%M:%S.txt'),
                    filemode='w')
logging.info(args)

## change
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
logging.info(device)

# unpack datasets
species = os.path.basename(data).split('.')[1]
logging.info("#"*60)
logging.info("switching datasets: %s" % species)

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
anno = pd.DataFrame(anno, columns=["Cell", "Species", "Celltype", "Cluster"])
if subset_task and subset_task_by in ["Cell", "Species", "Celltype", "Cluster"]:
    subset_mask = anno[subset_task_by].values == subset_task
    anno = anno.loc[subset_mask]
    celltype = celltype[subset_mask]
    logging.info(anno.shape)
    logging.info(celltype.shape)
    
if sample_size:
    if mode == "train":
        anno = anno.sample(sample_size)
        anno.to_csv("./Log/sampled.anno.csv")
    else:
        anno = pd.read_csv("./Log/sampled.anno.csv", index_col=0)
        
    sample_idx = [c in anno.Cell.values for c in celltype]
    celltype = celltype[sample_idx]
    logging.info(anno.shape)
    logging.info(celltype.shape)

anno_cnt = anno.groupby(tower_by, sort=False)["Species"].count()

if mode != "test":
    x_train = h5file["train_data"][:].astype(np.float32)
    y_train_onehot = h5file["train_label"][:].astype(np.float32)
    x_val = h5file["val_data"][:].astype(np.float32)
    y_val_onehot = h5file["val_label"][:].astype(np.float32)

    if subset_task and subset_task_by in ["Cell", "Species", "Celltype", "Cluster"]:
        y_train_onehot = y_train_onehot[:,subset_mask]
        y_val_onehot = y_val_onehot[:,subset_mask]
    if sample_size:
        y_train_onehot = y_train_onehot[:,sample_idx]
        y_val_onehot = y_val_onehot[:,sample_idx]

    logging.info(x_train.shape)
    logging.info(x_val.shape)
    logging.info(y_train_onehot.shape)
    logging.info(y_val_onehot.shape)

    logging.debug(x_train[0,:,:5])
    logging.debug(y_train_onehot[0,:5])
    logging.debug(x_val[0,:,:5])
    logging.debug(y_val_onehot[0,:5])

x_test = h5file["test_data"][:].astype(np.float32)
y_test_onehot = h5file["test_label"][:].astype(np.float32)
if subset_task and subset_task_by in ["Cell", "Species", "Celltype", "Cluster"]:
    y_test_onehot = y_test_onehot[:,subset_mask]
if sample_size:
    y_test_onehot = y_test_onehot[:,sample_idx]

logging.info(x_test.shape)
logging.info(y_test_onehot.shape)

logging.debug(x_test[0,:,:5])
logging.debug(y_test_onehot[0,:5])

train_gene = h5file["train_gene"][:]
val_gene = h5file["val_gene"][:]
test_gene = h5file["test_gene"][:]

logging.debug(train_gene[:5])
logging.debug(val_gene[:5])
logging.debug(test_gene[:5])

h5file.close()

# trails
if trails_fname == "best_manual":
    params = best_params
    logging.info("using best manual model")
elif trails_fname == "explainable":
    params = explainable_params
    logging.info("using explainable model")
elif trails_fname == "NIN":
    params = explainable_params
    params["is_NIN"] = True
    logging.info("using NIN model")
else:
    trials = pickle.load(open(trails_fname, 'rb'))
    best = trials.argmin
    params = space_eval(param_space, best)
    params['leftpos'] = 10000 - int(params['pos'])
    params['rightpos'] = 10000 + int(params['pos'])
    logging.info("using model from trails:\t%s", trails_fname)

params['is_spatial_transform'] = False
params['anno_cnt'] = anno_cnt
params['pred_prob'] = pred_prob

if pos:
    params['leftpos'] = 10000 - pos
    params['rightpos'] = 10000 + pos
if left_pos is not None:
    params['leftpos'] = left_pos
if right_pos is not None:
    params['rightpos'] = right_pos
if numFiltersConv1:
    params['numFiltersConv1'] = numFiltersConv1
if filterLenConv1:
    params['filterLenConv1'] = filterLenConv1
if pooltype:
    params['pooltype'] = pooltype
if Pool1:
    params['maxPool1'] = Pool1
if globalpoolsize:
    params['globalpoolsize'] = globalpoolsize
if tower_hidden:
    params['tower_hidden'] = tower_hidden
if use_cbam:
    params['CBAM1'] = use_cbam
if activation1:
    params['activation'] = activation1
if use_BN:
    params['isBN'] = use_BN

# define datasets parameters
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size

logging.info(params)

# define train_datasets
if mode != "test":
    x_train = x_train[:, :, leftpos:rightpos]
    x_val = x_val[:, :, leftpos:rightpos]
    logging.info(x_train.shape)
    logging.info(x_val.shape)
    logging.debug(x_train[0,:,:5])
    logging.debug(x_val[0,:,:5])

    if use_data_rc_augment:
        logging.info("using sequence reverse complement augment...")

        x_train, y_train_onehot = seq_rc_augment(x_train, y_train_onehot)
        logging.info(x_train.shape)

    if use_data_shift_augment:
        logging.info("using sequence shift left and right augment...")

        x_train, y_train_onehot = seq_shift_augment(x_train, y_train_onehot)
        logging.info(x_train.shape)

    train_loader = DataLoader(list(zip(x_train, y_train_onehot)), batch_size=batch_size,
                                shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    validate_loader = DataLoader(list(zip(x_val, y_val_onehot)), batch_size=batch_size, 
                                shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

# test_loader
x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)
logging.debug(x_test[0,:,:5])

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=batch_size, 
                            shuffle=False, num_workers=2, drop_last=False, pin_memory=True)


model = CNN(params).to(device)
logging.info("weights inited and embedding weights loaded")
logging.info(model.__str__())

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9)
criterion = nn.BCELoss().to(device)

trainer = Trainer(model, criterion, optimizer, device)
logging.info('\n------'+'train'+'-------\n')

if mode == "train" or mode == "resume":
    trainer.train_until_converge(train_loader, validate_loader, test_loader, 500)

