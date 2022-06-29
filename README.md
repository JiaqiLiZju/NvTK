# NvTK 

<div align=left><img src="./Figures/Logo.png" width="80px"></img></div>

Source code used for ```Systematic evaluation of deep learning for single-cell genomics data using NvTK```.

## About NvTK

NvTK (NvwaToolKit), is a systemmatic and easy-using deep learning software in genomics. NvTK support modern deep learning achitectures in genomics, such as Residual Module, ResNet, Attention Module, CBAM, Transformer and so on. 

<div align=center><img src="./Figures/NvTK.png" width="450px"></img></div>

It's quite easy to train a deep learning model using a pre-defined model architecture in NvTK. I've re-implemented several published models in NvTK. At the same time, NvTK also support to automatically (or manually) search the best hyper-parameters of model architecture. Moreover, custumed and complicated model could be build with low-level modules in NvTK (NvTK.Trainer and Explainer always help me a lot). Importantly, NvTK is also easy to be extended with advanced deep learning modules based on pytorch. 

ps. Nvwa, the name of a mother god in ancient Chinese legend, is a deep learning–based strategy to predict expression landscapes and decipher regulatory elements (Filters) at the single-cell level. See our previous work in https://github.com/JiaqiLiZju/Nvwa.

## Installation
We recommend using NvTK  with Python 3.7 or above. 

### Installing NvTK from source:

First, download the latest commits from the source repository (or download the latest version of NvTK for a stable release):
```
git clone https://github.com/JiaqiLiZju/NvTK.git
```

The `setup.py` script requires following requirements. Please make sure you have these already installed.

### Requirements

- Python packages
```
python>=3.7
numpy
pandas>=0.21
matplotlib==3.0.*
# h5py > 2.10 may returns b'strings' reading h5file
h5py=2.10.0
tqdm
scikit-learn>=0.21.2
# torch >=1.10.1 support tensorboard, and ModifyOutputHook
torch>=1.10.1
tensorboard=2.7.0
captum=0.5.0
networkx
# higher version of scipy do not support resize
pillows
```

- external softwares (optional)
```
# meme could align the deep learning motif with known TFBS database
meme-5.4.1
# homer2 could search the motif in activated seqlets
homer2
```
<!-- biopython-1.79 -->

### Installation

If you would like to locally install NvTK, you can run
```sh
python setup.py install
```

### Load packages
If you would like to load NvTK in your python scripts instead of installing locally, you can run

```
import sys
sys.path.append("PATH/TO/NvTK/")

import NvTK
```

## Tutorials

### Documents
The documentation for NvTK is available [here]().

### Manuscript case studies

The code to reproduce case studies in the manuscript is available [here]().

Each case has its own directory and README describing how to run these cases. 
We recommend consulting the step-by-step breakdown of each case study that we provide in the methods section of [the manuscript](https://doi.org/10.1101/438291) as well.  

<!--
- Case study 1 finishes in about 1.5 days on a GPU node.
- Case study 2 takes 6-7 days to run training (distributed the work across 4 v100s) and evaluation.
- Case study 3 (variant effect prediction) takes about 1 day to run. 

The case studies in the manuscript focus on developing deep learning models for classification tasks. Selene does support training and evaluating sequence-based regression models, and we have provided a [tutorial to demonstrate this](https://github.com/FunctionLab/selene/blob/master/tutorials/regression_mpra_example/regression_mpra_example.ipynb).  

-->

### Manuscript benchmarks


The manuscript examples were only tested on GPU (Our GPU, NVIDIA Tesla A100).

**Important**: The tutorials and manuscript examples were originally run on NvTK version 0.1.0 (PyTorch version 0.4.1). Please note that models created with an older version of PyTorch (such as those downloadable with the manuscript case studies) are NOT compatible with newer versions of PyTorch. If you run into errors loading trained model weights files, it is likely the result of differences in PyTorch or CUDA toolkit versions.  

## News
- 2022.03.01: NvTK is quite unstable under activate development.
- 2022.03.23: updating BenchmarksInManuscript
- 2022.04.19: updating Documents, Tutorials
