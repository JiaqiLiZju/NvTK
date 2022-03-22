# NvTK

Source code used for ```Systematic evaluation of deep learning for mapping sequence to single-cell data using NvTK```

NvTK (NvwaToolKit), is a systemmatic and easy-using deep learning software in genomics. NvTK support modern deep learning achitectures in genomics, such as Residual Module, ResNet, Attention Module, CBAM, Transformer and so on. 

It's quite easy to train a deep learning model using a pre-defined model architecture in NvTK. I've re-implemented several published models in NvTK. At the same time, NvTK also support to automatically (or manually) search the best hyper-parameters of model architecture. Moreover, custumed and complicated model could be build with low-level modules in NvTK (NvTK.Trainer and Explainer always help me a lot). Importantly, NvTK is also easy to be extended with advanced deep learning modules based on pytorch. 

ps. Nvwa, the name of a mother god in ancient Chinese legend, is a deep learning–based strategy to predict expression landscapes and decipher regulatory elements (Filters) at the single-cell level. See our previous work in https://github.com/JiaqiLiZju/Nvwa.

## Requirements
- Python packages
```
python>=3.7
h5py-2.10.0
sklearn
torch
networkx
tensorboard
captum-0.5.0
pillow
ray[tune]-v1.10.0
```
pip install -U "ray[tune]"

- external softwares
```
meme-5.4.1
homer2
```
<!-- biopython-1.79 -->

## Note
- 2022.03.01: NvTK is quite unstable under activate development.

