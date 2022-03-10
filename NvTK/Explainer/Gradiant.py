import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from captum.attr import Saliency, LayerConductance


def contribution_input_grad(model, input_tensor, multiply_by_inputs=True):
    input_tensor.requires_grad_()
    y = model(input_tensor).sum()
    y.backward()

    grad = input_tensor.grad
    if multiply_by_inputs:
        grad = grad * input_tensor

    return grad.data.cpu().numpy()


def deep_explain_saliancy(model, input_tensor, n_class=1, use_abs=True):
    saliency = Saliency(model)
    saliency_val_l = []
    for i_class in range(n_class):
        attribution = saliency.attribute(input_tensor, target=i_class)
        saliency_vals = attribution.cpu().data.numpy()
        if use_abs:
            saliency_vals = np.abs(saliency_vals)
        saliency_val_l.append(saliency_vals)
    return np.array(saliency_val_l)


def input_saliancy_location(model, input_tensor, n_class=3, use_abs=True):
    saliency_val_l = deep_explain_saliancy(model, input_tensor, n_class=n_class, use_abs=use_abs)
    saliency_val = saliency_val_l.mean(0).mean(0).mean(0)
    saliency_length = pd.DataFrame(enumerate(saliency_val), columns=["location","saliancy"])
    return saliency_length


def plot_saliancy_location(model, input_tensor, n_class=3, use_abs=True):
    saliency_length = input_saliancy_location(model, input_tensor, n_class=n_class, use_abs=use_abs)
    plt.figure(figsize=(30,4))
    ax = sns.lineplot(x="location", y="saliancy", data=saliency_length)
    plt.show()
    plt.close()


def deep_explain_layer_conductance(model, model_layer, input_tensor, n_class=1):
    layer_cond = LayerConductance(model, model_layer)
    cond_val_l = []
    for i_class in range(n_class):
        attribution = layer_cond.attribute(input_tensor, target=i_class, internal_batch_size=32)
        cond_vals = attribution.detach().numpy()
        cond_val_l.append(cond_vals)
    return np.array(cond_val_l)


def label_neuron_importance(model, model_layer, input_tensor, label):
    n_class = len(label)
    imp = deep_explain_layer_conductance(model, model_layer, input_tensor, n_class=n_class)
    imp = imp.mean(-1).mean(1)
    df = pd.DataFrame(imp, index=label)
    return df


def plot_label_neuron_importance(model, model_layer, input_tensor, label):
    df = label_neuron_importance(model, model_layer, input_tensor, label)
    plt.figure(figsize=(30,4), cmap="vlag")
    ax = sns.heatmap(df)
    plt.show()
    plt.close()

