import torch
import logging
import itertools
import numpy as np
import pandas as pd


def foldchange(origin, modified):
    return modified / origin
    # return np.square(modified - origin)


# hook
class ModifyOutputHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.channels = None
        self.channel = 0

    def hook_fn(self, module, input, output):
        for channel in self.channels:
            self.channel = channel
            if isinstance(module, torch.nn.modules.conv.Conv1d):
                output_channel = output[:,self.channel,:]
                output[:,self.channel,:] = torch.zeros_like(output_channel).to(output_channel.device)#output_channel.mean()
            elif isinstance(module, torch.nn.modules.linear.Linear):
                output_channel = output[:,self.channel]
                output[:,self.channel] = torch.zeros_like(output_channel).to(output_channel.device)#output_channel.mean()
            # logging.info(output_channel[:5].cpu().detach().numpy())
            # logging.info(output_channel.mean().cpu().detach().numpy())
        return output

    def step_channel(self, idx):
        if isinstance(idx, (list, tuple)):
            self.channels = idx
        elif isinstance(idx, int):
            self.channels = [idx]

    def get_current_channel(self):
        return self.channel

    def close(self):
        self.hook.remove()


class ModifyInputHook():
    def __init__(self, module):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.channels = None
        self.channel = 0

    def hook_fn(self, module, input):
        for channel in self.channels:
            self.channel = channel
            if isinstance(module, torch.nn.modules.conv.Conv1d):
                input_channel = input[0][:,self.channel,:]
                input[0][:,self.channel,:] = torch.zeros_like(input_channel).to(input_channel.device)#input_channel.mean()
            elif isinstance(module, torch.nn.modules.linear.Linear):
                input_channel = input[0][:,self.channel]
                input[0][:,self.channel] = torch.zeros_like(input_channel).to(input_channel.device)#input_channel.mean()
            # logging.info(input_channel[:5].cpu().detach().numpy())
            # logging.info(input_channel.mean().cpu().detach().numpy())
        return input

    def step_channel(self, idx):
        if isinstance(idx, (list, tuple)):
            self.channels = idx
        elif isinstance(idx, int):
            self.channels = [idx]

    def get_current_channel(self):
        return self.channel
        
    def close(self):
        self.hook.remove()


def channel_target_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    criterion = torch.nn.BCELoss(reduction='none').to(device) # gene * cell
    target, pred_orig, loss_orig, pred_modified_foldchange = [], [], [], []

    # a normal feed-forward
    model.eval()
    with torch.no_grad():
        for x_tensor, t in data_loader:
            x_tensor = x_tensor.to(device)
            t = t.to(device)
            output = model(x_tensor)
            loss = criterion(output, t)

            target.append(t.cpu().data.numpy())
            pred_orig.append(output.cpu().data.numpy())
            loss_orig.append(loss.cpu().data.numpy())

        target = np.vstack(target)
        pred_orig = np.vstack(pred_orig)
        loss_orig = np.vstack(loss_orig)

        # feed-forward with ModifyOutputHook
        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            out_channels = hook_module.out_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            out_channels = hook_module.out_features # must hook on linear layer
        
        Modifier = ModifyOutputHook(hook_module)
        for idx in range(out_channels):
            logging.info("modifying channel_%d..." % idx)
            pred_modified, loss_modified = [], []
            Modifier.step_channel(idx)
            for x_tensor, t in data_loader:
                x_tensor = x_tensor.to(device)
                t = t.to(device)
                output = model(x_tensor) # batch_size * output_size
                loss = criterion(output, t)

                pred_modified.append(output.cpu().data.numpy())
                loss_modified.append(loss.cpu().data.numpy())
            pred_modified = np.vstack(pred_modified) 
            loss_modified = np.vstack(loss_modified) 

            fc = foldchange(pred_orig, pred_modified).mean(0) # output_size
            # fc = foldchange(loss_orig, loss_modified).mean(0) # output_size
            pred_modified_foldchange.append(fc)

        Modifier.close()
    return np.vstack(pred_modified_foldchange)


def layer_channel_combination_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    # a normal feed-forward
    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        # feed-forward with ModifyOutputHook
        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            out_channels = hook_module.out_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            out_channels = hook_module.out_features # must hook on linear layer

        Modifier = ModifyOutputHook(hook_module)
        for idx in itertools.combinations(range(out_channels), 2):
            logging.info("modifying channel_%d&%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0) # output_size
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)


def input_channel_target_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            in_channels = hook_module.in_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            in_channels = hook_module.in_features # must hook on linear layer
        Modifier = ModifyInputHook(hook_module)
        for idx in range(in_channels):
            logging.info("modifying channel_%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0)
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)


def input_layer_channel_combination_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            in_channels = hook_module.in_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            in_channels = hook_module.in_features # must hook on linear layer
        Modifier = ModifyInputHook(hook_module)
        for idx in itertools.combinations(range(in_channels), 2):
            logging.info("modifying channel_%d&%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0)
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

