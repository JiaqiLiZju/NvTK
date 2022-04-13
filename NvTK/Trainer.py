import os, time, shutil, logging, pickle, random
import numpy as np
import torch

from .Evaluator import calculate_correlation, calculate_roc

class Trainer(object):
    '''
    Model Trainer
    '''
    def __init__(self, model, criterion, optimizer, device, tasktype="regression",
                    metric_sample=100, item_sample=50000, patience=10,
                    resume=False, resume_dirs=None,
                    use_tensorbord=False, tensorbord_args={}):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience

        # task related
        self.pred_prob = tasktype != "regression"
        # sample tasks for calculate metric
        self.metric_sample = metric_sample
        self.item_sample = item_sample

        # Trainer will maintain logs information
        self.logs = {
            "train_batch_loss_list":[], 
            "val_batch_loss_list":[], 
            "test_batch_loss_list":[],

            "train_loss_list":[], 
            "val_loss_list":[], 
            "test_loss_list":[],

            "val_metric_list":[], 
            "test_metric_list":[], 
            "train_metric_list":[],
            "lrs":[],

            "best_val_loss":np.inf,
            "best_val_r":-np.inf,
            "best_val_epoch":0,
            }

        # TODO resume model training
        if resume and resume_dirs:
            pass
        
        # TODO capture error when loading tensorbord in lower version
        if use_tensorbord:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorbord = True
            self.writer = SummaryWriter(**tensorbord_args)
        else:
            self.tensorbord = False

    def train_until_converge(self, train_loader, validate_loader, test_loader, EPOCH, resume=False, verbose_step=5):
        # graph
        if self.tensorbord:
            try:
                self.add_graph(next(iter(train_loader))[0][:2].shape)
            except BaseException as e:
                logging.warning("tensorbord cannot added graph")
                logging.warning(e)

        for epoch in range(EPOCH):
            # train
            train_batch_loss, train_loss = self.train_per_epoch(train_loader, epoch, verbose_step=20)

            # train metrics
            train_batch_loss, train_loss, train_pred_prob, train_target_prob = self.predict(train_loader)
            train_metric = self.evaluate(train_pred_prob, train_target_prob)

            # validation metrics
            val_batch_loss, val_loss, val_pred_prob, val_target_prob = self.predict(validate_loader)
            val_metric = self.evaluate(val_pred_prob, val_target_prob)

            # test metrics
            test_batch_loss, test_loss, test_pred_prob, test_target_prob = self.predict(test_loader)
            test_metric = self.evaluate(test_pred_prob, test_target_prob)

            # lr_scheduler
            _lr = self.optimizer.param_groups[0]['lr']
            # lr_scheduler.step(val_loss)

            # logs
            logging.info("Train\t Accuracy: %.4f\t Loss: %.4f\t\n" % (train_metric, train_loss))
            logging.info("Eval\t Accuracy: %.4f\t Loss: %.4f\t\n" % (val_metric, val_loss))
            
            self.logs['train_batch_loss_list'].extend(train_batch_loss)
            self.logs['val_batch_loss_list'].extend(val_batch_loss)
            self.logs['test_batch_loss_list'].extend(test_batch_loss)
            self.logs['train_loss_list'].append(train_loss)
            self.logs['val_loss_list'].append(val_loss)
            self.logs['test_loss_list'].append(test_loss)
            self.logs['train_metric_list'].append(train_metric)
            self.logs['val_metric_list'].append(val_metric)
            self.logs['test_metric_list'].append(test_metric)
            self.logs['lrs'].append(_lr)
            self.show_trainer_log()

            if self.tensorbord:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/test', test_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_metric, epoch)
                self.writer.add_scalar('Accuracy/test', test_metric, epoch)

                try:
                    self.writer.add_histogram('Embedding.conv.bias', self.model.Embedding.conv.bias, epoch)
                    self.writer.add_histogram('Embedding.conv.weight', self.model.Embedding.conv.weight, epoch)
                except:
                    pass

            # update best
            if val_loss < self.logs["best_val_loss"] or val_metric > self.logs["best_val_r"]:
                self.logs["best_val_loss"] = val_loss
                self.logs["best_val_r"] = val_metric
                self.logs["best_val_epoch"] = epoch

                logging.info("Eval\t Best Eval Accuracy: %.4f\t Loss: %.4f\t at Epoch: %d\t lr: %.8f\n" % (val_metric, val_loss, epoch, _lr))
                logging.info("Eval\t Test Accuracy: %.4f\t Loss: %.4f\n" % (test_metric, test_loss))
                
                self.save_checkpoint(best=True)

            # or checkpoint
            elif epoch % 20 == 1:
                self.save_checkpoint()

            # early stop
            if epoch >= self.logs["best_val_epoch"] + self.patience: #patience_epoch:
                break
        
        # final
        fname = time.strftime("./Log/best_model@" + '%m%d_%H:%M:%S')
        shutil.copyfile("./Log/best_model.pth", fname+".params.pth")
        pickle.dump(self.logs, open(fname+".chekc_train_log.p", 'wb'))

        # final logs
        self.show_trainer_log(final=True)
        logging.info("\n"+self.model.__str__()+'\n')
        logging.info("Best Val Loss\t%.8f\t@epoch%d" % (self.logs['best_val_loss'], self.logs["best_val_epoch"]))
        logging.info("Best Val Metric\t%.8f\t@epoch%d" % (self.logs['best_val_r'], self.logs["best_val_epoch"]))
        logging.info("Best Test Metric\t%.8f\t@epoch%d" % (np.max(self.logs['test_metric_list']), np.argmax(self.logs['test_metric_list'])))
        logging.info("Best Test Loss\t%.8f\t@epoch%d" % (np.min(self.logs['test_loss_list']), np.argmin(self.logs['test_loss_list'])))

        self.load_best_model()

    def train_per_epoch(self, train_loader, epoch, verbose_step=5):
        # train one epoch
        batch_losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = self.train_batch(data, target)

            if batch_idx % verbose_step == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
            
            batch_losses.append(loss)
            average_loss = np.average(batch_losses)
        return batch_losses, average_loss
    
    def train_batch(self, data, target):
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def predict(self, data_loader):
        batch_losses, all_predictions, all_targets = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                test_loss = self.criterion(output, targets)
                batch_losses.append(test_loss.cpu().item())
                all_predictions.append(output.cpu().data.numpy())
                all_targets.append(targets.cpu().data.numpy())
        average_loss = np.average(batch_losses)
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        return batch_losses, average_loss, all_predictions, all_targets

    def evaluate(self, predict_prob, target_prob, sample_tasks=True, sample_items=True):
        item_size, output_size = target_prob.shape

        if self.metric_sample < output_size and sample_tasks:
            metric_sample_idx = random.sample(range(output_size), self.metric_sample)
            logging.info("sampled %d tasks for metric" % self.metric_sample)
            target_prob = target_prob[:, metric_sample_idx]
            predict_prob = predict_prob[:, metric_sample_idx]

        if self.item_sample < item_size and sample_items:
            item_sample_idx = random.sample(range(item_size), self.item_sample)
            logging.info("sampled %d items for metric" % self.item_sample)
            target_prob = target_prob[item_sample_idx, :]
            predict_prob = predict_prob[item_sample_idx, :]

        if self.pred_prob:
            fpr, tpr, roc_auc = calculate_roc(target_prob, predict_prob)
            roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
            metric = np.mean(roc_l)
        
        else:
            correlation, pvalue = calculate_correlation(target_prob, predict_prob)
            correlation_l = [correlation[k] for k in correlation.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
            metric = np.mean(correlation_l)

        return metric

    def add_graph(self, input_shape=(2, 4, 1000)):
        self.writer.add_graph(self.model, torch.rand(input_shape).to(self.device))

    def load_best_model(self, fpath="./Log/best_model.pth"):
        self.model.load(fpath)
        return self.model

    def get_current_model(self):
        return self.model

    def save_checkpoint(self, best=False):
        os.makedirs("./Log", exist_ok=True)
        self.model.save("./Log/chekc_model.pth")
        torch.save(self.model, "./Log/chekc_model.p")
        pickle.dump(self.logs, open("./Log/chekc_train_log.p", 'wb'))

        if best:
            self.model.save("./Log/best_model.pth")
            torch.save(self.model, "./Log/best_model.p")
    
    def show_trainer_log(self, final=False):
        show_train_log(self.logs['train_loss_list'], 
                    self.logs['train_metric_list'], 
                    self.logs['val_loss_list'], 
                    self.logs['val_metric_list'],
                    self.logs['test_loss_list'], 
                    self.logs['test_metric_list'], 
                    self.logs['lrs'], output_fname='current_logs.pdf')
        show_train_log(loss_train=self.logs['train_loss_list'], 
                    loss_val=self.logs['val_loss_list'], 
                    loss_test=self.logs['test_loss_list'], 
                    output_fname='current_logs_loss.pdf')
        show_train_log(loss_val=self.logs['val_batch_loss_list'], output_fname='current_batch_loss_val.pdf')
        show_train_log(loss_test=self.logs['test_batch_loss_list'], output_fname='current_batch_loss_test.pdf')
        show_train_log(loss_train=self.logs['train_batch_loss_list'], output_fname='current_batch_loss_train.pdf')
        show_train_log(loss_val=self.logs['val_loss_list'], output_fname='current_loss_val.pdf')
        show_train_log(loss_test=self.logs['test_loss_list'], output_fname='current_loss_test.pdf')
        show_train_log(loss_train=self.logs['train_loss_list'], output_fname='current_loss_train.pdf')
        show_train_log(acc_val=self.logs['train_metric_list'], output_fname='current_acc_train.pdf')
        show_train_log(acc_val=self.logs['val_metric_list'], output_fname='current_acc_val.pdf')
        show_train_log(acc_test=self.logs['test_metric_list'], output_fname='current_acc_test.pdf')
        show_train_log(lrs=self.logs['lrs'], output_fname='current_lrs.pdf')

        if final:
            show_train_log(loss_val=self.logs['val_loss_list'], output_fname='final_loss_val.pdf')
            show_train_log(loss_train=self.logs['train_loss_list'], output_fname='final_loss_train.pdf')
            show_train_log(acc_val=self.logs['train_metric_list'], output_fname='final_acc_train.pdf')
            show_train_log(acc_val=self.logs['val_metric_list'], output_fname='final_acc_val.pdf')
            show_train_log(acc_test=self.logs['test_metric_list'], output_fname='final_acc_test.pdf')
            show_train_log(lrs=self.logs['lrs'], output_fname='final_lrs.pdf')


def show_train_log(loss_train=None, loss_val=None, acc_val=None, 
                   loss_test=None, acc_test=None,
                   lrs=None,
                    fig_size=(12,8),
                    save=True,
                    output_dir='Figures',
                    output_fname="Training_loss_log.pdf",
                    style="seaborn-colorblind",
                    fig_title="Training Log",
                    dpi=500):
    
    os.makedirs(output_dir, exist_ok=True)
    
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.style.use(style)
    plt.figure()

    if loss_train:
        plt.plot(range(1, len(loss_train)+1), loss_train, 'b', label='Training Loss')
    if loss_val:
        plt.plot(range(1, len(loss_val)+1), loss_val, 'r', label='Validation Loss')
    if loss_test:
        plt.plot(range(1, len(loss_test)+1), loss_test, 'black', label='Test Loss')
    rate = 1
    if acc_val:
        plt.plot(range(1, len(acc_val)+1), list(map(lambda x: x*rate, acc_val)), 'g', label=str(rate)+'X Validation Accuracy')
    if acc_test:
        plt.plot(range(1, len(acc_test)+1), list(map(lambda x: x*rate, acc_test)), 'purple', label=str(rate)+'X Test Accuracy')
    if lrs:
        rate = int(1/lrs[0])
        plt.plot(range(1, len(lrs)+1), list(map(lambda x: x*rate, lrs)), 'y', label=str(rate)+'X Learning Rates')
    plt.title(fig_title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()

