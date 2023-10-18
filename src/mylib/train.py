import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl


def calc_f1(y_hat, y):
    x = (y_hat > 0).to(torch.int32)
    b, l, t = x.size()
    x = x.reshape((b * l, t))
    y = y.reshape((b * l, t))
    tp = ((x == 1) & (y == 1)).to(torch.float32).sum(dim=0)
    fp = ((x == 1) & (y == 0)).to(torch.float32).sum(dim=0)
    fn = ((x == 0) & (y == 1)).to(torch.float32).sum(dim=0)

    tp_mean = tp.mean()
    fp_mean = fp.mean()
    fn_mean = fn.mean()
    f1_micro = 2 * tp_mean / (2 * tp_mean + fp_mean + fn_mean)
    return {'f1_micro': f1_micro}


def calc_metrics(y_hat, y, metrics_func=[]):
    metrics = {}
    for f in metrics_func:
        metrics.update(f(y_hat, y))
    return metrics


class TokenClfModel(nn.Module):
    def __init__(self, embedder, embed_size=512, tags=400):
        super().__init__()
        self.embedder = embedder
        self.out = nn.Linear(embed_size, tags)

    def forward(self, x):
        input_id = x['input_ids'].squeeze(1)
        mask = x['attention_mask']
        x = self.embedder(input_ids=input_id, attention_mask=mask, return_dict=True)
        x = x.last_hidden_state
        y = self.out(x)
        return y


class ClfModel(pl.LightningModule):

    def __init__(self, model, loss, optimizer, scheduler, metric_functions):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_functions = metric_functions
        self.train_y_out = []
        self.train_y = []
        self.val_y_out = []
        self.val_y = []

    def calc_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _common_step(self, batch, batch_idx):
        (x, y) = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {
            'loss': loss,
            'y': y_hat
        }

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = res['loss']
        # self.train_y_out.append(res['y'])
        # self.train_y.append(batch[1])
        self.log_dict(
            {'loss/Train': loss.detach().cpu().item()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.logger.log_metrics(
            {
                'loss/Train': loss.detach().cpu().item(), 
                'grad_norm': self.calc_grad_norm(),
                'lr': self.optimizer.param_groups[0]['lr']
            }, 
            self.global_step)
        return res

    def on_training_epoch_end(self):
        y_hat = torch.cat(self.train_y_out, dim=0)
        y = torch.cat(self.train_y, dim=0)
        
        metrics = calc_metrics(y_hat.detach().cpu(), y.detach().cpu(), self.metric_functions)
        self.log_dict(
            {n+'/Train': v for n, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for name, val in metrics.items():
            self.logger.log_metrics({name + '/Train': val}, self.global_step)

        self.train_y_out.clear()
        self.train_y.clear()

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = res['loss']
        self.val_y_out.append(res['y'])
        self.val_y.append(batch[1])
        return res

    def on_validation_epoch_end(self):
        y_hat = torch.cat(self.val_y_out, dim=0)
        y = torch.cat(self.val_y, dim=0)

        metrics = calc_metrics(y_hat.detach().cpu(), y.detach().cpu(), self.metric_functions)
        self.log_dict(
            {n+'/Valid': v for n, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for name, val in metrics.items():
            self.logger.log_metrics({name + '/Valid': val}, self.global_step)

        self.val_y_out.clear()
        self.val_y.clear()
      
    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        return [self.optimizer], [self.scheduler]