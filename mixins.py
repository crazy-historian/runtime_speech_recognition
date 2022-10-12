
import torch
from torch import nn
from torchmetrics import (MetricCollection, MeanAbsoluteError,
                          MeanAbsolutePercentageError,
                          MeanSquaredError, R2Score,
                          Accuracy, Precision, Recall, F1Score)



class ClassifierMetricsMixin:
    def _init_metrics(self, loss_criterion: str = 'cross_entropy'):
        dist_sync_on_step = True
        metrics = MetricCollection({
            'accuracy': Accuracy(dist_sync_on_step=dist_sync_on_step),
            'precision': Precision(dist_sync_on_step=dist_sync_on_step),
            'recall': Recall(dist_sync_on_step=dist_sync_on_step),
            'f1': F1Score(dist_sync_on_step=dist_sync_on_step)})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.checkpoint_metrics = metrics.clone(prefix='val_')


class ValStepMixin:
    def validation_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1]
        y_hat = self(*x)
        loss = self.loss_criterion(y_hat, y.to(torch.float))

        return {'loss': loss, 'preds': y_hat.detach(), 'target': y.detach()}


class TestStepMixin:
    def test_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1]
        y_hat = self(*x)
        loss = self.loss_criterion(y_hat, y.to(torch.float))

        return {'loss': loss, 'preds': y_hat.detach(), 'target': y.detach()}


class ValTestStepMixin(ValStepMixin, TestStepMixin):
    ...


class TrainStepEndMixin:
    def training_step_end(self, outputs):
        preds, targets = self.scale_output(outputs['preds']), self.scale_output(outputs['target'])
        self.train_metrics(preds, targets)
        self.log_dict(self.train_metrics, prog_bar=False,
                      logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class ValStepEndMixin:
    def validation_step_end(self, outputs):
        preds, targets = self.scale_output(outputs['preds']), self.scale_output(outputs['target'])
        self.val_metrics(preds, targets)
        self.checkpoint_metrics(preds, targets)

        self.log_dict(self.val_metrics, prog_bar=False,
                      logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        self.log_dict(self.checkpoint_metrics, prog_bar=False,
                      logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class TestStepEndMixin:
    def training_step_end(self, outputs):
        preds, targets = self.scale_output(outputs['preds']), self.scale_output(outputs['target'])
        self.train_metrics(preds, targets)
        self.log_dict(self.train_metrics, prog_bar=False,
                      logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class TrainValTestEndMixin(TrainStepEndMixin, ValStepEndMixin, TestStepEndMixin):
    ...


class LogWeightsMixin:
    def on_train_epoch_end(self, trainer=None, pl_module=None):
        for name, params in self.named_parameters():
            if "weight" in name:
                self.log_weights(name, params)

    def log_weights(self, name, params):
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.add_histogram(tag=f'weights/{name}',
                                                 values=params.data.view(-1).detach(),
                                                 global_step=self.trainer.global_step)


class LogGradientsMixin:
    def on_after_backward(self, trainer=None, pl_module=None, optimizer=None):  # todo: another hook?
        if self.trainer.global_step % self.log_grad_and_acts_every_n_steps == 0:
            for name, params in self.named_parameters():
                if "weight" in name or params.grad is not None:
                    self.log_gradients(name, params)

    def log_gradients(self, name, params):
        self.log(f'train_meta_info/variance/gradients/{name}',
                 torch.std(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
                 logger=True)
        self.log(f'train_meta_info/mean/gradients/{name}',
                 torch.mean(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
                 logger=True)
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.add_histogram(tag=f'grads/{name}',
                                                 values=params.grad.data.view(-1).detach(),
                                                 global_step=self.trainer.global_step)


class LogWeightsAndGradientsMixin(LogWeightsMixin, LogGradientsMixin):
    ...


class TensorBoardLoggerMixin(TrainValTestEndMixin, LogWeightsAndGradientsMixin):
    ...


class ClassifierMixin(ClassifierMetricsMixin, ValTestStepMixin):
    def __init__(self):
        super().__init__()
        self._init_metrics()


class ClassifierTensorBoardMixin(ClassifierMixin, TensorBoardLoggerMixin):
    def __init__(self):
        super().__init__()

