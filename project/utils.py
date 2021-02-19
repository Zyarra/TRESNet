import numpy as np
import torch

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)


def filter_params(module, train_bn=True):
    """Yield the trainable parameters of a given module.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : If true the BN layers will be trained
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            filter_params(module=child, train_bn=train_bn)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience: How long to wait after last time validation loss improved.
                            Default: 7
            verbose: If True, prints a message for each validation loss improvement.
                            Default: False
            delta: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path: Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func: trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, loss, val_loss, model, optimizer, epoch, cpload):
        score = -val_loss
        if cpload:
            self.val_loss_min = cpload
            self.best_score = cpload

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model, optimizer=optimizer, loss=loss, epoch=epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.trace_func('<----------------------NEXT EPOCH ---------------------->')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch, loss=loss)
            self.counter = 0

    def save_checkpoint(self, loss, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path}\n')
            self.trace_func('<----------------------NEXT EPOCH ---------------------->')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'epoch': epoch,
                    'val_loss': val_loss}, self.path)
        self.val_loss_min = val_loss
