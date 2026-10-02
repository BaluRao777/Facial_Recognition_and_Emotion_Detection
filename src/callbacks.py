from tensorflow.keras.callbacks import Callback

from src.model import unfreeze_backbone


class FineTuneAtEpoch(Callback):
    """Unfreeze backbone layers at a given epoch."""

    def __init__(self, epoch: int, num_layers: int = 30, lr: float = 1e-4):
        super().__init__()
        self.epoch = epoch
        self.num_layers = num_layers
        self.lr = lr
        self.done = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self.done and epoch == self.epoch:
            unfreeze_backbone(self.model, self.num_layers, self.lr)
            self.done = True
            print(f"\n>>> Fine-tuning backbone (epoch {epoch}), lr={self.lr}\n")
