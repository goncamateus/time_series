import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


class MLP(pl.LightningModule):
    def __init__(self, input_size, horizons=12):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, horizons),
        )
        self.init_weights()

    def init_weights(self):
        def inside(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(inside)

    def forward(self, src):
        batch_size, reg_vars, seq_length = src.shape
        X = src.view(batch_size, 1, reg_vars * seq_length).float()
        return self.layers(X).squeeze()

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


class MLPLucas(pl.LightningModule):
    def __init__(self, window_size, n_comps, horizons=12):
        super().__init__()
        base_layer = nn.Sequential(
            nn.Linear(window_size, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )

        self.comp_layers = nn.ModuleList([base_layer] * n_comps)
        self.fc_out = nn.Sequential(
            nn.Linear(n_comps, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, horizons),
        )

    def forward(self, src):
        batch_size, reg_vars, seq_length = src.shape
        outputs = torch.zeros((batch_size, 1, seq_length)).to(self.device)
        for i, layer in enumerate(self.comp_layers):
            outputs[:, :, i] = layer(src[:, :, i])
        return self.fc_out(outputs).squeeze()

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
