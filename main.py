import matplotlib.pyplot as plt
import seaborn
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.dataset import MTSFDataset
from src.model import Transformer

seaborn.set()


def main():

    train_dataset = MTSFDataset(window=5, horizon=3, data_name='G7')
    test_dataset = MTSFDataset(window=5, horizon=3,
                               data_name='G7', set_type='test')
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=8)

    trainer = Trainer(gpus=1, max_epochs=5)
    input_example = next(iter(train_loader))[0]
    input_size = input_example.shape[1]*input_example.shape[2]

    device = torch.device('cuda')
    model = Transformer(input_size=input_size,
                        horizons=3, device=device,
                        forward_expansion=2, num_layers=1).to(device)

    trainer.fit(model, train_dataloader=train_loader)
    trainer.test(model, test_dataloaders=test_loader)

    for X, y in test_loader:
        y_hat = model(X.to(device), X.to(device)).detach().cpu().numpy()
        plt.plot(y.numpy())
        plt.plot(y_hat)
        plt.show()


if __name__ == '__main__':
    main()
