import torch

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import pytorch_lightning as pl


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):

    def __init__(self, embed_size=512, num_heads=8,
                 forward_expansion=1, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size,
                                       num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        normed = self.norm1(attention + query)
        droped_out = self.dropout(normed)
        feeded = self.feed_forward(droped_out)
        normed_again = self.norm2(feeded + droped_out)

        return self.dropout(normed_again)


class Encoder(nn.Module):

    def __init__(self, input_size, embed_size=512, num_heads=8,
                 forward_expansion=1, dropout=0.0,
                 device=torch.device('cpu'), num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.device = device

        self.series_embedding = nn.Linear(input_size, embed_size)
        self.positional_embedding = nn.Embedding(input_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads,
                                 forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        batch_size, reg_vars, seq_length = X.shape
        positions = torch.arange(0, reg_vars*seq_length).expand(
            batch_size, seq_length*reg_vars).to(self.device)
        X = X.view(batch_size, 1, reg_vars*seq_length)
        series_embedded = self.series_embedding(X)
        pos_embedded = self.positional_embedding(positions)
        embed = series_embedded + pos_embedded
        droped_out = self.dropout(embed)
        for layer in self.layers:
            out = layer(droped_out, droped_out, droped_out, mask)

        return out


class DecoderBlock(nn.Module):

    def __init__(self, embed_size=512, num_heads=8,
                 forward_expansion=1, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size,
                                       num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = TransformerBlock(embed_size, num_heads,
                                            forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, value, key, target_mask=None, src_mask=None):
        attention = self.attention(X, X, X, target_mask)
        normed = self.norm(attention + X)
        query = self.dropout(normed)
        transformed = self.transformer(value, key, query, src_mask)
        return transformed


class Decoder(nn.Module):

    def __init__(self, embed_size=512, num_heads=8,
                 forward_expansion=1, dropout=0.0, horizons=12,
                 device=torch.device('cpu'), num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.horizons = horizons
        self.device = device

        self.series_embedding = nn.Linear(horizons, embed_size)
        self.positional_embedding = nn.Embedding(horizons, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, num_heads,
                             forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_out, target_mask=None, src_mask=None):
        batch_size, horizons = X.shape
        positions = torch.arange(0, horizons).expand(
            batch_size, horizons).to(self.device)
        X = X.view(batch_size, 1, horizons)
        series_embedded = self.series_embedding(X)
        pos_embedded = self.positional_embedding(positions)
        out = series_embedded + pos_embedded
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, encoder_out, encoder_out,
                        target_mask, src_mask)

        out = self.fc_out(out)
        out = out.view(batch_size, self.horizons)
        return out


class Transformer(pl.LightningModule):

    def __init__(self, input_size, horizons=12,
                 embed_size=512, num_layers=1,
                 forward_expansion=1, num_heads=8,
                 dropout=0.0, device=torch.device('cpu')):
        super().__init__()
        self.encoder = Encoder(input_size=input_size,
                               embed_size=embed_size,
                               num_heads=num_heads,
                               forward_expansion=forward_expansion,
                               dropout=dropout, device=device,
                               num_layers=num_layers)

        self.decoder = Decoder(embed_size=embed_size,
                               num_heads=num_heads,
                               forward_expansion=forward_expansion,
                               dropout=dropout, horizons=horizons,
                               device=device, num_layers=num_layers)

    def forward(self, src, tgt):
        encoded = self.encoder(src)
        decoded = self.decoder(tgt, encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X, y)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X, y)
        loss = F.mse_loss(y_hat, y)
        self.log('validation_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    device = torch.device("cpu")
    print(device)

    X = torch.ones((64, 3, 3)).float().to(device)
    print(X.size())
    y = torch.ones((64, 2)).float().to(device)
    print(y.size())

    model = Transformer(input_size=3*3, horizons=2, device=device).to(
        device
    )
    out = model(X, y)
    print(out.size())
