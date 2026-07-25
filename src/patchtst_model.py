# PatchTST forecaster (patch + Transformer). One global model over all
# Store x Dept series, channel-independent (each series is its own univariate
# signal). The window is cut into patches, each patch becomes a token, a small
# Transformer runs over them, then we project to the horizon. Windows are
# z-normalised in and de-normalised out (RevIN-style) so one model can cover
# series of very different scales. Kept separate from the DLinear module.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-5


def _build_windows(train: pd.DataFrame, seq_len: int, pred_len: int,
                   window_stride: int = 1):
    """(X, y) windows across all series. X: (N, seq_len), y: (N, pred_len).

    window_stride > 1 keeps only every k-th window start, which cuts the training
    set (and CPU time) roughly k-fold without dropping any series.
    """
    xs, ys = [], []
    for _, g in train.groupby(["Store", "Dept"]):
        y = g.sort_values("Date")["y"].to_numpy(dtype=np.float32)
        if len(y) < seq_len + pred_len:
            continue
        last = len(y) - seq_len - pred_len + 1
        starts = list(range(0, last, window_stride))
        if starts and starts[-1] != last - 1:
            starts.append(last - 1)          # always keep the most recent window
        for i in starts:
            xs.append(y[i:i + seq_len])
            ys.append(y[i + seq_len:i + seq_len + pred_len])
    if not xs:
        raise ValueError("not enough history to build windows")
    return np.stack(xs), np.stack(ys)


class PatchTSTNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, patch_len: int = 8,
                 stride: int = 4, d_model: int = 64, n_heads: int = 4,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x):                      # x: (B, seq_len), already normalised
        # unfold into overlapping patches -> (B, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        h = self.embed(patches) + self.pos
        h = self.encoder(h)
        return self.head(h.reshape(h.size(0), -1))


def _train(net, X, y, epochs, batch_size, lr, device):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # per-window normalisation (RevIN-lite)
            mu = xb.mean(dim=1, keepdim=True)
            sd = xb.std(dim=1, keepdim=True) + EPS
            opt.zero_grad()
            pred = net((xb - mu) / sd) * sd + mu
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    net.eval()
    return net


class PatchTSTForecaster(BaseEstimator, RegressorMixin):
    """fit(raw_cols, y) stores history + trains; predict(raw_cols) forecasts."""

    def __init__(self, seq_len: int = 52, pred_len: int = 1, patch_len: int = 8,
                 stride: int = 4, d_model: int = 64, n_heads: int = 4, depth: int = 2,
                 dropout: float = 0.1, epochs: int = 10, batch_size: int = 1024,
                 lr: float = 1e-3, window_stride: int = 1, device: str | None = None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.window_stride = window_stride
        self.device = device

    def fit(self, X: pd.DataFrame, y):
        df = X[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["y"] = np.asarray(y, dtype=np.float32)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history_ = {}
        for (s, d), g in df.groupby(["Store", "Dept"]):
            self.history_[(int(s), int(d))] = \
                g.sort_values("Date")["y"].to_numpy(dtype=np.float32)
        self.global_mean_ = float(df["y"].mean())
        Xw, yw = _build_windows(df, self.seq_len, self.pred_len, self.window_stride)
        net = PatchTSTNet(self.seq_len, self.pred_len, self.patch_len, self.stride,
                          self.d_model, self.n_heads, self.depth, self.dropout)
        self.net_ = _train(net, Xw, yw, self.epochs, self.batch_size, self.lr,
                           self.device_)
        return self

    def _forecast_batch(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(windows.astype(np.float32)).to(self.device_)
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + EPS
        with torch.no_grad():
            out = self.net_((x - mu) / sd) * sd + mu
        return out[:, 0].cpu().numpy()          # one-step-ahead

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        X = X.reset_index(drop=True)
        # recursive but date-batched: all series that need a given date are
        # forecast together in one forward pass, then appended to history.
        hist = {k: v.copy() for k, v in self.history_.items()}
        out = np.full(len(X), self.global_mean_, dtype=np.float32)
        for date in np.sort(X["Date"].unique()):
            rows = X.index[X["Date"] == date]
            keys, idxs, windows = [], [], []
            for i in rows:
                key = (int(X.at[i, "Store"]), int(X.at[i, "Dept"]))
                past = hist.get(key)
                if past is None or len(past) == 0:
                    continue
                if len(past) < self.seq_len:
                    pad = np.full(self.seq_len - len(past), past[0], dtype=np.float32)
                    win = np.concatenate([pad, past])
                else:
                    win = past[-self.seq_len:]
                keys.append(key)
                idxs.append(i)
                windows.append(win)
            if not windows:
                continue
            preds = np.clip(self._forecast_batch(np.stack(windows)), 0, None)
            for key, i, p in zip(keys, idxs, preds):
                out[i] = p
                hist[key] = np.append(hist[key], np.float32(p))
        return out


def build_patchtst_pipeline(**kwargs) -> Pipeline:
    return Pipeline([("patchtst", PatchTSTForecaster(**kwargs))])
