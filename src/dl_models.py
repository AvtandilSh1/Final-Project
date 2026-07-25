"""Global DLinear / N-BEATS forecasters for Store×Dept weekly series.

Fits on a lookback window of past Weekly_Sales and predicts the next step(s).
The sklearn-style estimators keep train history so ``predict`` works on raw
Kaggle rows ``[Store, Dept, Date, IsHoliday]``.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset


def _series_key(store, dept):
    return (int(store), int(dept))


def build_windows(train: pd.DataFrame, seq_len: int, pred_len: int = 1):
    """Build (X, y) windows across all Store×Dept series. X shape (N, seq_len)."""
    xs, ys = [], []
    for (_, _), g in train.groupby(["Store", "Dept"]):
        y = g.sort_values("Date")["Weekly_Sales"].to_numpy(dtype=np.float32)
        if len(y) < seq_len + pred_len:
            continue
        for i in range(len(y) - seq_len - pred_len + 1):
            xs.append(y[i: i + seq_len])
            ys.append(y[i + seq_len: i + seq_len + pred_len])
    if not xs:
        raise ValueError("Not enough history to build windows")
    return np.stack(xs), np.stack(ys)


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L)
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.unsqueeze(1)).squeeze(1)


class DLinearNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        trend = self.decomp(x)
        seasonal = x - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


class NBeatsBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, width: int = 128, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = seq_len
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(width, seq_len)
        self.theta_f = nn.Linear(width, pred_len)

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBeatsNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, width: int = 128,
                 n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(seq_len, pred_len, width, n_layers) for _ in range(n_blocks)]
        )

    def forward(self, x):
        residual = x
        forecast = 0.0
        for block in self.blocks:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f
        return forecast


def _train_net(net, X, y, epochs: int, batch_size: int, lr: float, device: str):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return net


class GlobalSeriesRegressor(BaseEstimator, RegressorMixin):
    """Sklearn wrapper: fit on raw train frame columns + y, predict raw rows."""

    def __init__(self, architecture: str = "dlinear", seq_len: int = 52,
                 pred_len: int = 1, epochs: int = 8, batch_size: int = 2048,
                 lr: float = 1e-3, kernel_size: int = 25, width: int = 128,
                 n_blocks: int = 3, n_layers: int = 4, device: str | None = None):
        self.architecture = architecture
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kernel_size = kernel_size
        self.width = width
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.device = device

    def _make_net(self):
        if self.architecture == "dlinear":
            return DLinearNet(self.seq_len, self.pred_len, self.kernel_size)
        if self.architecture == "nbeats":
            return NBeatsNet(self.seq_len, self.pred_len, self.width,
                             self.n_blocks, self.n_layers)
        raise ValueError(self.architecture)

    def fit(self, X, y):
        # X is DataFrame with Store, Dept, Date (from a thin preprocessor) OR we
        # expect callers to pass a frame via fit_frame.
        raise RuntimeError("Use fit_frame(train_df) for this estimator")

    def fit_frame(self, train: pd.DataFrame):
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        self.history_ = {}
        for (s, d), g in train.groupby(["Store", "Dept"]):
            g = g.sort_values("Date")
            self.history_[_series_key(s, d)] = {
                "dates": g["Date"].to_numpy(),
                "y": g["Weekly_Sales"].to_numpy(dtype=np.float32),
            }
        self.global_mean_ = float(train["Weekly_Sales"].mean())
        Xw, yw = build_windows(train, self.seq_len, self.pred_len)
        net = self._make_net()
        self.net_ = _train_net(net, Xw, yw, self.epochs, self.batch_size, self.lr, device)
        self.net_.eval()
        return self

    def _window_for(self, store, dept, date) -> np.ndarray | None:
        hist = self.history_.get(_series_key(store, dept))
        if hist is None:
            return None
        dates, y = hist["dates"], hist["y"]
        # use all points strictly before `date`
        mask = dates < np.datetime64(pd.Timestamp(date))
        y_past = y[mask]
        if len(y_past) < self.seq_len:
            if len(y_past) == 0:
                return None
            pad = np.full(self.seq_len - len(y_past), y_past[0], dtype=np.float32)
            y_past = np.concatenate([pad, y_past])
        return y_past[-self.seq_len:].astype(np.float32)

    def predict_frame(self, test: pd.DataFrame) -> np.ndarray:
        # recursive multi-step: after predicting a date, append to history so
        # later weeks of the same series can use the forecast.
        hist_y = {k: v["y"].copy() for k, v in self.history_.items()}
        hist_d = {k: list(v["dates"]) for k, v in self.history_.items()}
        out = np.zeros(len(test), dtype=np.float32)
        device = self.device_
        order = test.sort_values(["Date", "Store", "Dept"]).index
        with torch.no_grad():
            for idx in order:
                row = test.loc[idx]
                key = _series_key(row["Store"], row["Dept"])
                date = pd.Timestamp(row["Date"])
                y_past = hist_y.get(key)
                if y_past is None or len(y_past) == 0:
                    pred = self.global_mean_
                else:
                    if len(y_past) < self.seq_len:
                        pad = np.full(self.seq_len - len(y_past), y_past[0], dtype=np.float32)
                        window = np.concatenate([pad, y_past])[-self.seq_len:]
                    else:
                        window = y_past[-self.seq_len:]
                    xt = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
                    pred = float(self.net_(xt).cpu().numpy().ravel()[0])
                pred = max(pred, 0.0)
                out[idx] = pred
                if key not in hist_y:
                    hist_y[key] = np.array([pred], dtype=np.float32)
                    hist_d[key] = [date]
                else:
                    hist_y[key] = np.append(hist_y[key], np.float32(pred))
                    hist_d[key].append(date)
        return out


class RawDLPipeline(BaseEstimator, RegressorMixin):
    """Minimal Pipeline stand-in: fit/predict on RAW_COLS + stores history in train."""

    def __init__(self, model: GlobalSeriesRegressor):
        self.model = model

    def fit(self, X, y=None):
        # X must be a DataFrame with Store, Dept, Date and Weekly_Sales column,
        # OR (X raw cols, y). Prefer passing full train via fit_full.
        raise RuntimeError("Call fit_full(train_df)")

    def fit_full(self, train: pd.DataFrame):
        self.model.fit_frame(train)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict_frame(X)
