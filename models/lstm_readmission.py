"""
LSTM binary classifier for 30-day readmission from ICU vital-sign sequences.
Train: python -m models.lstm_readmission --data-dir data --epochs 30
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from models.readmission_dataset import (
    CHART_ITEM_ORDER,
    N_VITAL_FEATURES,
    get_readmission_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "data"
DEFAULT_PROCESSED = DEFAULT_DATA / "processed"
DEFAULT_RAW = DEFAULT_DATA / "raw"
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints"


class ReadmissionLSTM(nn.Module):
    """LSTM over vital sequences -> binary readmission."""

    def __init__(
        self,
        input_size: int = N_VITAL_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # Use last non-padded output (we pass full seq; for padding we could use lengths)
        out = out[:, -1, :]  # (B, hidden_size)
        logits = self.fc(out).squeeze(-1)
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append((logits > 0).long().cpu().numpy())
            labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = (preds == labels).mean()
    # Binary metrics
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--readmission-days", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()
    args.processed_dir = args.processed_dir or args.data_dir / "processed"
    args.raw_dir = args.raw_dir or args.data_dir / "raw"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading readmission dataset (sequences + labels)...")
    X, y, lengths, stay_ids = get_readmission_dataset(
        args.processed_dir,
        args.raw_dir,
        seq_len=args.seq_len,
        interval_minutes=args.interval_minutes,
        readmission_days=args.readmission_days,
    )
    print(f"  Stays: {len(y)}, readmission rate: {y.mean():.2%}, seq shape: {X.shape}")

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ReadmissionLSTM(
        input_size=N_VITAL_FEATURES,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    # Class weight for imbalanced readmission (optional)
    pos_weight = None
    if y.sum() > 0 and y.sum() < len(y):
        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}  loss={loss:.4f}  val acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                args.save_dir / "lstm_readmission_best.pt",
            )
    print(f"Best val F1: {best_f1:.3f}. Saved to {args.save_dir / 'lstm_readmission_best.pt'}")


if __name__ == "__main__":
    main()
