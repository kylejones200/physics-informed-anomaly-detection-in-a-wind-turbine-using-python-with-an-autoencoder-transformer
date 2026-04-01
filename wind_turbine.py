# wind_turbine.py

import argparse
import glob
import numpy as np
import os
import time
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(n_features, dropout=0):
    return nn.Sequential(
        nn.Conv2d(n_features, 32, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Conv2d(32, 64, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Conv2d(64, 128, kernel_size=2, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=2, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, n_features, kernel_size=2, padding=1),
    )

def load_data(data_dir):
    input_files = glob.glob(os.path.join(data_dir, '*.npy'))
    data = [np.load(i) for i in input_files]
    return np.vstack(data)

def train_epoch(optimizer, criterion, model, train_loader, test_loader):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    test_loss = 0.0
    model.eval()
    for x, y in test_loader:
        output = model(x)
        loss = criterion(output, y)
        test_loss += loss.item()
    return train_loss, test_loss

def train(args):
    best_loss = float('inf')
    X = load_data(args.train)
    criterion = nn.MSELoss()
    kf = TimeSeriesSplit(n_splits=args.k_fold_splits)

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        if args.k_index_only >= 0 and args.k_index_only != i:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        X_train = torch.from_numpy(X_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)

        train_ds = torch.utils.data.TensorDataset(X_train, X_train)
        test_ds = torch.utils.data.TensorDataset(X_test, X_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

        model = create_model(args.num_features, args.dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            t0 = time.time()
            train_loss, test_loss = train_epoch(optimizer, criterion, model, train_loader, test_loader)
            logging.info(f"Fold={i} Epoch={epoch} TrainLoss={train_loss:.4f} TestLoss={test_loss:.4f} Time={time.time()-t0:.2f}s")
            if test_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(args.output_data_dir, 'model_state.pth'))
                best_loss = test_loss

    model = create_model(args.num_features, args.dropout_rate)
    model.load_state_dict(torch.load(os.path.join(args.output_data_dir, 'model_state.pth')))
    torch.save(model, os.path.join(args.model_dir, 'model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold_splits', type=int, default=6)
    parser.add_argument('--k_index_only', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_features', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    args = parser.parse_args()
    train(args)