from __future__ import print_function
import argparse
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from model import Model
# from utils import *
import torch

def read_data(datasets_dir="./data", path='data.pkl.gzip', frac = 0.):
    
    print("... read data")
    data_file = os.path.join(datasets_dir, path)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    X_train, y_train = X, y
    return X_train, y_train


def preprocessing(X_train, y_train):
    X_train = np.dot(X_train[...,:3], [0.2989, 0.5870, 0.1140]).astype('float32') 
    return X_train, y_train


def train_model(X_train, y_train, path, num_epochs=50, learning_rate=1e-4, batch_size=64):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("... train model")
    model = Model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_train_torch = torch.from_numpy(X_train[:,np.newaxis,...]).to(device)
    y_train_torch = torch.from_numpy(y_train).to(device)
    
    # Create TensorDataset and DataLoader to randomize batches
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for t in range(num_epochs):
        print(f"[EPOCH]: {t}", end='\r')
        
        # Iterate over the randomized batches
        for batch_X, batch_Y in train_loader:
            # Move the data to the appropriate device (GPU/CPU)
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Forward pass
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.save(path)


