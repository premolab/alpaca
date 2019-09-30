import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.custom_dataset import loader

class BaseFit():
    def __init__(self):
        pass
    
    def fit(
            self, train_set, val_set, epochs=10000, verbose=True,
            validation_step=100, patience=5, batch_size=500, dropout_rate=0):
        train_loader = loader(*train_set, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        current_patience = patience

        # Train the model
        for epoch in range(epochs):
            for points, labels in train_loader:
                # Move tensors to the configured device
                points = points.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self(points, train=True, dropout_rate=dropout_rate)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Print intermediate results and check patience
            if (epoch + 1) % validation_step == 0:
                val_loss = self.evaluate(val_set)
                if verbose:
                    self._print_status(epoch, epochs, loss.item(), val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        break
        self.val_loss = val_loss

    def evaluate(self, dataset, y_scaler=None):
        """ Return model losses for provided data loader """
        data_loader = loader(*dataset)
        with torch.no_grad():
            losses = []
            for points, labels in data_loader:
                points = points.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)
                outputs = self(points)
                if y_scaler is not None:
                    outputs = torch.Tensor(y_scaler.inverse_transform(outputs.cpu()))
                    labels = torch.Tensor(y_scaler.inverse_transform(labels.cpu()))
                losses.append(self.criterion(outputs, labels).item())

        return sum(losses)/len(losses)

    def _print_status(self, epoch, epochs, loss, val_loss):
        print('Epoch [{}/{}], Loss: {:.4f}, Validation loss: {:.4f}'
              .format(epoch + 1, epochs, loss, val_loss))

