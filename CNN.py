import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn_model import ConvNet
import numpy as np

class CNN:

    def __init__(self,
                 epochs=5,
                 batch_size=50,
                 num_features = 250,
                 lr=0.001,
                 channels_in=1,
                 channels_out=50,
                 kernel_size_conv=5,
                 dilation_conv=1,
                 padding_conv=0,
                 stride_conv=1,
                 kernel_size_mxp=5,
                 dilation_mxp=1,
                 padding_mxp=0,
                 stride_mxp=2):

        self.val_acc_list = np.zeros(epochs)
        self.tr_losses = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)
        self.epochs = epochs
        self.num_features = num_features
        self.channels_in = channels_in
        self.batch_size= batch_size
        self.model = ConvNet(
            channels_in=self.channels_in,
            channels_out=channels_out,
            num_features=num_features,
            kernel_size_conv=kernel_size_conv,
            dilation_conv=dilation_conv,
            padding_conv=padding_conv,
            stride_conv=stride_conv,
            kernel_size_mxp=kernel_size_mxp,
            dilation_mxp=dilation_mxp,
            padding_mxp=padding_mxp,
            stride_mxp=stride_mxp)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def fit(self, tr_data, val_data):


        for epoch in range(self.epochs):
            # batch the train data and shuffle them
            train_loader = DataLoader(dataset=tr_data,
                                      batch_size=self.batch_size,
                                      shuffle=True)

            # batch validation data
            val_loader = DataLoader(dataset=val_data,
                                    batch_size=self.batch_size,
                                    shuffle=False)

            # train model
            self.model.train()
            for x, y in train_loader:

                # reshape records from 2D to 3D
                x = x.reshape(x.shape[0], self.channels_in, self.num_features)
                # Run the forward pass
                outputs = self.model(x)

                # calculate losses
                loss = self.criterion(outputs.float(), y.float())
                self.tr_losses[epoch] += loss.item()

                # Backprop and perform Adam optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.tr_losses[epoch] = self.tr_losses[epoch] / len(train_loader)
            print(f'Epoch [{epoch + 1}/{self.epochs}]')
            print(f'\tTr. Loss: {self.tr_losses[epoch]:.4f}')

            # prediction for validation data
            self.model.eval()
            with torch.no_grad():

                for x, y in val_loader:
                    # reshape records from 2D to 3D
                    x = x.reshape(x.shape[0], self.channels_in, self.num_features)
                    # Run the forward pass
                    outputs = self.model(x)

                    # calculate losses
                    loss = self.criterion(outputs.float(), y.float())
                    self.val_losses[epoch] += loss.item()
                    _, predicted = torch.max(outputs.data, 1)

                    # calculate accuracy
                    self.val_acc_list[epoch] += (predicted == np.argmax(y, axis=1)).sum().item()

            self.val_acc_list[epoch] = self.val_acc_list[epoch] * 100 / len(val_data)
            self.val_losses[epoch] = self.val_losses[epoch] / len(val_loader)
            print(f'\tVal. Loss: {self.val_losses[epoch]:.4f}')
            print(f'\tVal. Accuracy: {self.val_acc_list[epoch]}')
        return self

    def predict(self, test_data):

        # batch test data
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        predictions = []
        acc = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                # reshape records from 2D to 3D
                x = x.reshape(x.shape[0], self.channels_in, self.num_features)
                # Run the forward pass
                outputs = self.model(x)

                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())

                # calculate accuracy
                acc += (predicted == np.argmax(y, axis=1)).sum().item()

        acc = acc * 100 / len(test_data)
        return predictions, acc
