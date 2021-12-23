# flowers_torch.py - Fang Xi Lin 2021

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import argparse
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    """
    This class returns a convolution neural network PyTorch model,
    with num_features feature maps in the first convolutional layer, 2 *
    num_features in the second convolutional layer, 3 * num_features in the third
    convolutional layer, a dropout layer with rate drate and hidden_size
    neurons in the fully-connected layer.

    Inputs:
        num_features: int, the number of feature maps in the convolution layer.

        hidden_size: int, the number of nodes in the fully-connected layer.

        drate: float, dropout rate in the fully-connected layer.

        output_size: int, the number of nodes in the output layer,
            default = 17.

    Output: the constructed Keras model.

    """
    def __init__(self, num_features, hidden_size, drate, output_size = 17):
        super(CNN, self).__init__()

        # Initialize
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.drate = drate
        self.output_size = output_size

        self.feat_arr = [3, self.num_features, 2*self.num_features, 3*self.num_features]
        # Conv layers with max pooling
        conv_blocks = [self.__conv_block(in_f, out_f, kernel_size=2) for in_f, out_f in zip(self.feat_arr, self.feat_arr[1:])]
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Flattening layer
        # self.flatten  = nn.Flatten()

        # Dense layer with dropout
        self.fc_block = nn.Sequential(
            nn.Linear(5*5*self.feat_arr[-1], self.hidden_size),
            nn.Dropout(self.drate),
            nn.ReLU()
        )

        # Final classification layer
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def __conv_block(self,in_f, out_f, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, *args, **kwargs),
            nn.MaxPool2d((2, 2), stride = (2, 2)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_blocks(x)

        x = x.view(x.size(0), -1) # flatten

        x = self.fc_block(x)

        x = self.fc2(x)

        return x

    def evaluate(self, dataloader):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)

                outputs = self(x)

                prob, y_predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (y_predicted == y).sum().item()
        accuracy = correct/total

        return accuracy

class FlowersDataset(Dataset):
    """
    This class loads image and label files, then split them into a training
    (80 %) and test (20%) set.

    Inputs:
        img_path: str, relative path to image file.
        label_path: str, relative path to targets file.
        train: bool, which set of data to load.
	transform: pytorch transform, transforms to perform
    """
    def __init__(self, img_path, label_path, train=True, transform=None):
        #
        self.transform = transform

        self.x = np.load(img_path)
        self.x = self.x.astype(np.uint8)
        # self.x = np.moveaxis(self.x, -1, 1)

        self.y = np.load(label_path) - 1
        self.y = self.y.astype(np.uint8)

        if train:
            self.x, self.y = self.x[self.x.shape[0]//5:], self.y[self.y.shape[0]//5:]
        else:
            self.x, self.y = self.x[:self.x.shape[0]//5], self.y[:self.y.shape[0]//5]
        # print(self.x.shape)
        self.N = self.x.shape[0]

    def __getitem__(self, index):
        x_single = self.x[index]
        y_single = self.y[index]
        if self.transform is not None:
            x_single = self.transform(x_single)
        return (x_single, y_single)

    def __len__(self):
        return self.N

    def show_images(self, title=''):
        """
        This functon plots the first 32 images with given labels of the dataset.

        Inputs:
            images: uint8 Numpy array, image data with shape
                (num_samples, 50, 50, 3).
            labels: uint8 Numpy array, one-hot encoded labels with shape
                (num_samples, 17).
            title: str, title of the plot.

        Returns:
            Nothing returned.
        """

        plt.figure(figsize=(12,12))
        plt.suptitle(title, fontsize=30)
        for i in range(32):
            image, label = self[i]
            image = image / 2 + 0.5 #unnormalize
            npimg = np.transpose(image.numpy(), (1,2,0))
            plt.subplot(4,8,i+1)
            plt.imshow(npimg)
            plt.title('{}'.format(label))
            plt.xticks([])
            plt.yticks([])
        plt.show()

        return

def get_transforms():
    """
    Returns the training pytorch transformations and testing pytorch transformations.
    """
# Rotating and flipping flowers still look like flowers. Shift and zoom ranges is good for the small image size. Shear is a good approximation of perspective. 'reflect' fill avoids overfitting on the edge effects of other fill modes, and brightness range is reasonable for the natural lighting in the dataset.
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ColorJitter(brightness=.2,contrast=0.,saturation=0.),
         transforms.RandomAffine(degrees=180, translate=(0.1,0.1), scale=(1.0,1.05), shear=20),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
         ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
         ]
    )

    return train_transform, test_transform


def main(image_path = '50x50flowers.images.npy', label_path = '50x50flowers.targets.npy',  show_images=False, show_history=False, epochs=300, verbose=False):

    # get data info
    train_transform, test_transform = get_transforms()
    batch_size = 32

    # load data
    x_train = FlowersDataset(image_path, label_path, train=True, transform=train_transform)
    train_dataloader = torch.utils.data.DataLoader(x_train, batch_size = batch_size, shuffle=True, num_workers=2)

    x_test = FlowersDataset(image_path, label_path, train=False, transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(x_test, batch_size = batch_size, shuffle=True, num_workers=2)

    if show_images:
        x_train.show_images(title='Augmented data')

    #total number of batches
    num_batches = len(x_train)//batch_size

    #initialize CNN, loss function and optimizer
    cnn = CNN(num_features = 32, hidden_size = 128, drate=0.35, output_size = 17).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-8)

    # training mode
    cnn.train()

    #training loop
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, start=0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero out gradient buffer
            optimizer.zero_grad()

            # evaluate network (forward pass)
            outputs = cnn(inputs)
            # compute loss
            loss = criterion(outputs, labels)
            # backpropagate gradients
            loss.backward()
            # update weights and biases
            optimizer.step()

            running_loss += loss.item()
            if verbose:
                print('Epoch {}/{}, Batch {}/{}, Loss: {}'.format(epoch+1, epochs, i+1, num_batches, loss.item()))
        print('Epoch {}/{}, Epoch Average Loss: {}'.format(epoch+1, epochs, running_loss/num_batches))

    # evaluation mode
    cnn.eval()
    train_score = cnn.evaluate(train_dataloader)
    test_score = cnn.evaluate(test_dataloader)

    print('Training score is {}'.format(train_score))
    print('Test score is {}'.format(test_score))
    print('Accuracy difference is {} %'.format(100 * (train_score - test_score) ) )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural network for downsized flowers dataset.')
    parser.add_argument('--image-path', help='Relative path to images file')
    parser.add_argument('--label-path', help='Relative path to labels file')
    parser.add_argument('--show-images', action="store_true", help='Display sample and augmented data')
    parser.add_argument('--show-history', action="store_true", help='Display loss and accuracy history after training')
    parser.add_argument('-e', '--epochs', type=int, help='Number to epochs to train')
    parser.add_argument('-v', '--verbose', action="store_true", help='Verbosity')
    args = parser.parse_args()

    main(image_path=args.image_path, label_path=args.label_path,show_images=args.show_images, show_history=args.show_history, epochs=args.epochs, verbose=args.verbose)
