import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset


class CNN(nn.Module):
    def __init__(self, num_features, hidden_size, drate, output_size = 17):
        super(CNN, self).__init__()

        # Initialize
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.drate = drate
        self.output_size = output_size

        # First conv layer with max pooling
        self.conv1    = nn.Conv2d(3, self.num_features, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Second conv layer
        self.conv2    = nn.Conv2d(self.num_features, 2*self.num_features, kernel_size = (3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Third conv layer
        self.conv3    = nn.Conv2d(2*self.num_features, 3*self.num_features, kernel_size = (3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Flattening layer
        # self.flatten  = nn.Flatten()

        # Dense layer with dropout
        self.fc1 = nn.Linear(4*4*3*self.num_features, self.hidden_size)
        self.dropout = nn.Dropout(self.drate)

        # Final classification layer
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1) # flatten

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.softmax(x, -1)

        return x

class FlowersDataset(Dataset):
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

cnn = CNN(32,256,0.35,17)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

x_train = FlowersDataset(drive+'50x50flowers.images.npy',drive+'50x50flowers.targets.npy', train=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(x_train, batch_size = 32, shuffle=True, num_workers=2)

x_test = FlowersDataset(drive+'50x50flowers.images.npy',drive+'50x50flowers.targets.npy', train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(x_test, batch_size = 32, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-7)


for epoch in range(10):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
