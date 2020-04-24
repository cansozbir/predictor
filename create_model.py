import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  
        self.conv2 = nn.Conv2d(8, 16, 3) 
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


def get_num_correrct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()


transform = transforms.Compose([transforms.ToTensor()])
# ,transforms.Normalize(
#     (0.5,), (0.5,))])  # mean, std_deriv


train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transform
)

test_val = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transform
)

test_set, val_set = torch.utils.data.random_split(test_val, [5000, 5000])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=256, shuffle=True, num_workers=4)

net = Net()

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


def train(epoch_num):

    net.train()

    for epoch in range(epoch_num):
        total_loss = 0
        total_correct = 0
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += get_num_correrct(outputs, labels)
        print("[train] epoch:", epoch,
              "total_correct:", total_correct,
              "total_loss:", total_loss,
              "accuracy:", total_correct/len(train_set)
              )
        validate(epoch)


def validate(epoch):
    with torch.no_grad():
        net.eval()
        total_loss_val = 0
        total_correct_val = 0
        for batch in val_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss_val += loss.item()
            total_correct_val += get_num_correrct(outputs, labels)
    print("[val] epoch:", epoch,
          "total_correct:", total_correct_val,
          "total_loss:", total_loss_val,
          "accuracy:", total_correct_val/len(val_set)
          )


def test():
    with torch.no_grad():
        net.eval()
        total_loss = 0
        total_correct = 0
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += get_num_correrct(outputs, labels)

        print("[test] total_correct:", total_correct,
              "total_loss:", total_loss,
              "accuracy:", total_correct/len(test_set)
              )



train(epoch_num=5)
test()
example = torch.rand(1, 1, 28, 28)
trace = torch.jit.trace(net, example)
trace.save("mymodel.pt")
