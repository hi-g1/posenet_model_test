import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import AverageMeter

from PIL import Image
import cv2

import torch
import torchvision
from torchvision import transforms

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
# the validation transforms
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(root='./dataset/val', transform=val_transform)

image, lable = train_dataset[0]

batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = torchvision.models.squeezenet1_1()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

num_epochs = 10
loss_meter = AverageMeter()

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    for param_group in optimizer.param_groups:
        print('learing rage: ', param_group['lr'])

    # train
    model.train()
    print ('------------------- Train: Epoch [{}/{}] -------------------'.format(\
        epoch+1, num_epochs) )

    loss_meter.reset()

    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        loss_meter.update(loss.item(), X.size()[0] )

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(\
                epoch+1, num_epochs, i+1, total_step, loss_meter.avg) )

    print ('==> Train loss: {:.4f}'.format(loss_meter.avg) )

    # val
    model.eval()
    print ('------------------- Val.: Epoch [{}/{}] -------------------'.format(\
        epoch+1, num_epochs) )

    loss_meter.reset()

    for i, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # logging
        loss_meter.update(loss.item(), X.size()[0] )

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(\
                epoch+1, num_epochs, i+1, total_step, loss_meter.avg) )

    print ('==> Val. loss: {:.4f}'.format(loss_meter.avg) )

torch.save(model.state_dict(), 'trained.pt')

model.load_state_dict(torch.load('trained.pt'))
model.eval()

cm = np.zeros(shape=(2,2) )

for i, (X, y) in enumerate(val_loader):
    X = X.to(device)
    y = y.to(device)

    # forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # prediction
    preds = torch.argmax(outputs.data, 1)
    cm += confusion_matrix(y.cpu(), preds.cpu(),labels=[0,1])

acc = np.sum(np.diag(cm)/np.sum(cm) )
print ('==> Val. Accuracy: {:.4f}'.format(acc) )
print('Confusion matrix')
print(cm)