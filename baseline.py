import torch
import torchvision
import os
import glob
import random
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from skimage import io,transform

# 训练集train_s1,测试集test_s1,训练脚本baseline.py在同一个文件夹下
img_path = []
for i in range(0,415):
    img_path.append('test_s1/'+str(i)+'.jpg')

preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.TenCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


class TestDataset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = img_path
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        return img

    def __len__(self):
        return len(self.images)


class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=transform):
        self.transform = transform
        zero_dir = os.path.join(img_dir, "0")
        one_dir = os.path.join(img_dir,"1")
        two_dir = os.path.join(img_dir, "2")
        three_dir = os.path.join(img_dir, "3")
        four_dir = os.path.join(img_dir, "4")
        five_dir = os.path.join(img_dir,"5")
        six_dir = os.path.join(img_dir, "6")
        seven_dir = os.path.join(img_dir, "7")
        eight_dir = os.path.join(img_dir, "8")
        nine_dir = os.path.join(img_dir,"9")
        ten_dir = os.path.join(img_dir, "10")
        eleven_dir = os.path.join(img_dir, "11")
        twelve_dir = os.path.join(img_dir, "12")
        thirteen_dir = os.path.join(img_dir,"13")
        fourteen_dir = os.path.join(img_dir, "14")
        fifteen_dir = os.path.join(img_dir, "15")
        sixteen_dir = os.path.join(img_dir, "16")
        seventeen_dir = os.path.join(img_dir, "17")
        eighteen_dir = os.path.join(img_dir, "18")
        nineteen_dir = os.path.join(img_dir, "19")

        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(zero_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(one_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(two_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(three_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(four_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(five_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(six_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(seven_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(eight_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(nine_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(ten_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(eleven_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(twelve_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(thirteen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(fourteen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(fifteen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(sixteen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(seventeen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(eighteen_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(nineteen_dir, "*.jpg")))
        random.shuffle(imgsLib)
        self.imgsLib = imgsLib

    def __getitem__(self, index):
        img_paths = self.imgsLib[index]
        label = torch.tensor(int(img_paths.split('\\')[1]))
        img = Image.open(img_paths)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgsLib)

data_transform = transforms.Compose([
    transforms.Resize(512),  # resize, the smaller edge will be matched.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomResizedCrop(320, scale=(0.3, 1.0)),

    transforms.ToTensor(),  # convert a PIL image or ndarray to tensor.
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_data = TrainDataset(img_dir = "train_s1",transform=data_transform)

test_data = TestDataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.00006
num_epochs = 100
num_classes = 20

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 20)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.5, 0.9), eps=1e-8)

echo = int(len(train_loader) / 2)
for epoch in range(num_epochs):
    model.train()
    total_loss=0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        images.cuda()
        model.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % echo == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, total_loss / echo))


model.eval()
with torch.no_grad():
    nparam = sum(p.numel() for p in model.parameters())
    nparam = str(nparam)
    with open('predicted.txt', 'a+') as f:
        f.write(nparam + '\n')
        for i, images in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().detach().numpy().tolist()
            output = '%d:%d' % (i, predicted[0])
            if i == 414:
                f.write(output)
                f.close()
            else:
                f.write(output + '\n')
