_exp_name = "sample"
# Import necessary packages.
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
from collections import defaultdict

# This is for the progress bar.
from tqdm.auto import tqdm
import random
import argparse

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--n_epochs', type = int, default = 4)
parser.add_argument('--patience', type = int, default = 300)
parser.add_argument('--learning_rate', type = float, default = 0.0003)
parser.add_argument('--input_size', type = int, default = 128)
parser.add_argument('--ckpt', type = str, default = "")
parser.add_argument('--p2', action='store_true')

config = parser.parse_args()
print(config)
batch_size    = config.batch_size
n_epochs      = config.n_epochs
patience      = config.patience
learning_rate = config.learning_rate
input_size    = config.input_size
ckpt_path     = config.ckpt

myseed = 5278  # set a random seed for reproducibility
device = "cuda:0"
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
IS_TRAIN = True
NUM_TTA_SAMPLE = 200
NUM_VALIDE_TTA = 3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
# 
# Please refer to PyTorch official website for details about different transforms.

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Shape shift
    transforms.Resize((512, 512)),
    transforms.RandomChoice([transforms.RandomCrop((450, 450)), transforms.Resize((input_size, input_size))]),
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomChoice([transforms.RandomRotation(degrees=90), transforms.Resize((input_size, input_size))]),
    
    # Color
    transforms.ColorJitter(),
    transforms.RandomGrayscale(0.1),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomEqualize(0.5),
    transforms.RandomChoice([transforms.RandomAdjustSharpness(3, 1.0), transforms.GaussianBlur(31)]),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

# There are tons of model in Pytoch lib https://pytorch.org/vision/stable/models.html

class My_Classifier(nn.Module):
    def __init__(self):
        super(My_Classifier, self).__init__()
        # VGG16
        # self.model = models.vgg16_bn(pretrained=False)
        # self.model.classifier[6] = nn.Linear(4096, 11)

        # Efficient Net
        self.model = models.efficientnet_b4(pretrained=False)
        self.model.classifier[1] = nn.Linear(1792, 11)
        
        # Resnet101
        # self.model = models.resnet101(pretrained=False)
        # self.model.fc = nn.Linear(2048, 11)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256* 32* 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)
        x1 = self.relu(x1)

        x2 = self.cnn_layer2(x1)
        x2 += x1
        x2 = self.relu(x2)

        x3 = self.cnn_layer3(x2)
        x3 = self.relu(x3)

        x4 = self.cnn_layer4(x3)
        x4 += x3
        x4 = self.relu(x4)

        x5 = self.cnn_layer5(x4)
        x5 = self.relu(x5)

        x6 = self.cnn_layer6(x5)
        x6 += x5
        x6 = self.relu(x6)

        xout = x6.flatten(1)
        xout = self.fc_layer(xout)
        return xout

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        # Skip connection Reference
        # https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L56-L72
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# batch_size = 64
_dataset_dir = "./food11"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=train_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# The number of training epochs and patience.
# n_epochs = 4
# patience = 300 # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
if config.p2:
    model = Residual_Network().to(device)
else:
    model = My_Classifier().to(device)
# load checkpoint 
if ckpt_path != "":
    print(f"Loading Checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):
    if not IS_TRAIN:
        break
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc  = sum(train_accs) / len(train_accs)

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"Epoch({epoch + 1:03d}/{n_epochs:03d}) train_loss {train_loss:.5f} | train_acc {train_acc:.5f} | valid_loss {valid_loss:.5f} | valid_acc {valid_acc:.5f}")

    # Print the information.
    # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

# test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# # Testing and generate prediction CSV
if config.p2:
    model_best = Residual_Network().to(device)
else:
    model_best = My_Classifier().to(device)

model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []

file_names = sorted(os.listdir(os.path.join(_dataset_dir, "test")))

with torch.no_grad():
    final_prediction = []
    # Test Time Argumentation
    for fname in file_names:
        print(f"Testing {os.path.join(_dataset_dir, 'test', fname)}")
        im = Image.open(os.path.join(_dataset_dir, "test", fname))
        
        vote_count = defaultdict(int)
        for i in range(NUM_TTA_SAMPLE):
            im_tfm = train_tfm(im)
            im_tfm = im_tfm[None, :] # [1, 3, 224, 224]
            test_pred  = model_best(im_tfm.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            vote_count[test_label[0]] += 1
        
        # Calcualte max count in vote_count
        max_count = 0
        max_key = None
        for i in vote_count:
            if vote_count[i] > max_count: 
                max_count = vote_count[i]
                max_key = i
        
        print(vote_count)
        final_prediction.append(max_key)
        # prediction += test_label.squeeze().tolist()

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(file_names)+1)]
df["Category"] = final_prediction
df.to_csv("submission.csv",index = False)
print("Save testing result to submission.csv")