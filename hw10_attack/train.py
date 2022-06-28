# **Homework 10 - Adversarial Attack**
# Slides: https://reurl.cc/7DDxnD
# ## Enviroment & Download
# We make use of [pytorchcv](https://pypi.org/project/pytorchcv/) to obtain CIFAR-10 pretrained model, so we need to set up the enviroment first. We also need to download the data (200 images) which we want to attack.
import torch
import torch.nn as nn
import os
import glob
import shutil
import numpy as np
from PIL import Image
import random
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
device = "cuda:0"
batch_size = 8
random.seed(5278)
DIM_P = 0.0 # DIM's probabiliy 
# Global Settings 
# **[NOTE]**: Don't change the settings here, or your generated image might not meet the constraint.
# * $\epsilon$ is fixed to be 8. But on **Data section**, we will first apply transforms on raw pixel value (0-255 scale) **by ToTensor (to 0-1 scale)** and then **Normalize (subtract mean divide std)**. $\epsilon$ should be set to $\frac{8}{255 * std}$ during attack.
# 
# * Explaination (optional)
#     * Denote the first pixel of original image as $p$, and the first pixel of adversarial image as $a$.
#     * The $\epsilon$ constraints tell us $\left| p-a \right| <= 8$.
#     * ToTensor() can be seen as a function where $T(x) = x/255$.
#     * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
#     * After applying ToTensor() and Normalize() on $p$ and $a$, the constraint becomes $\left| N(T(p))-N(T(a)) \right| = \left| \frac{\frac{p}{255}-mean}{std}-\frac{\frac{a}{255}-mean}{std} \right| = \frac{1}{255 * std} \left| p-a \right| <= \frac{8}{255 * std}.$
#     * So, we should set $\epsilon$ to be $\frac{8}{255 * std}$ after ToTensor() and Normalize().

# the mean and std are the calculated statistics from cifar_10 dataset
cifar_10_mean = (0.491, 0.482, 0.447) # mean for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201) # std for the three channels of cifar_10 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)
epsilon = 8/255/std
root = './data' # directory for storing benign images
# benign images: images which do not contain adversarial perturbations
# adversarial images: images which include adversarial perturbations
# Construct dataset and dataloader from root directory. Note that we store the filename of each image for future usage.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')

# to evaluate the performance of model on benign images
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y) # calculate loss
    loss.backward() # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv

# alpha and num_iter can be decided by yourself
alpha = 0.8/255/std
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0):
    x_adv = x
    
    if random.random() > DIM_P:
        new_dim = (random.randint(28,31),  random.randint(28,31))
        x_adv = F.resize(x_adv, size=new_dim, interpolation=InterpolationMode.NEAREST)
        # print(f"new_dim = {x_adv.shape}")
        pad_top = (32 - new_dim[0])//2
        pad_bot =  32 - new_dim[0] - pad_top
        pad_left  = (32 - new_dim[1])//2
        pad_right =  32 - new_dim[1] - pad_left
        x_adv = F.pad(x_adv, padding=[pad_left, pad_top, pad_right, pad_bot], fill=0, padding_mode="constant")
        # print(f"after pad = {x_adv.shape}")
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # Momentum calculation
        grad    = x_adv.grad.detach()
        grad_n1 = torch.linalg.norm(x_adv.grad.detach(), ord=1, dim=(2,3))
        b, c = grad_n1.shape
        for batch in range(b):
            for channel in range(c):
                grad[batch, channel] = grad[batch, channel] / grad_n1[batch, channel]
        momentum = momentum*decay + grad
        x_adv = x_adv + alpha * momentum.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

# ## Utils -- Attack
# * Recall
#   * ToTensor() can be seen as a function where $T(x) = x/255$.
#   * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
# 
# * Inverse function
#   * Inverse Normalize() can be seen as a function where $N^{-1}(x) = x*std+mean$ where $mean$ and $std$ are constants.
#   * Inverse ToTensor() can be seen as a function where $T^{-1}(x) = x*255$.
# 
# * Special Noted
#   * ToTensor() will also convert the image from shape (height, width, channel) to shape (channel, height, width), so we also need to transpose the shape back to original shape.
#   * Since our dataloader samples a batch of data, what we need here is to transpose **(batch_size, channel, height, width)** back to **(batch_size, height, width, channel)** using np.transpose.

# perform adversarial attack and generate adversarial examples
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn) # obtain adversarial examples
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))

# Model / Loss Function
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
# Single model 
# model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
# Ensemble model
class Ensemble(nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.model_list = model_list
        [m.eval() for m in model_list] # Switch all model to eval() mode

    def forward(self, x):
        output = torch.zeros([8, 10], dtype=torch.float64).to(device)
        for model in model_list:
            output += model(x) * (1/len(self.model_list))
        return output

model_list = [ptcv_get_model('resnet56_cifar10', pretrained=True).to(device),
              ptcv_get_model('preresnet20_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet110_cifar10', pretrained=True).to(device),
              ptcv_get_model('resnext29_32x4d_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet20_cifar10', pretrained=True).to(device), 
              ptcv_get_model('densenet40_k12_cifar10', pretrained=True).to(device), 
              ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device), 
              ptcv_get_model('diaresnet20_cifar10', pretrained=True).to(device),
              ptcv_get_model('shakeshakeresnet26_2x32d_cifar10', pretrained=True).to(device), 
              ptcv_get_model('ror3_56_cifar10', pretrained=True).to(device),
              ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device) ]

model = Ensemble(model_list)

loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

# ## FGSM
# adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
# print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')
# create_dir(root, 'fgsm', adv_examples, adv_names)
# os.system("cd /home/lab530/KenYu/ML2022/hw10_attack/fgsm")
# os.system("tar zcf ../fgsm.tgz *")
# print(f"output fgsm result to fgsm.tgz")

# ## I-FGSM
# adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
# print(f'ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')
# create_dir(root, 'ifgsm', adv_examples, adv_names)
# os.system("cd /home/lab530/KenYu/ML2022/hw10_attack/ifgsm")
# os.system("tar zcf ../ifgsm.tgz *")
# print(f"output ifgsm result to ifgsm.tgz")

# ## MI-FGSM
adv_examples, mifgsm_acc, mifgsm_loss = gen_adv_examples(model, adv_loader, mifgsm, loss_fn)
print(f'mifgsm_acc = {mifgsm_acc:.5f}, mifgsm_loss = {mifgsm_loss:.5f}')
create_dir(root, 'mifgsm', adv_examples, adv_names)
os.system("cd /home/lab530/KenYu/ML2022/hw10_attack/mifgsm; tar zcf ../mifgsm.tgz *")
print(f"output ifgsm result to mifgsm.tgz")
