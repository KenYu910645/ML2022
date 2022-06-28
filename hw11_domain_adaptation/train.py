import pandas as pd
import cv2
import numpy as np
import argparse
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# # Model
# Feature Extractor: Classic VGG-like architecture
# Label Predictor / Domain Classifier: Linear models.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

# # Start Training
# ## DaNN Implementation
# 
# In the original paper, Gradient Reversal Layer is used.
# Feature Extractor, Label Predictor, and Domain Classifier are all trained at the same time. In this code, we train Domain Classifier first, and then train our Feature Extractor (same concept as Generator and Discriminator training process in GAN).
# 
# ## Reminder
# * Lambda, which controls the domain adversarial loss, is adaptive in the original paper. You can refer to [the original work](https://arxiv.org/pdf/1505.07818.pdf) . Here lambda is set to 0.1.
# * We do not have the label to target data, you can only evaluate your model by uploading your result to kaggle.:)

def train_epoch(source_dataloader, target_dataloader, epoch, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.to(config.device)
        source_label = source_label.to(config.device)
        target_data = target_data.to(config.device)
        
        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(config.device)
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        
        # lamb schedualing
        if not config.fix_lamb:
            p = epoch / config.num_epochs
            lamb = 2. / (1+np.exp(-10*p)) - 1

        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        # print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Parse argument
    parser.add_argument('--num_epochs', type = int, default = 6000)
    parser.add_argument('--save_interval', type = int, default = 100) # Save models every 10 epoches
    parser.add_argument('--lamb', type = float, default = 1.0)
    parser.add_argument('--device', type = str, default = "cuda:0")
    parser.add_argument('--fix_lamb', type = bool, default = False)
    parser.add_argument('--exp_name', type = str, default = 'very_long')
    config = parser.parse_args()
    print(config)

    source_transform = transforms.Compose([
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array, 
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    
    target_transform = transforms.Compose([
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to 
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    
    source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
    target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
    
    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    # Pre-processing, Here we use Adam as our optimizor.
    feature_extractor = FeatureExtractor().to(config.device)
    label_predictor = LabelPredictor().to(config.device)
    domain_classifier = DomainClassifier().to(config.device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

    # training 
    for epoch in range(config.num_epochs):
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, epoch, lamb=config.lamb)
        if (epoch+1)%config.save_interval == 0:
            torch.save(feature_extractor.state_dict(), f'{config.exp_name}_extractor_model.bin')
            torch.save(label_predictor.state_dict(), f'{config.exp_name}_predictor_model.bin')
            print("Saved models")
        
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, train_D_loss, train_F_loss, train_acc))

    # Inference
    result = []
    result_logits = []
    label_predictor.eval()
    feature_extractor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.to(config.device)

        class_logits = label_predictor(feature_extractor(test_data))
        result_logits.append(class_logits.cpu().detach().numpy())
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)
    result_logits = np.concatenate(result_logits)
    print(f"len(result_logits) = {len(result_logits)}")
    
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(config.exp_name + "_submit.csv",index=False)
    print(f"Output prediction to {config.exp_name}_submit.csv")

    # Generate logit, for label balancing
    dic = {'id': []}
    for i in range(10):
        dic[f"c_{i}"] = []
    df = pd.DataFrame(dic)
    for i, logit in enumerate(result_logits):
        df.loc[len(df.index)] = np.insert(logit, 0, i)
    df.to_csv(f"{config.exp_name}_logits.csv",index=False)
    print(f"Output logits to {config.exp_name}_logits.csv")