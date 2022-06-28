import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
import argparse
import time

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        # self.encoder = nn.Sequential( (046/050) Save best model, loss = 0.1399, Score: 0.71554
        #     nn.Linear(64 * 64 * 3, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(), 
        #     nn.Linear(64, 12), 
        #     nn.ReLU(), 
        #     nn.Linear(12, 3)
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 12),
        #     nn.ReLU(), 
        #     nn.Linear(12, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(), 
        #     nn.Linear(128, 64 * 64 * 3), 
        #     nn.Tanh()
        # )
        # self.encoder = nn.Sequential( # (049/050) Save best model, loss = 0.0664, Score: 0.75729
        #     nn.Linear(64 * 64 * 3, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(), 
        #     nn.Linear(256, 64), 
        #     nn.ReLU(), 
        #     nn.Linear(64, 16)
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(16, 64),
        #     nn.ReLU(), 
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024),
        #     nn.ReLU(), 
        #     nn.Linear(1024, 64 * 64 * 3), 
        #     nn.Tanh()
        # )
        # self.encoder = nn.Sequential( # (099/100) Save best model, loss = 0.0762
        #     nn.Linear(64 * 64 * 3, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(), 
        #     nn.Linear(512, 256), 
        #     nn.ReLU(), 
        #     nn.Linear(256, 128), 
        #     nn.ReLU(), 
        #     nn.Linear(128, 64), 
        #     nn.ReLU(), 
        #     nn.Linear(64, 32), 
        #     nn.ReLU(), 
        #     nn.Linear(32, 16)
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ReLU(), 
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(), 
        #     nn.Linear(1024, 64 * 64 * 3), 
        #     nn.Tanh()
        # )

        # self.encoder = nn.Sequential( # (097/100) Save best model, loss = 0.0353, Score: 
        #     nn.Linear(64 * 64 * 3, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(), 
        #     nn.Linear(1024, 256), 
        #     nn.ReLU(), 
        #     nn.Linear(256, 64)
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.ReLU(), 
        #     nn.Linear(256, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 4096),
        #     nn.ReLU(), 
        #     nn.Linear(4096, 64 * 64 * 3), 
        #     nn.Tanh()
        # )
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 8192),
            nn.ReLU(),
            nn.Linear(8192, 2048),
            nn.ReLU(), 
            nn.Linear(2048, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(), 
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(), 
            nn.Linear(8192, 64 * 64 * 3), 
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.ReLU(),
			      nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			      nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
			      nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

def eval_model(config):
    anomality = list()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            img = data.float().cuda()
            if config.model_type in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)
            if config.model_type in ['vae', 'my_vae']:
                output = output[0]
            if config.model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])
            anomality.append(loss)
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['score'])
    print(f"Write to {config.out_file}")
    df.to_csv(config.out_file, index_label = 'ID')

def train_model(config):
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = np.inf
    model.train()

    # qqdm_train = qqdm(range(config.num_epochs), desc=format_str('bold', 'Description'))
    # for epoch in qqdm_train:
    t_start = time.time()
    for epoch in range(config.num_epochs):
        tot_loss = list()
        for data in train_dataloader:

            # ===================loading=====================
            img = data.float().cuda()
            if config.model_type in ['fcn']:
                img = img.view(img.shape[0], -1)

            # ===================forward=====================
            output = model(img)
            if config.model_type in ['vae', 'my_vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            tot_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, 'best_model_{}.pt'.format(config.model_type))
            print("({:03d}/{:03d}) Save best model, loss = {:.4f}".format(epoch+1, config.num_epochs, mean_loss))
        # ===================log========================
        print("({:03d}/{:03d})    elapse : {:.2f}s    loss : {:.4f}".format(epoch+1, config.num_epochs, time.time() - t_start, mean_loss))
        t_start = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Parse argument
    parser.add_argument('--num_epochs', type = int, default = 100) # 50
    parser.add_argument('--batch_size', type = int, default = 2000)
    parser.add_argument('--eval_batch_size', type = int, default = 200)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--model_type', type = str, default = 'fcn')
    parser.add_argument('--out_file', type = str, default = 'prediction.csv')
    config = parser.parse_args()

    SEED = 5278
    train = np.load('data/trainingset.npy', allow_pickle=True)
    test = np.load('data/testingset.npy', allow_pickle=True)

    # Random seed
    same_seeds(SEED)

    # Build training dataloader
    x = torch.from_numpy(train)
    train_dataset = CustomTensorDataset(x)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    # build testing dataloader
    data = torch.tensor(test, dtype=torch.float32)
    test_dataset = CustomTensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.eval_batch_size, num_workers=1)

    # Model
    # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
    model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE()}
    model = model_classes[config.model_type].cuda()
    train_model(config)
    # 
    # load trained model
    checkpoint_path = f'best_model_{config.model_type}.pt'
    model = torch.load(checkpoint_path)
    model.eval()
    # 
    eval_loss = nn.MSELoss(reduction='none')
    # 
    eval_model(config)

