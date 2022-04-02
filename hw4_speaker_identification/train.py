# # Task description
# - Classify the speakers of given features.
# - Main goal: Learn how to use transformer.
# - Baselines:
#   - Easy: Run sample code and know how to use transformer.
#   - Medium: Know how to adjust parameters of transformer.
#   - Strong: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer. 
#   - Boss: Implement [Self-Attention Pooling](https://arxiv.org/pdf/2008.01077v1.pdf) & [Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf) to further boost the performance.
# 
# - Other links
#   - Kaggle: [link](https://www.kaggle.com/t/ac77388c90204a4c8daebeddd40ff916)
#   - Slide: [link](https://docs.google.com/presentation/d/1HLAj7UUIjZOycDe7DaVLSwJfXVd3bXPOyzSb6Zk3hYU/edit?usp=sharing)
#   - Data: [link](https://drive.google.com/drive/folders/1vI1kuLB-q1VilIftiwnPOCAeOOFfBZge?usp=sharing)
# 
# # Download dataset
# - Data is [here](https://drive.google.com/drive/folders/1vI1kuLB-q1VilIftiwnPOCAeOOFfBZge?usp=sharing)
# Python import 
import numpy as np
import json
import csv
import os
import random
import json
import math
import argparse
# Pytorch 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from conformer import Conformer
# tqdm
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ## Dataset
# - Original dataset is [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
# - The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb2.
# - We randomly select 600 speakers from Voxceleb2.
# - Then preprocess the raw waveforms into mel-spectrograms.
# 
# - Args:
#   - data_dir: The path to the data directory.
#   - metadata_path: The path to the metadata.
#   - segment_len: The length of audio segment for training. 
# - The architecture of data directory \\
#   - data directory \\
#   |---- metadata.json \\
#   |---- testdata.json \\
#   |---- mapping.json \\
#   |---- uttr-{random string}.pt \\
# 
# - The information in metadata
#   - "n_mels": The dimention of mel-spectrogram.
#   - "speakers": A dictionary. 
#     - Key: speaker ids.
#     - value: "feature_path" and "mel_len"
# 
# 
# For efficiency, we segment the mel-spectrograms into segments in the traing step.

class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
        
        # Load the mapping from speaker neme to their corresponding id. 
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]
        
        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]
        
        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
    def __len__(self):
            return len(self.data)
 
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
 
    def get_speaker_number(self):
        return self.speaker_num

# ## Dataloader
# - Split dataset into training dataset(90%) and validation dataset(10%).
# - Create dataloader to iterate the data.

def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(config.trainsplit * len(dataset)) # int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    # y_dict = {}
    # for x, y in validset:
    #     y_scale = y.numpy()[0]
    #     try: 
    #         y_dict[y_scale] += 1
    #     except KeyError:
    #         y_dict[y_scale] = 1
    # print(y_dict)
    # print(len(y_dict))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num
# # Model
# - TransformerEncoderLayer:
#   - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
#   - Parameters:
#     - d_model: the number of expected features of the input (required).
#     - nhead: the number of heads of the multiheadattention models (required).
#     - dim_feedforward: the dimension of the feedforward network model (default=2048).
#     - dropout: the dropout value (default=0.1).
#     - activation: the activation function of intermediate layer, relu or gelu (default=relu).
# 
# - TransformerEncoder:
#   - TransformerEncoder is a stack of N transformer encoder layers
#   - Parameters:
#     - encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#     - num_layers: the number of sub-encoder-layers in the encoder (required).
#     - norm: the layer normalization component (optional).

class Confomer_warp(nn.Module):
    def __init__(self, n_spks):
        super().__init__()
        self.conformer = Conformer(num_classes=n_spks, 
                                    input_dim=40, 
                                    encoder_dim=config.encoder_dim, 
                                    num_encoder_layers=config.n_encoder_layers,
                                    input_dropout_p = config.dropout,
                                    feed_forward_dropout_p = config.dropout,
                                    attention_dropout_p = config.dropout,
                                    conv_dropout_p = config.dropout,
                                    num_attention_heads = config.n_heads
                                    )
        self.pred_layer = nn.Linear(config.encoder_dim, n_spks) # leanth
    
    def forward(self, mels):
        out = self.conformer.encoder(mels, 40)[0]
        # mean pooling
        stats = out.mean(dim=1)
        out = self.pred_layer(stats)
        return out

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        #
        # Conformer Refernce: https://github.com/lucidrains/conformer
        # https://github.com/sooftware/conformer
        # self.conformer = ConformerBlock(
        #     dim = 512,
        #     dim_head = 64,
        #     heads = 8,
        #     ff_mult = 4,
        #     conv_expansion_factor = 2,
        #     conv_kernel_size = 31,
        #     attn_dropout = 0.,
        #     ff_dropout = 0.,
        #     conv_dropout = 0.
        # )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2
        )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

# # Learning rate schedule
# - For transformer architecture, the design of learning rate schedule is different from that of CNN.
# - Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
# - The warmup schedule
#   - Set learning rate to 0 in the beginning.
#   - The learning rate increases linearly from 0 to initial learning rate during warmup period.

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# # Model Function
# - Model forward function.
def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    # Calculate CTC Loss
    loss = criterion(outs, labels)
    
    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy

# Validate, Calculate accuracy of the validation set.
def valid(dataloader, model, criterion, device): 
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0  

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
    model.train()

    return running_accuracy / len(dataloader), running_loss


def train_main():
    device = torch.device(DEVICE)
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(config.data_dir, config.batch_size, config.n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)
    
    # model = Classifier(n_spks=speaker_num).to(device)
    # https://github.com/sooftware/conformer
    model = Confomer_warp(n_spks=speaker_num).to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None
    batch_loss_avg = 0
    batch_accuracy_avg = 0
    for step in range(config.total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        batch_loss_avg += batch_loss/config.valid_steps
        batch_accuracy_avg += batch_accuracy/config.valid_steps

        # Do validation
        if (step + 1) % config.valid_steps == 0 and len(valid_loader) > 0:
            valid_accuracy, valid_loss = valid(valid_loader, model, criterion, device)
            print("({:06d}/{:06d})[Train] loss: {:.6f}, acc: {:.6f} [Valid] loss: {:.6f}, acc: {:.6f} ".format(step + 1, config.total_steps, batch_loss_avg, batch_accuracy_avg, valid_loss, valid_accuracy))
            batch_loss_avg = 0
            batch_accuracy_avg = 0

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

        # Save the best model so far.
        if ((step + 1) % config.save_steps == 0 and best_state_dict is not None) or len(valid_loader) <= 0 :
            torch.save(best_state_dict, config.save_path)
            print(f"Save model at {step + 1}")

# Inference, Dataset of inference
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel

def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)

def infer_main():
    device = torch.device(DEVICE)
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(config.data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open()

    dataset = InferenceDataset(config.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!",flush = True)

    speaker_num = len(mapping["id2speaker"])
    # model = Classifier(n_spks=speaker_num).to(device)
    model = Confomer_warp(n_spks=speaker_num).to(device)
    
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush = True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(config.output_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

if __name__ == "__main__": 
    SEED = 5278
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = "./Dataset")
    parser.add_argument('--save_path', type = str, default = "model.ckpt")
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--n_workers', type = int, default = 8)
    parser.add_argument('--valid_steps', type = int, default = 2000)
    parser.add_argument('--warmup_steps', type = int, default = 1000)
    parser.add_argument('--save_steps', type = int, default = 10000)
    parser.add_argument('--total_steps', type = int, default = 300000)
    parser.add_argument('--device', type = str, default = "cuda")
    parser.add_argument('--n_encoder_layers', type = int, default = 3)
    parser.add_argument('--encoder_dim', type = int, default = 32)
    parser.add_argument('--dropout', type = float, default = 0.1)
    parser.add_argument('--trainsplit', type = float, default = 0.1)
    parser.add_argument('--n_heads', type = int, default = 8)
    parser.add_argument('--output_fn', type = str, default = "summit.csv")

    # config.trainsplit
    config = parser.parse_args()
    DEVICE = config.device # DEVICE = "cuda"
    set_seed(SEED)
    train_main()
    infer_main()