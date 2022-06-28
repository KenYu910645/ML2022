# This is the sample code for homework 15.
import glob, random
from collections import OrderedDict
import os
import numpy as np
from tqdm.auto import tqdm
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display
import argparse

# ### Model Construction Preliminaries
# Since our task is image classification, we need to build a CNN-based model.  
# However, to implement MAML algorithm, we should adjust some code in `nn.Module`.
# Take a look at MAML pseudocode...
# On the 10-th line, what we take gradients on are those $\theta$ representing  
# <font color="#0CC">**the original model parameters**</font> (outer loop) instead of those in  the  
# <font color="#0C0">**inner loop**</font>, so we need to use `functional_forward` to compute the output  
# logits of input image instead of `forward` in `nn.Module`.
# 
# The following defines these functions.
# <!-- 由於在第10行，我們是要對原本的參數 θ 微分，並非 inner-loop (Line5~8) 的 θ' 微分，因此在 inner-loop，我們需要用 functional forward 的方式算出 input image 的 output logits，而不是直接用 nn.module 裡面的 forward（直接對 θ 微分）。在下面我們分別定義了 functional forward 以及 forward 函數。 -->

# ### Model block definition
def ConvBlock(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(
        x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True
    )
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x

# ### Model definition
class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        """
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: model parameters,
                i.e. weights and biases of convolution
                     and weights and biases of
                                   batch normalization
                type is an OrderedDict

        Arguments:
        x: input images [batch, 1, 28, 28]
        params: The model parameters,
                i.e. weights and biases of convolution
                     and batch normalization layers
                It's an `OrderedDict`
        """
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(
                x,
                params[f"conv{block}.0.weight"],
                params[f"conv{block}.0.bias"],
                params.get(f"conv{block}.1.weight"),
                params.get(f"conv{block}.1.bias"),
            )
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params["logits.weight"], params["logits.bias"])
        return x

# ### Create Label
# This function is used to create labels.  
# In a N-way K-shot few-shot classification problem,
# each task has `n_way` classes, while there are `k_shot` images for each class.  
# This is a function that creates such labels.
def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()

# ### Accuracy calculation
def calculate_accuracy(logits, labels):
    """utility function for accuracy calculation"""
    acc = np.asarray(
        [(torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())]
    ).mean()
    return acc

# ### Define Dataset
# Define the dataset.  
# The dataset returns images of a random character, with (`k_shot + q_query`) images,  
# so the size of returned tensor is `[k_shot+q_query, 1, 28, 28]`.  

# Dataset for train and val
class Omniglot(Dataset):
    def __init__(self, data_dir, k_way, q_query, task_num=None):
        self.file_list = [
            f for f in glob.glob(data_dir + "**/character*", recursive=True)
        ]
        # limit task number if task_num is set
        if task_num is not None:
            self.file_list = self.file_list[: min(len(self.file_list), task_num)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_way + q_query

    def __getitem__(self, idx):
        sample = np.arange(20)

        # For random sampling the characters we want.
        np.random.shuffle(sample)
        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        # `k_way + q_query` examples for each character
        imgs = torch.stack(imgs)[sample[: self.n]]
        return imgs

    def __len__(self):
        return len(self.file_list)

# ## **Step 3: Learning Algorithms**
# ### Transfer learning
# The solver first chose five task from the training set, then do normal classification training on the chosen five tasks. In inference, the model finetune for `inner_train_step` steps on the support set images, and than do inference on the query set images.
# For consistant with the meta-learning solver, the base solver has the exactly same input and output format with the meta-learning solver.

def BaseSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False,
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # Get data
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]

        if train:
            """ training loop """
            # Use the support set to calculate loss
            labels = create_label(n_way, k_shot).to(config.device)
            logits = model.forward(support_set)
            loss = criterion(logits, labels)

            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, labels))
        else:
            """ validation / testing loop """
            # First update model with support set images for `inner_train_step` steps
            fast_weights = OrderedDict(model.named_parameters())


            for inner_step in range(inner_train_step):
                # Simply training
                train_label = create_label(n_way, k_shot).to(config.device)
                logits = model.functional_forward(support_set, fast_weights)
                loss = criterion(logits, train_label)

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # Perform SGD
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            if not return_labels:
                """ validation """
                val_label = create_label(n_way, q_query).to(config.device)

                logits = model.functional_forward(query_set, fast_weights)
                loss = criterion(logits, val_label)
                task_loss.append(loss)
                task_acc.append(calculate_accuracy(logits, val_label))
            else:
                """ testing """
                logits = model.functional_forward(query_set, fast_weights)
                labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    batch_loss = torch.stack(task_loss).mean()
    task_acc = np.mean(task_acc)

    if train:
        # Update model
        model.train()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return batch_loss, task_acc

# ### Meta Learning
# Here is the main Meta Learning algorithm.
# Please finish the TODO blocks for the inner and outer loop update rules.
# - For implementing FO-MAML you can refer to [p.25 of the slides](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf#page=25&view=FitW).
# - For the original MAML, you can refer to [the slides of meta learning (p.13 ~ p.18)](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf#page=13&view=FitW).

def MetaSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # Get data
        support_set = meta_batch[: n_way * k_shot]
        query_set   = meta_batch[n_way * k_shot :]

        # Copy the params for inner loop
        fast_weights = OrderedDict(model.named_parameters())

        ### ---------- INNER TRAIN LOOP ---------- ###
        for inner_step in range(inner_train_step):
            # Simply training
            train_label = create_label(n_way, k_shot).to(config.device)
            logits = model.functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            # Inner gradients update!
            """ Inner Loop Update """ # TODO: Finish the inner loop update rule
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)                         #
            # SGD
            fast_weights = OrderedDict( (name, param - inner_lr * grad)
                                        for ((name, param), grad)
                                            in zip(fast_weights.items(), grads) )

        ### ---------- INNER VALID LOOP ---------- ###
        if not return_labels:
            """ training / validation """
            val_label = create_label(n_way, q_query).to(config.device)

            # Collect gradients for outer loop
            logits = model.functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            """ testing """
            logits = model.functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    # Update outer loop
    model.train()
    optimizer.zero_grad()

    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        """ Outer Loop Update """
        # TODO: Finish the outer loop update
        meta_batch_loss.backward()
        optimizer.step()

    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc

# ## **Step 4: Initialization**
# After defining all components we need, the following initialize a model before training.

# ### Dataloader initialization
def dataloader_init(datasets, shuffle=True, num_workers=2):
    train_set, val_set = datasets
    train_loader = DataLoader(
        train_set,
        # The "batch_size" here is not \
        #    the meta batch size, but  \
        #    how many different        \
        #    characters in a task,     \
        #    i.e. the "n_way" in       \
        #    few-shot classification.
        batch_size=config.n_way,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.n_way, num_workers=num_workers, shuffle=shuffle, drop_last=True
    )

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    return (train_loader, val_loader), (train_iter, val_iter)

# ### Model & optimizer initialization
def model_init():
    meta_model = Classifier(1, config.n_way).to(config.device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=config.meta_lr)
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    return meta_model, optimizer, loss_fn

# ### Utility function to get a meta-batch
def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    data = []
    for _ in range(meta_batch_size):
        try:
            # a "task_data" tensor is representing \
            #     the data of a task, with size of \
            #     [n_way, k_shot+q_query, 1, 28, 28]
            task_data = iterator.next()
        except StopIteration:
            iterator = iter(data_loader)
            task_data = iterator.next()
        train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
        val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)
    return torch.stack(data).to(config.device), iterator

# test dataset
class OmniglotTest(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.n = 5
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        support_files = [
            os.path.join(self.test_dir, "support", f"{idx:>04}", f"image_{i}.png")
            for i in range(self.n)
        ]
        query_files = [
            os.path.join(self.test_dir, "query", f"{idx:>04}", f"image_{i}.png")
            for i in range(self.n)
        ]

        support_imgs = torch.stack(
            [self.transform(Image.open(e)) for e in support_files]
        )
        query_imgs = torch.stack([self.transform(Image.open(e)) for e in query_files])

        return support_imgs, query_imgs

    def __len__(self):
        return len(os.listdir(os.path.join(self.test_dir, "support")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Parse argument
    # 
    parser.add_argument('--n_way', type = int, default = 5)
    parser.add_argument('--k_shot', type = int, default = 1)
    parser.add_argument('--q_query', type = int, default = 1)
    parser.add_argument('--train_inner_train_step', type = int, default = 8)
    parser.add_argument('--val_inner_train_step', type = int, default = 3)
    parser.add_argument('--inner_lr', type = float, default = 0.2)
    parser.add_argument('--meta_lr', type = float, default = 0.001)
    parser.add_argument('--meta_batch_size', type = int, default = 32)
    parser.add_argument('--max_epoch', type = int, default = 500)
    parser.add_argument('--eval_batches', type = int, default = 20)
    parser.add_argument('--train_data_path', type = str, default = "./Omniglot/images_background/")
    parser.add_argument('--task_num', type = int, default = 20)
    # 
    parser.add_argument('--device', type = str, default = 'cuda:1')
    parser.add_argument('--out_file', type = str, default = 'output.csv')
    # 
    parser.add_argument('--test_inner_train_step', type = int, default = 100)
    config = parser.parse_args()
    # 
    # device
    print(f"DEVICE = {config.device}")

    # Fix random seeds
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # ### Start training!
    solver = 'meta' # 'base' # base, meta
    meta_model, optimizer, loss_fn = model_init()
    # init solver and dataset according to solver type
    if solver == 'base':
        max_epoch = 5 # the base solver only needs 5 epochs
        Solver = BaseSolver
        train_set, val_set = torch.utils.data.random_split(
            Omniglot(config.train_data_path, config.k_shot, config.q_query, task_num=config.task_num), [5, 5]
        )
        (train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set), shuffle=False)
    elif solver == 'meta':
        Solver = MetaSolver
        dataset = Omniglot(config.train_data_path, config.k_shot, config.q_query)
        train_split = int(0.8 * len(dataset))
        val_split = len(dataset) - train_split
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_split, val_split]
        )
        (train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set))
    else:
        raise NotImplementedError
    
    ##########################
    ### main training loop ###
    ##########################
    for epoch in range(config.max_epoch):
        print("Epoch %d" % (epoch + 1))
        train_meta_loss = []
        train_acc = []
        # The "step" here is a meta-gradinet update step
        for step in tqdm(range(max(1, len(train_loader) // config.meta_batch_size))):
            x, train_iter = get_meta_batch(
                config.meta_batch_size, config.k_shot, config.q_query, train_loader, train_iter
            )
            meta_loss, acc = Solver(
                meta_model,
                optimizer,
                x,
                config.n_way,
                config.k_shot,
                config.q_query,
                loss_fn, 
                inner_train_step=config.train_inner_train_step
            )
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)
        print("  Loss    : ", "%.3f" % (np.mean(train_meta_loss)), end="\t")
        print("  Accuracy: ", "%.3f %%" % (np.mean(train_acc) * 100))

        # See the validation accuracy after each epoch.
        # Early stopping is welcomed to implement.
        val_acc = []
        for eval_step in tqdm(range(max(1, len(val_loader) // (config.eval_batches)))):
            x, val_iter = get_meta_batch(
                config.eval_batches, config.k_shot, config.q_query, val_loader, val_iter
            )
            # We update three inner steps when testing.
            _, acc = Solver(
                meta_model,
                optimizer,
                x,
                config.n_way,
                config.k_shot,
                config.q_query,
                loss_fn,
                inner_train_step=config.val_inner_train_step,
                train=False,
            )
            val_acc.append(acc)
        print("  Validation accuracy: ", "%.3f %%" % (np.mean(val_acc) * 100))
    
    # ### Testing the result
    # Since the testing data is sampled by TAs in advance, you should not change the code in `OmnigloTest` dataset, otherwise your score may not be correct on the Kaggle leaderboard.
    # However, fell free to chagne the variable `inner_train_step` to set the training steps on the query set images.
    
    test_batches = 20
    test_dataset = OmniglotTest("Omniglot-test")
    test_loader = DataLoader(test_dataset, batch_size=test_batches, shuffle=False)
    output = []
    for _, batch in enumerate(tqdm(test_loader)):
        support_set, query_set = batch
        x = torch.cat([support_set, query_set], dim=1)
        x = x.to(config.device)

        labels = Solver(
            meta_model,
            optimizer,
            x,
            config.n_way,
            config.k_shot,
            config.q_query,
            loss_fn,
            inner_train_step=config.test_inner_train_step,
            train=False,
            return_labels=True,
        )

        output.extend(labels)

    # write to csv
    with open(config.out_file, "w") as f:
        f.write(f"id,class\n")
        for i, label in enumerate(output):
            f.write(f"{i},{label}\n")
