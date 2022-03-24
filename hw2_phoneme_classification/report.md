# Report Questions
1. (2%) Implement 2 models with approximately the same number of
parameters, (A) one narrower and deeper (e.g. hidden_layers=6,
hidden_dim=1024) and (B) the other wider and shallower (e.g.
hidden_layers=2, hidden_dim=1700). Report training/validation accuracies
for both models.
```
hidden_layers = 6
hidden_dim = 1024
Train Acc: 0.768464 Loss: 0.723376 | Val Acc: 0.697791 loss: 0.997525
```
```
hidden_layers = 2
hidden_dim = 1700
Train Acc: 0.790975 Loss: 0.642395 | Val Acc: 0.707990 loss: 0.957660
```

2. (2%) Add dropout layers, and report training/validation accuracies with
dropout rates equal to (A) 0.25/(B) 0.5/(C) 0.75 respectively.
Dropout = 0.25
Train Acc: 0.676947 Loss: 1.027557 | Val Acc: 0.699363 loss: 0.947729
Dropout = 0.5
Train Acc: 0.612089 Loss: 1.267250 | Val Acc: 0.663171 loss: 1.081971
Dropout = 0.75
Train Acc: 0.507420 Loss: 1.671693 | Val Acc: 0.571902 loss: 1.415980

# Hyperparamter Tuning
## concat_nframes
concat_nframes = 1, Train Acc: 0.460765 Loss: 1.876439 | Val Acc: 0.457733 loss: 1.889798
concat_nframes = 3, Train Acc: 0.527284 Loss: 1.604429 | Val Acc: 0.525021 loss: 1.613698
concat_nframes = 5, Train Acc: 0.569158 Loss: 1.437357 | Val Acc: 0.565352 loss: 1.451889
concat_nframes = 7, Train Acc: 0.600337 Loss: 1.314451 | Val Acc: 0.595854 loss: 1.330958
concat_nframes = 9, Train Acc: 0.620403 Loss: 1.242060 | Val Acc: 0.614699 loss: 1.263183
concat_nframes = 11, Train Acc: 0.634625 Loss: 1.192598 | Val Acc: 0.627062 loss: 1.217146
concat_nframes = 13, Train Acc: 0.645126 Loss: 1.156165 | Val Acc: 0.636443 loss: 1.185240
concat_nframes = 15, Train Acc: 0.654117 Loss: 1.125561 | Val Acc: 0.643864 loss: 1.159003
concat_nframes = 17, Train Acc: 0.658831 Loss: 1.110749 | Val Acc: 0.647566 loss: 1.145717
concat_nframes = 19, Train Acc: 0.663342 Loss: 1.095437 | Val Acc: 0.652599 loss: 1.131236
concat_nframes = 21, Train Acc: 0.665973 Loss: 1.086367 | Val Acc: 0.653866 loss: 1.126962
concat_nframes = 23, Train Acc: 0.668964 Loss: 1.077869 | Val Acc: 0.655376 loss: 1.123864
concat_nframes = 25, Train Acc: 0.671356 Loss: 1.069241 | Val Acc: 0.656668 loss: 1.117480
concat_nframes = 27, Train Acc: 0.673030 Loss: 1.063654 | Val Acc: 0.658713 loss: 1.112688
concat_nframes = 29, Train Acc: 0.675000 Loss: 1.057856 | Val Acc: 0.659408 loss: 1.111326
concat_nframes = 31, Train Acc: 0.674947 Loss: 1.058872 | Val Acc: 0.658548 loss: 1.115292
concat_nframes = 33, Train Acc: 0.675431 Loss: 1.056738 | Val Acc: 0.658870 loss: 1.114567
concat_nframes = 35, Train Acc: 0.677840 Loss: 1.048276 | Val Acc: 0.658813 loss: 1.112710
concat_nframes = 37, Train Acc: 0.678057 Loss: 1.048830 | Val Acc: 0.658597 loss: 1.114388
concat_nframes = 39, Train Acc: 0.678264 Loss: 1.046738 | Val Acc: 0.658523 loss: 1.116510
conculsion: concat_nframes = 19 might be a good choice

## hidden_layers
hidden_layers = 1, Train Acc: 0.663342 Loss: 1.095437 | Val Acc: 0.652599 loss: 1.131236
hidden_layers = 2, Train Acc: 0.668376 Loss: 1.070635 | Val Acc: 0.657756 loss: 1.110547
hidden_layers = 3, Train Acc: 0.670404 Loss: 1.060443 | Val Acc: 0.659183 loss: 1.099336
hidden_layers = 4, Train Acc: 0.668912 Loss: 1.063594 | Val Acc: 0.658887 loss: 1.097427
hidden_layers = 5, Train Acc: 0.661728 Loss: 1.087761 | Val Acc: 0.652945 loss: 1.120069
hidden_layers = 6, Train Acc: 0.656251 Loss: 1.108532 | Val Acc: 0.647040 loss: 1.140369
hidden_layers = 7, Train Acc: 0.650320 Loss: 1.137852 | Val Acc: 0.640029 loss: 1.175111
hidden_layers = 8, Train Acc: 0.643713 Loss: 1.184423 | Val Acc: 0.632816 loss: 1.216210
hidden_layers = 9, Train Acc: 0.641748 Loss: 1.217196 | Val Acc: 0.632066 loss: 1.247059
hidden_layers = 10, Train Acc: 0.641307 Loss: 1.245090 | Val Acc: 0.628919 loss: 1.290873
conculsion: hidden_layers = 3 might be a good choice

## hidden_dim
hidden_dim = 8, Train Acc: 0.420216 Loss: 2.003623 | Val Acc: 0.421402 loss: 1.999117
hidden_dim = 16, Train Acc: 0.481153 Loss: 1.765994 | Val Acc: 0.481493 loss: 1.762961
hidden_dim = 32, Train Acc: 0.534891 Loss: 1.549958 | Val Acc: 0.534042 loss: 1.553651
hidden_dim = 64, Train Acc: 0.588259 Loss: 1.366736 | Val Acc: 0.583365 loss: 1.376734
hidden_dim = 128, Train Acc: 0.629257 Loss: 1.210180 | Val Acc: 0.624214 loss: 1.227887
hidden_dim = 256,  Train Acc: 0.670404 Loss: 1.060443 | Val Acc: 0.659183 loss: 1.099336
hidden_dim = 400, Train Acc: 0.695086 Loss: 0.972287 | Val Acc: 0.678238 loss: 1.031443
hidden_dim = 512, Train Acc: 0.709744 Loss: 0.919581 | Val Acc: 0.685805 loss: 1.005741
hidden_dim = 800, Train Acc: 0.739585 Loss: 0.813549 | Val Acc: 0.699060 loss: 0.963134
hidden_dim = 1024, Train Acc: 0.757734 Loss: 0.752121 | Val Acc: 0.704494 loss: 0.948815
hidden_dim = 1500, Train Acc: 0.793835 Loss: 0.629605 | Val Acc: 0.709686 loss: 0.964070
hidden_dim = 2048, Train Acc: 0.835254 Loss: 0.494024 | Val Acc: 0.709871 loss: 1.025235
conculsion: hidden_dim = 800 might be a good choice

## epoch
epoch = 5, Train Acc: 0.739585 Loss: 0.813549 | Val Acc: 0.699060 loss: 0.963134
epoch = 10,  Train Acc: 0.804825 Loss: 0.594218 | Val Acc: 0.704333 loss: 0.989826
epoch = 15, Train Acc: 0.856899 Loss: 0.424668 | Val Acc: 0.694918 loss: 1.136561
epoch = 20, Train Acc: 0.897662 Loss: 0.298322 | Val Acc: 0.685909 loss: 1.371586
epoch = 25, Train Acc: 0.926269 Loss: 0.211759 | Val Acc: 0.680835 loss: 1.605206
epoch = 30, Train Acc: 0.945504 Loss: 0.154174 | Val Acc: 0.674946 loss: 1.852128
conculsion: epoch = 5 might be a good choice

## Batchnorm 
nn.BatchNorm1d(output_dim)
without batch norm, Train Acc: 0.739585 Loss: 0.813549 | Val Acc: 0.699060 loss: 0.963134
with batch norm, Train Acc: 0.799357 Loss: 0.613472 | Val Acc: 0.703452 loss: 0.977680
conculsion: with batchnorm, it's loss goes down easier and faster.

## Dropout

# Summit result
## 1. 
concat_nframes = 19
batch_size = 512
num_epoch = 5
learning_rate = 0.0001
hidden_layers = 3
hidden_dim = 800
Train Acc: 0.739585 Loss: 0.813549 | Val Acc: 0.699060 loss: 0.963134
Summit score: 0.70158


## 2. 
concat_nframes = 19
batch_size = 512
num_epoch = 30
learning_rate = 0.0001
hidden_layers = 3
hidden_dim = 800
dropout = 0.5
batchnorm = True
Train Acc: 0.665804 Loss: 1.072449 | Val Acc: 0.714322 loss: 0.899154
Summit score: 0.71576

## 3. 
batch_size = 512
num_epoch = 30
learning_rate = 0.0001
hidden_layers = 3
hidden_dim = 2048
dropout = 0.5
batchnorm = True
Train Acc: 0.738253 Loss: 0.805192 | Val Acc: 0.752096 loss: 0.773446
Summit score: 0.75336

## 4. 
python ML2022Spring_HW2.py \
--concat_nframes 29 \
--batch_size     512 \
--num_epoch      70 \
--learning_rate  0.0001 \
--hidden_layers  10 \
--hidden_dim     2000 \
--dropout        0.5 \
--batchnorm \
[070/070] Train Acc: 0.767946 Loss: 0.743071 | Val Acc: 0.770041 loss: 0.741902

