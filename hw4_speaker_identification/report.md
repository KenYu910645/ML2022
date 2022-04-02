## ML HW04 Report

1. Make a brief introduction about a variant of Transformer. (2 pts)
Ans: Big Bird is a variant of a transformer which focuses on improving efficiency of the attention matrix. Its attention matrix is the combination of random attention, window attention and global attention. This improvement makes Big Bird focus on finding relationships between random frame, neighbor frame and globally information; therefore, Big Bird typically yields a better result than vanilla transformer.

2. Briefly explain why adding convolutional layers to Transformer can boost 
performance. (2 pts)
Convolutional layers could work greatly when data has strong locality. For example, frames in audio usually have a tight relationship with its neighbor frames. Neighbor frames could form a word or phoneme when considered as a whole, which gives the network a lot more information than considering a single frame. Thus, adding convolutional layers to the transformer is to exploit this trait in the dataset; and in most cases, this adjustment could boost its performance.


Baseline Methods
○ Simple(0.60824): Run sample code & know how to use Transformer.
○ Medium(0.70375): Know how to adjust parameters of Transformer.
○ Strong(0.77750): Construct Conformer, which is a variety of Transformer.
○ Boss(0.86500): Implement Self-Attention Pooling & Additive Margin Softmax to further boost the 
performance.


## Vanilla
python train.py \
--batch_size 32 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 70000 \
result > vanilla.txt
Train: 100% 2000/2000 [00:19<00:00, 100.63 step/s, avg_acc=0.75, avg_loss=1.04]
Valid: 100% 5664/5667 [00:00<00:00, 7188.03 uttr/s, accuracy=0.67, loss=1.48]
Summit: 0.59950

## Conformer
python train.py \
--batch_size 32 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
> result.conformer.txt
Step 280000, best model saved. (accuracy=0.7472)
Train: 100% 2000/2000 [00:47<00:00, 41.81 step/s, avg_acc=0.78, avg_loss=0.85]
Valid: 100% 5664/5667 [00:01<00:00, 4708.22 uttr/s, accuracy=0.75, loss=1.10]
c

## Desperate
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 100000 \
--n_encoder_layers 3 \
--encoder_dim 160 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate.txt
0.76325

## desperate_2
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 160 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate_2.txt
Summit: 0.76800

## desperate_3
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 150000 \
--n_encoder_layers 3 \
--encoder_dim 240 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate_3.txt
Summit: 0.77475

## desperate_4
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 240 \
--dropout 0.5 \
--trainsplit 0.99 \
> result/desperate_4.txt
Summit: 0.70450

## Desperate_5
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 240 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate_5.txt
Summit: 0.77100

## Desperate_6
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 320 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate_6.txt
Summit: 0.78575

## Desperate_7
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 400 \
--dropout 0.1 \
--trainsplit 0.99 \
> result/desperate_7.txt

## Desperate_8
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 240 \
--dropout 0.1 \
--trainsplit 0.99 \
--output_fn desperate_8.txt \
--n_heads 4 \
> result/desperate_8.txt;
Summit: 0.76800

## Desperate_9
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 256 \
--dropout 0.1 \
--trainsplit 0.99 \
--output_fn desperate_9.txt \
--n_heads 16 \
> result/desperate_9.txt;
Summit: 0.77225

## Desperate_10
python train.py \
--batch_size 64 \
--valid_steps 2000 \
--warmup_steps 1000 \
--total_steps 300000 \
--n_encoder_layers 3 \
--encoder_dim 240 \
--dropout 0.1 \
--trainsplit 0.99 \
--output_fn desperate_10.txt \
--n_heads 2 \
> result/desperate_10.txt;
Summit: 0.75875

## conformer encoder dim
### encoder_dim = 64
(150000/150000[Train] loss: 0.168754, acc: 0.955812 [Valid] loss: 118.098211, acc: 0.867232 
Save model at 150000
### encoder_dim = 80
(150000/150000[Train] loss: 0.072734, acc: 0.981844 [Valid] loss: 102.145349, acc: 0.890537 
Save model at 150000
### encoder_dim = 160
(150000/150000)[Train] loss: 0.008481, acc: 0.997750 [Valid] loss: 74.043092, acc: 0.923552 
Save model at 150000
summit: 0.78550

## conformer encoder layers(mod 16 must be zero)
### encoder_layers = 1
[Train] loss: 1.339277, acc: 0.674484 [Valid] loss: 270.686680, acc: 0.647952 
Save model at 150000
### encoder_layers = 2
[Train] loss: 1.145505, acc: 0.718281 [Valid] loss: 239.261314, acc: 0.684675 
Save model at 150000
### encoder_layers = 3
[Train] loss: 1.042559, acc: 0.739781 [Valid] loss: 224.861142, acc: 0.708510 
Save model at 150000
### encoder_layers = 4
(150000/150000)[Train] loss: 1.043084, acc: 0.739156 [Valid] loss: 222.209361, acc: 0.704096 
Save model at 150000
### encoder_layers = 5
(150000/150000)[Train] loss: 1.054640, acc: 0.734953 [Valid] loss: 220.474628, acc: 0.710452 
Save model at 150000
Summit score: 0.66150

## batch size 
### batch_size = 4
(150000/150000)[Train] loss: 2.480545, acc: 0.426000 [Valid] loss: 3701.009697, acc: 0.400600 
Save model at 150000
### batch_size = 8
(150000/150000)[Train] loss: 0.541049, acc: 0.871000 [Valid] loss: 574.743985, acc: 0.805261 
Save model at 150000
### batch_size = 16
(150000/150000)[Train] loss: 0.217577, acc: 0.945312 [Valid] loss: 219.122425, acc: 0.867232 
Save model at 150000
### batch_size = 32
(150000/150000)[Train] loss: 0.072734, acc: 0.981844 [Valid] loss: 102.145349, acc: 0.890537 
Save model at 150000
### batch_size = 64
(150000/150000)[Train] loss: 0.020335, acc: 0.995312 [Valid] loss: 48.516028, acc: 0.900391 
Save model at 150000
### batch_size = 128
(150000/150000)[Train] loss: 0.008125, acc: 0.997836 [Valid] loss: 28.189551, acc: 0.904119 
Save model at 150000
### batch_size = 256
(150000/150000)[Train] loss: 0.003953, acc: 0.998559 [Valid] loss: 15.755387, acc: 0.904830 
Save model at 150000

## Dropout 
### Dropout = 0.2
(150000/150000)[Train] loss: 0.049120, acc: 0.986898 [Valid] loss: 51.228907, acc: 0.898260
Save model at 150000
### Dropout = 0.3
(150000/150000)[Train] loss: 0.102021, acc: 0.971336 [Valid] loss: 57.489249, acc: 0.883878 
Save model at 150000
Summit: 0.68125
### Dropout = 0.4
(150000/150000)[Train] loss: 0.198713, acc: 0.943320 [Valid] loss: 62.274566, acc: 0.874112 
Save model at 150000














