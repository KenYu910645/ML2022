# Model Structure
## eff_b4.txt

python train.py \
--batch_size 64 \
--n_epochs 300 \
--patience 300 \
--input_size 224 \
--learning_rate 0.0003 > result/eff_b4.txt

Epoch(228/300) train_loss 0.03274 | train_acc 0.98770 | valid_loss 1.16298 | valid_acc 0.80464
[ Valid | 228/300 ] loss = 1.16298, acc = 0.80464 -> best
Best model found at epoch 227, saving model
Summit Result: 0.82968

## resnet101.txt
python train.py \
--batch_size 64 \
--n_epochs 300 \
--patience 300 \
--input_size 224 \
--learning_rate 0.0003 > result/resnet101.txt
Epoch(286/300) train_loss 0.01781 | train_acc 0.99385 | valid_loss 1.36456 | valid_acc 0.77936
[ Valid | 286/300 ] loss = 1.36456, acc = 0.77936 -> best
Best model found at epoch 285, saving model
Summit Result: 0.81175

## eff_b4_with_tfm
python train.py \
--batch_size 64 \
--n_epochs 300 \
--patience 300 \
--input_size 224 \
--learning_rate 0.0003 > result/eff_b4_train_tfm.txt
Epoch(228/300) train_loss 0.13857 | train_acc 0.95127 | valid_loss 1.33428 | valid_acc 0.71292
[ Valid | 228/300 ] loss = 1.33428, acc = 0.71292 -> best
Best model found at epoch 227, saving model

Summit Result: 0.76195

WithOut Gray + ColorJitter Transform
Epoch(280/300) train_loss 0.08069 | train_acc 0.97450 | valid_loss 1.00805 | valid_acc 0.79104
[ Valid | 280/300 ] loss = 1.00805, acc = 0.79104 -> best
Best model found at epoch 279, saving model

Summit Result: 0.81772


## eff_b4_with_tfm + TTA

Epoch(260/300) train_loss 0.03698 | train_acc 0.98740 | valid_loss 0.98445 | valid_acc 0.81373
[ Valid | 260/300 ] loss = 0.98445, acc = 0.81373 -> best
Best model found at epoch 259, saving model

Summit Result:  0.84760


## eff_b4_color_tfm + TTA 
python train.py \
--batch_size 64 \
--n_epochs 300 \
--patience 300 \
--input_size 224 \
--learning_rate 0.0003 \
--ckpt eff_b4_best.ckpt

Epoch(286/300) train_loss 0.06327 | train_acc 0.97812 | valid_loss 0.85502 | valid_acc 0.82860
[ Valid | 286/300 ] loss = 0.85502, acc = 0.82860 -> best
Best model found at epoch 285, saving model

Summit Result:  0.91035

