# 
Simple 14.58 1 hour
Medium 18.04 1 hour 40 mins
Strong 25.20 ~3 hours
Boss 29.13 > 12hours

Simple Baseline: Train a simple RNN seq2seq to acheive translation
Medium Baseline: Add learning rate scheduler and train longer
Strong Baseline: Switch to Transformer and tuning hyperparameter
Boss Baseline: Apply back-translation

● Problem 1 
○ Visualize the similarity between different pairs of positional 
embedding and briefly explain the result. 

● Problem 2
○ Clip gradient norm and visualize the changes of gradient norm in 
different steps. Circle two places with gradient explosion.

## Vanilla 
Summit score: 9.96

## Transformer + lr scheduler
epoch = 100 is OK 
2022-04-03 10:29:02 | INFO | hw5.seq2seq | loaded checkpoint checkpoints/rnn/avg_last_5_checkpoint.pt: step=unknown loss=4.060352325439453 bleu=17.392919789231534
2022-04-03 10:29:02 | INFO | hw5.seq2seq | begin validation
2022-04-03 10:29:29 | INFO | hw5.seq2seq | example source: and in the movie , he plays a downandout lawyer who's become an ambulance chaser .
2022-04-03 10:29:29 | INFO | hw5.seq2seq | example hypothesis: 在電影裡 , 他在電影中扮演了律師 , 他扮演了救護車 。
2022-04-03 10:29:29 | INFO | hw5.seq2seq | example reference: 在這部電影裡 , 保羅紐曼飾演一個落魄的律師一個專攬車禍官司的律師
2022-04-03 10:29:29 | INFO | hw5.seq2seq | validation loss:	4.0499
2022-04-03 10:29:29 | INFO | hw5.seq2seq | BLEU = 17.65 52.0/26.6/14.4/8.2 (BP = 0.877 ratio = 0.884 hyp_len = 98805 ref_len = 111811)

Summit score: 17.55

## Layers
--n_encoder_layers N
--n_decoder_layers N
--n_epoch 30
### layer = 3
(030/030)[Train] loss: 3.709499 [Valid] loss: 3.59595
2022-04-03 21:02:41 | INFO | hw5.seq2seq |  BLEU = 22.38 57.9/31.7/18.1/10.9 (BP = 0.913 ratio = 0.917 hyp_len = 102495 ref_len = 111811)
### layer = 4
(000030/000030)[Train] loss: 3.573526 [Valid] loss: 3.506424
2022-04-03 19:24:44 | INFO | hw5.seq2seq | BLEU = 23.27 58.5/32.5/18.9/11.5 (BP = 0.919 ratio = 0.922 hyp_len = 103087 ref_len = 111811)

### layer = 5
(030/030)[Train] loss: 3.465100 [Valid] loss: 3.435249
2022-04-03 21:02:41 | INFO | hw5.seq2seq | BLEU = 24.15 58.7/32.9/19.4/12.0 (BP = 0.933 ratio = 0.936 hyp_len = 104603 ref_len = 111811)

### layer = 6 
(100/100)[Train] loss: 3.205117 [Valid] loss: 3.295372
2022-04-04 05:58:50 | INFO | hw5.seq2seq | BLEU = 25.49 60.8/35.0/21.0/13.3 (BP = 0.918 ratio = 0.921 hyp_len = 102974 ref_len = 111811)

### layer = 8 
(100/100)[Train] loss: 3.205117 [Valid] loss: 3.295372
2022-04-05 18:53:05 | INFO | hw5.seq2seq | BLEU = 25.49 60.8/35.0/21.0/13.3 (BP = 0.918 ratio = 0.921 hyp_len = 102974 ref_len = 111811)

## ffn
### ffn = 2048
(100/100)[Train] loss: 3.075147 [Valid] loss: 3.240474
2022-04-04 20:35:28 | INFO | hw5.seq2seq | BLEU = 26.03 61.0/35.4/21.4/13.6 (BP = 0.924 ratio = 0.927 hyp_len = 103644 ref_len = 111811)



