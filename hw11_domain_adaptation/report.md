
# Simple Basline (0.5 + 0.5 pts, acc≥0.44616, < 1hour)
    * Just run the code and submit answer.
# Medium Baseline (0.5 + 0.5 pts, acc≥0.64576, 2~4 hours)
* Set proper λ in DaNN algorithm.
* Luck, Training more epochs.
# Strong Baseline (0.5 +0.5  pts, acc≥0.75840, 5~6 hours)
* The Test data is label-balanced, can you make use of this additional 
information?
* Luck, Trail & Error :)
# Boss Baseline (0.5 + 0.5 pts, acc ≥0.80640)
* All the techniques you’ve learned in CNN.
* Change optimizer, learning rate, set lr_scheduler, etc...
* Ensemble the model or output you tried.
* Implement other advanced adversarial training. 
* For example, MCD MSDA DIRT-T
* Huh, semi-supervised learning may help, isn’t it? 
* What about unsupervised learning? (like Universal Domain Adaptation?)


## Vanilla model , epoch=200, lamb = 0.1
epoch 199: train D loss: 0.3737, train F loss: 0.0108, acc 0.9838
Score: 0.51348

## Vanilla model , epoch=200, lamb = 0.31
epoch 200: train D loss: 0.5929, train F loss: -0.0966, acc 0.9774
Score: 0.60451

## Vanilla model, epoch =200, lamb = 1.0
epoch 200: train D loss: 0.6559, train F loss: -0.5551, acc 0.9720
Score: 0.63878

## Vanilla model, epoch =200, lamb = 6.0
epoch 200: train D loss: 0.6883, train F loss: -3.9856, acc 0.9604
Score: 0.53598

## YC's model 
epoch 200: train D loss: 0.6663, train F loss: 0.4299, acc 0.6248
Score: 0.48446

## Vanilla, epoch=1500
epoch 1500: train D loss: 0.6701, train F loss: -0.6294, acc 0.9892
Score: 0.68450


## Adaptive Lamb, epoch = 200, adaptive lamb
epoch 200: train D loss: 0.6641, train F loss: -0.5838, acc 0.9776
Score: 0.67958

Balanced (top-5), score: 0.67526

## 


