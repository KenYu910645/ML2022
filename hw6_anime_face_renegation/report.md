## ML_HW6

Simple baseline: FID ≤ 30000, AFD ≥ 0
Medium baseline: FID ≤ 12000, AFD ≥ 0.4
Strong baseline: FID ≤ 10000, AFD ≥ 0.5
Boss baseline: FID ≤ 9000, AFD ≥ 0.6

Simple baseline Use sample code(DCGAN) < 1 hour
Medium baseline Use DCGAN with more epochs 1 ~ 1.5 hours
Strong baseline Use WGAN or WGAN-GP 2 ~ 3 hours
Boss baseline StyleGAN < 5 hours



1. 
Describe the difference between WGAN* and GAN**, list at least two 
differences
1. WGAN applys weight cliping to discriminator's weight when update its parameters everytime. 
2. WGAN's Discriminator don't use Sigmoid layer as its last layer.
3. WGAN don't use momentum-based optimizer(e.g. Adam); instead, WGAN use SGD or RMSProp.
4. WGAN's loss fucntion don't utilize logarithmic function.


2. Please plot the “Gradient norm” result.
a. Use training dataset, set the number of discriminator 
layer to 4 (minimum requirement)
b. Plot two setting:
i. weight clipping
ii. gradient penalty
c. Y-axis: gradient norm(log scale), X-axis: discriminator layer number (from low to high)






