# HW8 Anomaly Detection
# simple: 0.53150
# median: 0.73171
# Strong : 0.78215
# boss: 0.82154

1. Make a brief introduction about variational autoencoder (VAE). List one 
advantage comparing with vanilla autoencoder and one problem of VAE.
Ans: The representation of VAE is sampled a from a Gaussian distribution. And the distribution is derived from encoder's result. Moreover, VAE also draw an additional sample from a normal distribution and add it to representation. This design create an advantage for VAE which allow user to generate different type of output by assigning the additional sample. On the other hand, the downside of VAE is it tends to generate a vague image after reconstrution.


2. Train a fully connected autoencoder and adjust at least two different 
element of the latent representation. Show your model architecture, plot 
out the original image, the reconstructed images for each adjustment and 
describe the differences





# FCN vanilla
(046/050) Save best model, loss = 0.1399
Score: 0.71554
# 
(049/050) Save best model, loss = 0.0664
Score 0.75729




