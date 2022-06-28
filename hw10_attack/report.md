### Simple baseline (acc <= 0.70)
* Hints: FGSM
### Medium baseline (acc <= 0.50)
* Hints: Ensemble Attack + random few model + IFGSM
### Strong baseline (acc <= 0.30)
* Hints: 
    1. Ensemble Attack + paper B  (pick right models) + IFGSM  / 
    2. Ensemble Attack + many models + MIFGSM
### Boss baseline (acc <= 0.15)
* Hints: Ensemble Attack + paper B (pick right models) + DIM-MIFGSM 

# IFGSM
Submit score: 	0.720

## MIFGSM
mifgsm_acc = 0.01000, mifgsm_loss = 14.04825
Submit score: 0.600

## Ensemble
model_list = [ptcv_get_model('resnet110_cifar10', pretrained=True).to(device),
              ptcv_get_model('resnext29_32x4d_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet110_cifar10', pretrained=True).to(device),
              ptcv_get_model('pyramidnet164_a270_bn_cifar10', pretrained=True).to(device),
              ptcv_get_model('densenet190_k40_bc_cifar10', pretrained=True).to(device)]

benign_acc = 0.96000, benign_loss = 0.12029
mifgsm_acc = 0.02500, mifgsm_loss = 11.97387

Submit score : 	0.340

## Ensemble pick simple one 
model_list = [ptcv_get_model('resnet56_cifar10', pretrained=True).to(device),
              ptcv_get_model('preresnet20_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet110_cifar10', pretrained=True).to(device),
              ptcv_get_model('resnext29_32x4d_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet20_cifar10', pretrained=True).to(device), 
              ptcv_get_model('densenet40_k12_cifar10', pretrained=True).to(device), 
              ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device)]

benign_acc = 0.95000, benign_loss = 0.11874
mifgsm_acc = 0.01000, mifgsm_loss = 12.13649
Submit score : 0.200

### + DIM (Transform inside batch is all the same) 
model_list same as above

benign_acc = 0.95000, benign_loss = 0.11874
p = 0.5
mifgsm_acc = 0.02000, mifgsm_loss = 11.88624


### 
model_list = [ptcv_get_model('resnet56_cifar10', pretrained=True).to(device),
              ptcv_get_model('preresnet20_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet110_cifar10', pretrained=True).to(device),
              ptcv_get_model('resnext29_32x4d_cifar10', pretrained=True).to(device),
              ptcv_get_model('seresnet20_cifar10', pretrained=True).to(device), 
              ptcv_get_model('densenet40_k12_cifar10', pretrained=True).to(device), 
              ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device), 
              ptcv_get_model('diaresnet20_cifar10', pretrained=True).to(device),
              ptcv_get_model('shakeshakeresnet26_2x32d_cifar10', pretrained=True).to(device), 
              ptcv_get_model('ror3_56_cifar10', pretrained=True).to(device),
              ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device) ]

mifgsm_acc = 0.02500, mifgsm_loss = 10.83858

## Report Question
1. Depending on your best experimental results, briefly explain how you 
generate the transferable noises, and the resulting accuracy on the Judge 
Boi.
I use MI-FGSM as my attacking method and ensemble model as my proxy model. My Ensemble model is consist of seven models which are resnet56, preresnet20, seresnet110, resnext29, seresnet20, densenet40_k12 and wrn16_10. I also found that picking a simpler proxy model with swallow layers could help preformance a lot. Through these methods, I was able to achieve 0.2 on Judge Boi.


