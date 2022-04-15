# stylegan2_pytorch --data ./faces \
# --name hello_gan_multigpu \
# --results_dir ./style_gan/results \
# --models_dir ./style_gan/ckpt \
# --network-capacity 32 \
# --multi-gpus \
# --new

# Generate image
stylegan2_pytorch  --generate \
--name hello_gan_multigpu \
--models_dir ./style_gan/ckpt \
--network-capacity 32 \
--multi-gpus
