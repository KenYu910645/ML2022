CUDA_LAUNCH_BLOCKING=1 python train.py \
--n_encoder_layers 6 \
--n_decoder_layers 6 \
--encoder_ffn_embed_dim 2048 \
--decoder_ffn_embed_dim 2048 \
--n_epoch 100 \
--output_fn ffn_2048_summit.txt
> result/ffn_2048.txt;

# python train.py \
# --n_encoder_layers 6 \
# --n_decoder_layers 6 \
# --n_heads 8 \
# --n_epoch 100 \
# --output_fn heads_8_summit.txt \
# > result/heads_8.txt;

# python train.py \
# --n_encoder_layers 6 \
# --n_decoder_layers 6 \
# --n_heads 16 \
# --n_epoch 100 \
# --output_fn heads_16_summit.txt \
# > result/heads_16.txt;

# python train.py \
# --n_encoder_layers 6 \
# --n_decoder_layers 6 \
# --encoder_embed_dim 512 \
# --decoder_embed_dim 512 \
# --n_heads 4 \
# --n_epoch 100 \
# --output_fn embed_512_summit.txt \
# > result/embed_512.txt;

# python train.py \
# --n_encoder_layers 6 \
# --n_decoder_layers 6 \
# --encoder_embed_dim 1024 \
# --decoder_embed_dim 1024 \
# --n_heads 4 \
# --n_epoch 100 \
# --output_fn embed_1024_summit.txt \
# > result/embed_1024.txt;

# python train.py \
# --n_encoder_layers 6 \
# --n_decoder_layers 6 \
# --encoder_ffn_embed_dim 4096 \
# --decoder_ffn_embed_dim 4096 \
# --n_heads 4 \
# --n_epoch 100 \
# --output_fn ffn_4096_summit.txt \
# > result/ffn_4096.txt;
