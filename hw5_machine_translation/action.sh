python train.py \
--n_encoder_layers 6 \
--n_decoder_layers 6 \
--encoder_embed_dim 512 \
--decoder_embed_dim 512 \
--n_heads 8 \
--encoder_ffn_embed_dim 4096 \
--decoder_ffn_embed_dim 4096 \
--n_epoch 100 \
--output_fn base_summit.txt \
> result/base.txt;

