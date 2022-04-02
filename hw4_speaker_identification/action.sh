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
bash summit.sh;

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
bash summit.sh;

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
bash summit.sh;
