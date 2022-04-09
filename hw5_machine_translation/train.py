import sys
import pprint
import logging
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
# import matplotlib.pyplot as plt
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.models import (
    FairseqEncoder, 
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)
from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)
import shutil
import sacrebleu
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast
from fairseq.modules import MultiheadAttention
from fairseq.models.transformer import base_architecture
import pandas as pd

# * Controls every batch to contain no more than N tokens, which optimizes GPU memory efficiency
# * Shuffles the training set for every epoch
# * Ignore sentences exceeding maximum length
# * Pad all sentences in a batch to the same length, which enables parallel computing by GPU
# * Add eos and shift one token
#     - teacher forcing: to train the model to predict the next token based on prefix, we feed the right shifted target sequence as the decoder input.
#     - generally, prepending bos to the target would do the job (as shown below)
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
#     - in fairseq however, this is done by moving the eos token to the begining. Empirically, this has the same effect. For instance:
#     ```
#     # output target (target) and Decoder input (prev_output_tokens): 
#                    eos = 2
#                 target = 419,  711,  238,  888,  792,   60,  968,    8,    2
#     prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
#     ```
# 
# 

def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond 
        # first call of this method has no effect. 
    )
    return batch_iterator


# * each batch is a python dict, with string key and Tensor value. Contents are described below:
# ```python
# batch = {
#     "id": id, # id for each example 
#     "nsentences": len(samples), # batch size (sentences)
#     "ntokens": ntokens, # batch size (tokens)
#     "net_input": {
#         "src_tokens": src_tokens, # sequence in source language
#         "src_lengths": src_lengths, # sequence length of each example before padding
#         "prev_output_tokens": prev_output_tokens, # right shifted target, as mentioned above.
#     },
#     "target": target, # target sequence
# }
# ```

# # Model Architecture
# * We again inherit fairseq's encoder, decoder and model, so that in the testing phase we can directly leverage fairseq's beam search decoder.
# ## Seq2Seq
# - Composed of **
# Encoder** and **Decoder**
# - Recieves inputs and pass to **Encoder** 
# - Pass the outputs from **Encoder** to **Decoder**
# - **Decoder** will decode according to outputs of previous timesteps as well as **Encoder** outputs  
# - Once done decoding, return the **Decoder** outputs

class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

# # Model Initialization
# # HINT: transformer architecture


def build_model(args, task):
    """ build a model instance based on hyperparameters """
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    
    # encoder decoder
    # HINT: switch to TransformerEncoder & TransformerDecoder
    # encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)
    
    # initialization for seq2seq model is important, requires extra handling
    def init_params(module):
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)
            
    # weight initialization
    model.apply(init_params)
    return model
# ## Architecture Related Configuration
# For strong baseline, please refer to the hyperparameters for *transformer-base* in Table 3 in [Attention is all you need](#vaswani2017)



# HINT: these patches on parameters for Transformer
def add_transformer_args(args):
    args.encoder_attention_heads=config_arg.n_heads
    args.encoder_normalize_before=True
    
    args.decoder_attention_heads=config_arg.n_heads
    args.decoder_normalize_before=True
    
    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024
    
    # patches on default parameters for Transformer (those not set above)
    
    base_architecture(arch_args)

# # Optimization
# ## Loss: Label Smoothing Regularization
# * let the model learn to generate less concentrated distribution, and prevent over-confidence
# * sometimes the ground truth may not be the only answer. thus, when calculating loss, we reserve some probability for incorrect labels
# * avoids overfitting
# 
# code [source](https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html)

class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，the cross-entropy when target is one-hot. following line is same as F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  reserve some probability for other labels. thus when calculating cross-entropy, 
        # equivalent to summing the log probs of all labels
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # when calculating cross-entropy, add the loss of other labels
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
# ## Optimizer: Adam + lr scheduling
# Inverse square root scheduling is important to the stability when training Transformer. It's later used on RNN as well.
# Update the learning rate according to the following equation. Linearly increase the first stage, then decay proportionally to the inverse square root of timestep.
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$

def get_rate(d_model, step_num, warmup_step):
    # lr = 0.00001
    # Change lr from constant to the equation shown above
    lr = d_model**-0.5 * min(step_num**-0.5, step_num*warmup_step**-1.5)
    return lr

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)

# ## Scheduling Visualized
# plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
# plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
# None

# # Training Procedure
# ## Training


def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps) # gradient accumulation: update every accum_steps samples
    
    stats = {"loss": []}
    scaler = GradScaler() # automatic mixed precision (amp) 
    
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=config_arg.device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)            
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                
                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0)) # (sample_size or 1.0) handles the case of a zero gradient
        # TODO Do something about it.

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) # grad norm clipping prevents gradient exploding

        if config_arg.p1:
            gnorm_list.append(gnorm.item())
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    # logger.info(f"training loss: {loss_print:.4f}")
    return stats
# ## Validation & Inference
# To prevent overfitting, validation is required every epoch to validate the performance on unseen data.
# - the procedure is essensially same as training, with the addition of inference step
# - after validation we can save the model weights
# 
# Validation loss alone cannot describe the actual performance of the model
# - Directly produce translation hypotheses based on current model, then calculate BLEU with the reference translation
# - We can also manually examine the hypotheses' quality
# - We use fairseq's sequence generator for beam search to generate translation hypotheses
# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search

def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs

def validate(model, task, criterion, log_to_wandb=True):
    # logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=config_arg.device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # do inference
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    # logger.info("example source: " + srcs[showid])
    # logger.info("example hypothesis: " + hyps[showid])
    # logger.info("example reference: " + refs[showid])
    
    # show bleu results
    # logger.info(f"validation loss:\t{stats['loss']:.4f}")
    # logger.info(stats["bleu"].format())
    return stats

# # Save and Load Model Weights
def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu    
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")

def generate_prediction(model, task, split="test", outfile="./prediction.txt"):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=config_arg.device)

            # do inference
            s, h, r = inference_step(sample, model)
            
            hyps.extend(h)
            idxs.extend(list(sample['id']))
            
    # sort based on the order before preprocess
    hyps = [x for _,x in sorted(zip(idxs,hyps))]
    
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h+"\n")

if __name__ == "__main__":
    IS_TRAIN = True
    N_ITERT_P1 = 1000
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default = "cuda:0")
    parser.add_argument('--n_encoder_layers', type = int, default = 1) # 6
    parser.add_argument('--n_decoder_layers', type = int, default = 1) # 6
    parser.add_argument('--encoder_embed_dim', type = int, default = 256) # 512
    parser.add_argument('--decoder_embed_dim', type = int, default = 256)
    parser.add_argument('--encoder_ffn_embed_dim', type = int, default = 512) # 4096
    parser.add_argument('--decoder_ffn_embed_dim', type = int, default = 1024)
    parser.add_argument('--dropout', type = float, default = 0.3) # 0.1
    parser.add_argument('--n_heads', type = int, default = 4) # 8, 16
    parser.add_argument('--output_fn', type = str, default = "summit.txt")
    parser.add_argument('--n_epoch', type = int, default = 100)
    parser.add_argument('--p1', action="store_true")
    parser.add_argument('--p2', action="store_true")
    config_arg = parser.parse_args()
    arch_args = Namespace(
        encoder_embed_dim=config_arg.encoder_embed_dim,
        encoder_ffn_embed_dim=config_arg.encoder_ffn_embed_dim,
        encoder_layers=config_arg.n_encoder_layers,
        decoder_embed_dim=config_arg.decoder_embed_dim,
        decoder_ffn_embed_dim=config_arg.decoder_ffn_embed_dim,
        decoder_layers=config_arg.n_decoder_layers,
        share_decoder_input_output_embed=True,
        dropout=config_arg.dropout,
    )
    # Set random seed 
    seed = 73
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = Namespace(
        datadir = "./DATA/data-bin/ted2020",
        savedir = "./checkpoints/rnn",
        source_lang = "en",
        target_lang = "zh",
        
        # cpu threads when fetching & processing data.
        num_workers=2,  
        # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
        # max_tokens=8192,
        max_tokens=8192,
        accum_steps=2,
        
        # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
        lr_factor=2.,
        lr_warmup=4000,
        
        # clipping gradient norm helps alleviate gradient exploding
        clip_norm=1.0,
        
        # maximum epochs for training
        max_epoch=config_arg.n_epoch,
        start_epoch=1,
        
        # beam size for beam search
        beam=5,  # 4
        # generate sequences of maximum length ax + b, where x is the source length
        max_len_a=1.2, 
        max_len_b=10, 
        # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
        post_process = "sentencepiece",
        
        # checkpoints
        keep_last_epochs=5,
        resume=None, # if resume from checkpoint name (under config.savedir)
        
        # logging
        use_wandb=False,
    )

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

    # # CUDA Environments
    cuda_env = utils.CudaEnvironment()
    utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])

    ## setup task
    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    # generally, 0.1 is good enough
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    sample = task.dataset("valid")[1]
    pprint.pprint(sample)
    pprint.pprint(
        "Source: " + \
        task.source_dictionary.string(
            sample['source'],
            config.post_process,
        )
    )
    pprint.pprint(
        "Target: " + \
        task.target_dictionary.string(
            sample['target'],
            config.post_process,
        )
    )

    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    sample = next(demo_iter)

    add_transformer_args(arch_args)

    if config.use_wandb:
        wandb.config.update(vars(arch_args))

    model = build_model(arch_args, task)
    logger.info(model)

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim, 
        factor=config.lr_factor, 
        warmup=config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

    # ## Training loop
    sequence_generator = task.build_generator([model], config)

    model = model.to(device=config_arg.device)
    criterion = criterion.to(device=config_arg.device)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    
    if config_arg.p2:
        pos_emb = model.decoder.embed_positions.weights.cpu().detach()
        N = pos_emb.shape[0]
        sim_mat = []
        for i in range(N):
            sim_mat.append(torch.nn.functional.cosine_similarity(pos_emb[i], pos_emb).cpu().detach().numpy())
        with open('p2.pkl', 'wb') as f:
            import pickle
            pickle.dump(sim_mat, f)
        exit()

    # try_load_checkpoint(model, optimizer, name=config.resume)
    gnorm_list = [] # For p1
    if IS_TRAIN:
        while epoch_itr.next_epoch_idx <= config.max_epoch:
            # train for one epoch
            train_stats = train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
            if len(gnorm_list) >= N_ITERT_P1:
                df = pd.DataFrame()
                df['t'] = list(range(0, len(gnorm_list)))
                df['gnorm'] = gnorm_list
                df.to_csv("p1.csv", index=False)
                print("Output gnorm to p1.csv")
                exit()
            stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
            print("({:03d}/{:03d})[Train] loss: {:.6f} [Valid] loss: {:.6f}".format(epoch_itr.epoch, config.max_epoch, np.mean(train_stats['loss']), stats['loss']))
            logger.info(stats["bleu"].format())
            epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

    # # Submission
    # averaging a few checkpoints can have a similar effect to ensemble
    checkdir=config.savedir

    os.system(f"python ./fairseq/scripts/average_checkpoints.py \
    --inputs {checkdir} \
    --num-epoch-checkpoints 5 \
    --output {checkdir}/avg_last_5_checkpoint.pt")
    # ## Confirm model weights used to generate submission
    # checkpoint_last.pt : latest epoch
    # checkpoint_best.pt : highest validation bleu
    # avg_last_5_checkpoint.pt:　the average of last 5 epochs
    try_load_checkpoint(model, name="checkpoint_best.pt") # avg_last_5_checkpoint is weird.......
    # validate(model, task, criterion, log_to_wandb=False)
    generate_prediction(model, task, outfile=config_arg.output_fn)