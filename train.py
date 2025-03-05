from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb

import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Implement a greedy decoding strategy: generate one token at a time,
    always pick the highest-probability next token from the prediction of the model.

    The function repeatedly calls model.decode(...) with the current partial sequence,
    appends the predicted token, then stops when reaching `max_len` or the `[EOS]` token.

    This is a simple approach for demonstration or quick checks, in typical machine translation 
    or text generation, doing beam search for higher quality.

    Special tokens:
    - [UNK] (Unknown)
    - [PAD] (Padding)
    - [SOS] (Start of Sequence)
    - [EOS] (End of Sequence)

    """
    # Query the tokenizer for the integer IDs of [SOS] and [EOS], 
    # so we know how to start the sequence and when to stop.
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # 1) Encode the source
    encoder_output = model.encode(source, source_mask)

    # 2) Initialize the decoder input with the [SOS] token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        # Stop if we've hit max_len
        if decoder_input.size(1) == max_len:
            break

        # 3) Build a causal mask for the target tokens so it can't attend to future positions.
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # 4) Decode
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 5) Project the last hidden state to vocabulary logits
        prob = model.project(out[:, -1])

        # 6) Pick the most likely next token with argmax
        _, next_word = torch.max(prob, dim=1)

        # 7) Append the predicted token to 'decoder_input'
        next_token_id = next_word.item()
        next_token_tensor = torch.empty(1, 1).type_as(source).fill_(next_token_id).to(device)
        decoder_input = torch.cat(
            [decoder_input, next_token_tensor], dim=1)

        # 8) Stop if we predicted [EOS]
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    """
    Runs a short validation loop after each epoch to get a quick sense of 
    how the model is doing by printing examples and computing metrics (CER, WER, BLEU).


    """
    model.eval()    # 1) Set model to evaluation mode
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80  # If cannot get the console width, use 80 as default

    with torch.no_grad():   # 2) no gradient calculation
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)   # shape (batch=1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)     # shape (batch=1, 1, 1, seq_len)

            # 3) ensure batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # 4) Decode the output with a gredy approach
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # 5) Get original text and predicted text
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Keep track for metrics
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # 6) Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # 7) Evaluate metrics on all predicted vs expected texts
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})


def get_all_sentences(ds, lang):
    """
    A generator function that yields the raw text of each sentence for one language
    from the dataset
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Check if a tokenizer file already exists. If not, trains a new WordLevel tokenizer on the dataset
    (using the huggingface `tokenizers`)

    Ref: 
        - https://huggingface.co/docs/tokenizers/quicktour
    """
    # Path to a tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    """
    Data preparation pipeline:
     - Load the raw data from HuggingFace's **opus_books** dataset with
    the given language pair (e.g., `en-de`, or `en-it`).
    """
    # 1) Load the raw data from HuggingFace.
    # Since it only has the train split, so we'll split it later.
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # 2) Build (or load) the source and target tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 3) Split the dataset into train / val (90% for train, 10% for val)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # 4) Wrap them inn our custom BilinguaDataset
    train_ds = BilingualDataset(
        train_ds_raw, 
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )
    val_ds = BilingualDataset(
        val_ds_raw, 
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )

    # 5) Find the maximum length of each sentence in the source and target sentence for informational logs
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # 6) Create PyTorch DataLoaders
    # Training uses batch_size from config, random order
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Validation is done one example at a time, so batch_size = 1
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # 7) Return the dataloaders and tokenizers
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    A simple function that calls `build_transformer(...)` with the specified source
    & target vocab sizes, and sequence_length and d_model from the config.

    Returns:
        a fully assembled Transformer Pytorch module.
    """

    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config["seq_len"], 
        config['seq_len'], 
        d_model=config['d_model']
    )
    
    return model


def train_model(config):
    """
    The main function that orchestrates everything.

    """
    # 1) Decide on device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Make sure model folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # 3) Load data & tokenizers
    # Each batch from train_dataloader has items like: 
    #   encoder_input, decoder_input, encoder_maskm decoder_mask, label.
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    # 4) Create model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # 5) Create optimizer, using a standard Adam optimier
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 6) Preload a saved model checkpoint
    initial_epoch = 0
    global_step = 0
    # If preload is set in the config file, we load previous checkpoint (epoch=x.pt)
    # We restore model weights, optimizer states, epoch, global_step to resume training from there.
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    # 7) Define loss function (CrossEntropy + ignoreing PAD + label smoothing)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),    # ignores [PAD] tokens in the target, so they don't affect the loss.
        label_smoothing=0.1
    ).to(device)

    # 8) Setup W&B metrics x-axis
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    # 9) Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()    # free up some GPU memory
        model.train()               # sets the model in training mode

        # tqdm progress bar
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            # a) Move data to device for each tensor in the batch
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (B, seq_len)

            # b) Forward pass
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)     # (B, seq_len, vocab_size)

            # c) Compute loss
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # d) Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # e) Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # f) Step count
            global_step += 1

        # 10) validation at the end of every epoch
        run_validation(
            model, val_dataloader, 
            tokenizer_src, tokenizer_tgt, 
            config['seq_len'], device, 
            lambda msg: batch_iterator.write(msg),
            global_step
        )

        # 11) Save checkpoint at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    wandb.init(
        # set the wandb project where this run will be logged
        project="ai_explained-transformer_from_scratch",
        
        # track hyperparameters and run metadata
        config=config
    )
    
    train_model(config)
