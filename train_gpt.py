from data_utils import MergedMelHarmDataset, GenCollator, compute_normalized_token_entropy
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, get_cosine_schedule_with_warmup
import torch
from torch.optim import AdamW
from torcheval.metrics.text import Perplexity
from tqdm import tqdm
import argparse
import csv

tokenizers = {
    'ChordSymbolTokenizer': ChordSymbolTokenizer,
    'RootTypeTokenizer': RootTypeTokenizer,
    'PitchClassTokenizer': PitchClassTokenizer,
    'RootPCTokenizer': RootPCTokenizer,
    'GCTRootPCTokenizer': GCTRootPCTokenizer,
    'GCTSymbolTokenizer': GCTSymbolTokenizer,
    'GCTRootTypeTokenizer': GCTRootTypeTokenizer
}

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a GPT model with a specific harmonic tokenizer.')

    # Define arguments
    parser.add_argument('-t', '--tokenizer', type=str, help='Specify the tokenizer name among: ' + repr(tokenizers.keys()), required=True)
    parser.add_argument('-d', '--datatrain', type=str, help='Specify the full path to the root folder of the training xml/mxl files', required=True)
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 16.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    tokenizer_name = args.tokenizer
    # root_dir = '/media/maindisk/maximos/data/hooktheory_xmls'
    train_dir = args.datatrain
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 100
    if args.epochs:
        epochs = args.epochs
    lr = 5e-5
    if args.learningrate:
        lr = args.learningrate
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize

    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)

    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

    train_dataset = MergedMelHarmDataset(train_dir, tokenizer, max_length=512, num_bars=64, return_harmonization_labels=True)
    val_dataset = MergedMelHarmDataset(val_dir, tokenizer, max_length=512, num_bars=64, return_harmonization_labels=True)
    collator = GenCollator(tokenizer)

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_positions=512,
        n_layer=8,
        n_head=8, #16,
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        resid_pdrop=0.25, #0.1,
        embd_pdrop=0.25, #0.1,
        attn_pdrop=0.25, #0.1,
        n_embd=512
    )

    model = GPT2LMHeadModel(config)
    model.train()

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    warmup_steps = int(0.02 * total_steps)  # 10% of total steps for warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    perplexity_metric = Perplexity(ignore_index=-100).to(device)

    # save results
    os.makedirs('results/gpt', exist_ok=True)
    results_path = 'results/gpt/' + tokenizer_name + '.csv'
    result_fields = ['epoch', 'step', 'train_loss', 'train_acc', \
                    'train_ppl', 'train_te', 'val_loss', \
                    'val_acc', 'val_ppl', 'val_te', 'sav_version']
    with open( results_path, 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )
    
    # keep best validation loss for saving
    best_val_loss = np.inf
    save_dir = 'saved_models/gpt/' + tokenizer_name + '/'
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = save_dir + tokenizer_name + '.pt'
    saving_version = 0

    def validation_loop(epoch, step, train_loss, train_accuracy, \
                        train_perplexity, train_token_entropy, best_val_loss, saving_version):
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with torch.no_grad():
            with tqdm(valloader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch} | val')
                for batch in tepoch:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # update loss
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    # accuracy
                    predictions = outputs.logits.argmax(dim=-1).roll(shifts=(0,1), dims=(0,1))
                    mask = labels != -100
                    running_accuracy += (predictions[mask] == labels[mask]).sum().item()/mask.sum().item()
                    val_accuracy = running_accuracy/batch_num
                    # perplexity
                    running_perplexity += perplexity_metric.update(outputs.logits, labels.roll(shifts=(0,-1), dims=(0,1))).compute().item()
                    val_perplexity = running_perplexity/batch_num
                    # token entropy
                    _, entropy_per_batch = compute_normalized_token_entropy(outputs.logits, labels.roll(shifts=(0,-1), dims=(0,1)), pad_token_id=-100)
                    running_token_entropy += entropy_per_batch
                    val_token_entropy = running_token_entropy/batch_num
                    
                    tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
        if best_val_loss > val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
            print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, train_loss, train_accuracy, \
                            train_perplexity, train_token_entropy, \
                            val_loss, val_accuracy, \
                            val_perplexity, val_token_entropy, \
                            saving_version] )
        return best_val_loss, saving_version
    # end validation_loop
    step = 0
    # Training loop
    for epoch in range(epochs):  # Number of epochs
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        train_dataset.randomization_rate = 0.2 * ( (epoch/epochs)**0.5 )
        print(f'training with randomization_rate: {train_dataset.randomization_rate}')
        with tqdm(trainloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update loss
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = outputs.logits.argmax(dim=-1).roll(0,1).roll(shifts=(0,1), dims=(0,1))
                mask = labels != -100
                running_accuracy += (predictions[mask] == labels[mask]).sum().item()/mask.sum().item()
                train_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(outputs.logits, labels.roll(shifts=(0,-1), dims=(0,1))).compute().item()
                train_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(outputs.logits, labels.roll(shifts=(0,-1), dims=(0,1)), pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy/batch_num
                
                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step%(total_steps//epochs) == 0 or step == total_steps:# step%(total_steps//100) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_loop(
                        epoch,
                        step,
                        train_loss,
                        train_accuracy,
                        train_perplexity,
                        train_token_entropy,
                        best_val_loss,
                        saving_version
                    )
# end main

if __name__ == '__main__':
    main()