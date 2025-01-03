from data_utils import MergedMelHarmDataset, MLMCollator
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaForMaskedLM
import torch
from torch.optim import AdamW
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
    parser = argparse.ArgumentParser(description='Script for MLM training a tiny RoBERTa model with a specific harmonic tokenizer.')

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

    train_dataset = MergedMelHarmDataset(train_dir, tokenizer, max_length=2048)
    val_dataset = MergedMelHarmDataset(val_dir, tokenizer, max_length=2048)
    collator = MLMCollator(tokenizer)

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

    model_config = RobertaConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        mask_token_id=tokenizer.vocab[tokenizer.mask_token],
        max_position_embeddings=2048,
    )

    model = RobertaForMaskedLM(model_config)
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

    # save results
    os.makedirs('results/mlm', exist_ok=True)
    results_path = 'results/mlm/' + tokenizer_name + '.csv'
    result_fields = ['epoch', 'train_loss', 'tran_acc', 'val_loss', 'val_acc', 'sav_version']
    with open( results_path, 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )
    
    # keep best validation loss for saving
    best_val_loss = np.inf
    save_dir = 'saved_models/mlm/' + tokenizer_name + '/'
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = save_dir + tokenizer_name + '.pt'
    saving_version = 0

    # Training loop
    for epoch in range(epochs):  # Number of epochs
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        print('training')
        with tqdm(trainloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update loss
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                running_accuracy += (predictions[mask] == labels[mask]).sum().item()/mask.sum().item()
                train_accuracy = running_accuracy/batch_num
                
                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        print('validation')
        with torch.no_grad():
            with tqdm(valloader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch} | val')
                for batch in tepoch:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    
                    # update loss
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    # accuracy
                    predictions = outputs.logits.argmax(dim=-1)
                    mask = labels != -100
                    running_accuracy += (predictions[mask] == labels[mask]).sum().item()/mask.sum().item()
                    val_accuracy = running_accuracy/batch_num
                    
                    tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
        if best_val_loss > val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
            print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, train_accuracy, val_loss, val_accuracy, saving_version] )
# end main

if __name__ == '__main__':
    main()