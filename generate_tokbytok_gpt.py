from data_utils import MergedMelHarmDataset, GenCollator
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel
import torch
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import pickle
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
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 16.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    tokenizer_name = args.tokenizer
    # root_dir = '/media/maindisk/maximos/data/hooktheory_xmls'
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize

    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)

    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

    val_dataset = MergedMelHarmDataset(val_dir, tokenizer, max_length=2048, return_harmonization_labels=True)
    collator = GenCollator(tokenizer)

    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

    model_path = 'saved_models/gen/' + tokenizer_name + '/' + tokenizer_name + '.pt'

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_positions=2048,
        n_layer=4,
        n_head=4,
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        n_embd=256
    )

    model = GPT2LMHeadModel(config)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)

    model.eval()

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    model.to(device)

    val_loss = 0
    running_loss = 0
    batch_num = 0
    running_accuracy = 0
    val_accuracy = 0
    print('validation')
    tokenized = {
        'labels': [],
        'predictions': []
    }
    result_fields = ['labels', 'predictions']
    with open( 'tokenized/gen/' + tokenizer_name + '.csv', 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )
    with torch.no_grad():
        with tqdm(valloader, unit='batch') as tepoch:
            tepoch.set_description(f'run')
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

                for j in range(len( labels )):
                    # create tokenized labels
                    lab_sentence = labels[j]
                    pred_sentence = predictions[j]
                    tmp_label_toks = []
                    tmp_pred_toks = []
                    for i in range(len( lab_sentence )):
                        if lab_sentence[i] > 0:
                            tmp_label_toks.append( tokenizer.ids_to_tokens[ int(lab_sentence[i]) ].replace(' ','x') )
                            tmp_pred_toks.append( tokenizer.ids_to_tokens[ int(pred_sentence[i]) ].replace(' ','x') )
                    tokenized['labels'].append( tmp_label_toks )
                    tokenized['predictions'].append( tmp_pred_toks )
                    with open( 'tokenized/gen/' + tokenizer_name + '.csv', 'a' ) as f:
                        writer = csv.writer(f)
                        writer.writerow( [' '.join(tmp_label_toks), ' '.join(tmp_pred_toks)] )
                
                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
    # save all results to csv
    with open('tokenized/gen/' + tokenizer_name + '.pickle','wb') as handle:
        pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end main

if __name__ == '__main__':
    main()