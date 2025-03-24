from data_utils import SeparatedMelHarmDataset
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq
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
    parser = argparse.ArgumentParser(description='Script for generating token-by-token with GPT2.')

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

    model_path = 'saved_models/bart/' + tokenizer_name + '/' + tokenizer_name + '.pt'

    bart_config = BartConfig(
        vocab_size=len(tokenizer.vocab),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=512,
        encoder_layers=8,
        encoder_attention_heads=16,
        encoder_ffn_dim=512,
        decoder_layers=8,
        decoder_attention_heads=16,
        decoder_ffn_dim=512,
        d_model=512,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        dropout=0.1
    )

    model = BartForConditionalGeneration(bart_config)
    
    val_dataset = SeparatedMelHarmDataset(val_dir, tokenizer, max_length=512, num_bars=8)
    def create_data_collator(tokenizer, model):
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    collator = create_data_collator(tokenizer, model=model)

    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=collator)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)

    checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    model.load_state_dict(checkpoint)

    model.eval()
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

    save_dir = 'tok_by_tok/bart/'
    os.makedirs('tok_by_tok/', exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    result_fields = ['labels', 'predictions']
    with open( save_dir + tokenizer_name + '.csv', 'w' ) as f:
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
                predictions = outputs.logits.argmax(dim=-1)
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
                    with open( save_dir + tokenizer_name + '.csv', 'a' ) as f:
                        writer = csv.writer(f)
                        writer.writerow( [' '.join(tmp_label_toks), ' '.join(tmp_pred_toks)] )
                
                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
    # save all results to csv
    with open(save_dir + tokenizer_name + '.pickle','wb') as handle:
        pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end main

if __name__ == '__main__':
    main()