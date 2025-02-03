from data_utils import MergedMelHarmDataset, PureGenCollator
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
    parser.add_argument('-s', '--temperature', type=float, help='Temperature for sampling. Defaults to 1.0.', required=False)
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
    temperature = 1.0
    if args.temperature:
        temperature = args.temperature

    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)

    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

    val_dataset = MergedMelHarmDataset(val_dir, tokenizer, max_length=512, return_harmonization_labels=True)
    collator = PureGenCollator(tokenizer)

    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

    model_path = 'saved_models/gpt_reg/' + tokenizer_name + '/' + tokenizer_name + '.pt'

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_positions=512,
        n_layer=8,
        n_head=8,
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        n_embd=512
    )

    model = GPT2LMHeadModel(config)

    if device_name == 'cpu':
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
            checkpoint = torch.load(model_path, weights_only=True)
        else:
            print('Selected device not available: ' + device_name)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    
    model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    output_folder = 'tokenized/gpt_reg_' + str(temperature) + '/'

    os.makedirs(output_folder, exist_ok=True)

    tokenized = {
        'melodies': [],
        'real': [],
        'generated': []
    }
    result_fields = ['melody', 'real', 'generated']
    with open( output_folder + tokenizer_name + '.csv', 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )
    with torch.no_grad():
        with tqdm(valloader, unit='batch') as tepoch:
            tepoch.set_description(f'run')
            for batch in tepoch:
                for b in batch['input_ids']:
                    melody_tokens = []
                    real_tokens = []
                    generated_tokens = []
                    # find the start harmony token
                    start_harmony_position = np.where( b == tokenizer.vocab[tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
                    real_ids = b
                    input_ids = b[:(start_harmony_position+1)].to(device)
                    for i in input_ids:
                        melody_tokens.append( tokenizer.ids_to_tokens[ int(i) ].replace(' ','x') )

                    for i in range(start_harmony_position, len(real_ids), 1):
                        if real_ids[i] != tokenizer.pad_token_id:
                            real_tokens.append( tokenizer.ids_to_tokens[ int(real_ids[i]) ].replace(' ','x') )
                    
                    outputs = model.generate(
                        input_ids=input_ids.reshape(1, input_ids.shape[0]),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=temperature
                    )
                    for i in range(start_harmony_position, len(outputs[0]), 1):
                        generated_tokens.append( tokenizer.ids_to_tokens[ int(outputs[0][i]) ].replace(' ','x') )
                    
                    with open( output_folder + tokenizer_name + '.csv', 'a' ) as f:
                        writer = csv.writer(f)
                        writer.writerow( [' '.join(melody_tokens), ' '.join(real_tokens), ' '.join(generated_tokens)] )

                    tokenized['melodies'].append( melody_tokens )
                    tokenized['real'].append( real_tokens )
                    tokenized['generated'].append( generated_tokens )
    # save all results to csv
    with open(output_folder + tokenizer_name + '.pickle','wb') as handle:
        pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end main

if __name__ == '__main__':
    main()