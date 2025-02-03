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
        encoder_attention_heads=8,
        encoder_ffn_dim=512,
        decoder_layers=8,
        decoder_attention_heads=8,
        decoder_ffn_dim=512,
        d_model=512,
        encoder_layerdrop=0.3,
        decoder_layerdrop=0.3,
        dropout=0.3
    )

    model = BartForConditionalGeneration(bart_config)

    val_dataset = SeparatedMelHarmDataset(val_dir, tokenizer, max_length=512, num_bars=64)
    def create_data_collator(tokenizer, model):
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    collator = create_data_collator(tokenizer, model=model)

    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

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

    output_folder = 'tokenized/bart_' + str(temperature) + '/'

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
                for bi in range( len(batch['input_ids']) ):
                    melody_tokens = []
                    real_tokens = []
                    generated_tokens = []
                    # find the start harmony token
                    # start_harmony_position = np.where( b == tokenizer.vocab[tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
                    real_ids = batch['labels'][bi]
                    input_ids = batch['input_ids'][bi].to(device)
                    for i in input_ids:
                        melody_tokens.append( tokenizer.ids_to_tokens[ int(i) ].replace(' ','x') )

                    for i in range(0, len(real_ids), 1):
                        if real_ids[i] != tokenizer.pad_token_id and real_ids[i] >= 0:
                            real_tokens.append( tokenizer.ids_to_tokens[ int(real_ids[i]) ].replace(' ','x') )
                    
                    outputs = model.generate(
                        input_ids=input_ids.reshape(1, input_ids.shape[0]),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=temperature
                    )
                    for i in range(1, len(outputs[0]), 1):
                        generated_tokens.append( tokenizer.ids_to_tokens[ int(outputs[0][i]) ].replace(' ','x') )
                    

                    # remove pad from melody tokens
                    melody_tokens = [i for i in melody_tokens if i != tokenizer.pad_token]

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