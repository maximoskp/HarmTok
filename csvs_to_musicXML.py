from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
import os
import pandas as pd
import concurrent.futures

tokenizers = {
    'ChordSymbolTokenizer': ChordSymbolTokenizer,
    'RootTypeTokenizer': RootTypeTokenizer,
    'PitchClassTokenizer': PitchClassTokenizer,
    'RootPCTokenizer': RootPCTokenizer,
    'GCTRootPCTokenizer': GCTRootPCTokenizer,
    'GCTSymbolTokenizer': GCTSymbolTokenizer,
    'GCTRootTypeTokenizer': GCTRootTypeTokenizer
}

# tokenized_folders = ['gpt_0.8','gpt_1.0','gpt_1.2',\
#                     'bart_0.8','bart_1.0','bart_1.2',\
#                     'gpt_reg_0.8','gpt_reg_1.0','gpt_reg_1.2',\
#                     'bart_reg_0.8','bart_reg_1.0','bart_reg_1.2']

# tokenized_folders = ['bart_0.8','bart_1.0','bart_1.2',\
#                     'gpt_0.8','gpt_1.0','gpt_1.2',\
#                     'bart_beam_5', 'gpt_beams_5']
tokenized_folders = ['bart_beam_7_temp_0x0', 'gpt_beam_7_temp_0x0']
# tokenized_folders = ['gpt_beam_5']

tokenizer_names = ['ChordSymbolTokenizer', 'PitchClassTokenizer']
# tokenizer_names = ['PitchClassTokenizer']

def process_folder(tok_folder, tokenizer_name):
    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)
    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

    c = pd.read_csv(f'tokenized/{tok_folder}/{tokenizer_name}.csv')
    mxl_folder = f'musicXMLs/{tok_folder}/{tokenizer_name}/'
    midi_folder = f'MIDIs/{tok_folder}/{tokenizer_name}/'

    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    for i in range(len(c['melody'])):
        x_real = c['melody'].iloc[i].split() + c['real'].iloc[i].split()
        x_gen = c['melody'].iloc[i].split() + c['generated'].iloc[i].split()
        try:
            tokenizer.decode(x_real, output_format='file', output_path=f'{mxl_folder}mxl_{i:04}_real.mxl')
            tokenizer.decode(x_gen, output_format='file', output_path=f'{mxl_folder}mxl_{i:04}_gen.mxl')
            # os.system(f'mscore -o {midi_folder}mid_{i:04}_real.mid {mxl_folder}mxl_{i:04}_real.mxl')
            os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_folder}mid_{i:04}_real.mid {mxl_folder}mxl_{i:04}_real.mxl')
            # os.system(f'mscore -o {midi_folder}mid_{i:04}_gen.mid {mxl_folder}mxl_{i:04}_gen.mxl')
            os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_folder}mid_{i:04}_gen.mid {mxl_folder}mxl_{i:04}_gen.mxl')
        except Exception as e:
            print(f'skipping {tok_folder}/{tokenizer_name} at index {i} due to {e}')

# Use ThreadPoolExecutor to parallelize the outer loops
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_folder, tok_folder, tokenizer_name) 
               for tok_folder in tokenized_folders for tokenizer_name in tokenizer_names}

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # To catch any exceptions raised inside the thread
        except Exception as e:
            print(f"Thread encountered an error: {e}")
