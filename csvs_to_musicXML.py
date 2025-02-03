from data_utils import MergedMelHarmDataset, PureGenCollator
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
import pandas as pd

# models = ['gpt', 'bart']
models = ['gpt_reg', 'bart_reg']
tokenizer_names = ['ChordSymbolTokenizer', 'RootTypeTokenizer', \
              'PitchClassTokenizer', 'RootPCTokenizer']
# tokenizer_names = ['ChordSymbolTokenizer', 'RootTypeTokenizer', \
#               'PitchClassTokenizer']
tokenizers = {
    'ChordSymbolTokenizer': ChordSymbolTokenizer,
    'RootTypeTokenizer': RootTypeTokenizer,
    'PitchClassTokenizer': PitchClassTokenizer,
    'RootPCTokenizer': RootPCTokenizer,
    'GCTRootPCTokenizer': GCTRootPCTokenizer,
    'GCTSymbolTokenizer': GCTSymbolTokenizer,
    'GCTRootTypeTokenizer': GCTRootTypeTokenizer
}

for model in models:
    for tokenizer_name in tokenizer_names:
        melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
        harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)
        tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)
        c = pd.read_csv( 'tokenized/' + model + '/' + \
                        tokenizer_name + '.csv' )
        mxl_folder = 'musicXMLs/' + model + '/' + tokenizer_name + '/'
        midi_folder = 'MIDIs/' + model + '/' + tokenizer_name + '/'
        os.makedirs(mxl_folder, exist_ok=True)
        os.makedirs(midi_folder, exist_ok=True)
        for i in range(len( c['melody'] )):
            x_real = c['melody'].iloc[i].split() + c['real'].iloc[i].split()
            x_gen = c['melody'].iloc[i].split() + c['generated'].iloc[i].split()
            tokenizer.decode(x_real, output_format='file', \
                output_path = mxl_folder + f'mxl_{i:04}_real.mxl')
            tokenizer.decode(x_gen, output_format='file', \
                output_path = mxl_folder + f'mxl_{i:04}_gen.mxl')
            os.system('mscore -o ' + f'{midi_folder}mid_{i:04}_real.mid' + \
                      ' ' + f'{mxl_folder}mxl_{i:04}_real.mxl')
            os.system('mscore -o ' + f'{midi_folder}mid_{i:04}_gen.mid' + \
                      ' ' + f'{mxl_folder}mxl_{i:04}_gen.mxl')