import os
import csv
import zlib
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer

root_dir = '/media/maindisk/maximos/data/hooktheory_xmls/'
data_files = []

# Walk through all subdirectories and files
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.xml') or file.endswith('.mxl'):
            full_path = os.path.join(dirpath, file)
            data_files.append(full_path)

print('Total files from Hook Theory dataset:', len(data_files))

# prepare stats
stats = {}

def compute_compression_rate(array: np.ndarray, compression_method=zlib.compress) -> float:
    """
    Compute the compression rate of a NumPy array.

    Parameters:
        array (np.ndarray): The NumPy array to compress.
        compression_method (callable): The compression method to use. 
                                       Default is `zlib.compress`.

    Returns:
        float: The compression rate (compressed size / original size).
    """
    # Convert the array to bytes
    array_bytes = array.tobytes()
    
    # Compress the byte representation
    compressed_bytes = compression_method(array_bytes)
    
    # Compute sizes
    original_size = len(array_bytes)
    compressed_size = len(compressed_bytes)
    
    # Calculate compression rate
    compression_rate = compressed_size / original_size

    return compression_rate

def initialize_stats(key, tokenizer):
    stats[key] = {
        'vocab_size': len(tokenizer.vocab),
        'seq_lens': [],
        'compression_rates': []
    }
# end initialize_stats

def update_stats(key, toks):
    for t in toks['ids']:
        stats[key]['seq_lens'].append( len(t) )
        stats[key]['compression_rates'].append( compute_compression_rate(np.array(t)) )
    stats[key]['mean_len'] = np.mean(stats[key]['seq_lens'])
    stats[key]['std_len'] = np.std(stats[key]['seq_lens'])
    stats[key]['max_len'] = np.max(stats[key]['seq_lens'])
    stats[key]['mean_compression'] = np.mean(stats[key]['compression_rates'])
    stats[key]['std_compression'] = np.std(stats[key]['compression_rates'])
# end update_stats

def print_stats(key):
    print('vocab_size: ', stats[key]['vocab_size'])
    print('mean len: ', stats[key]['mean_len'])
    print('std len: ', stats[key]['std_len'])
    print('max len: ', stats[key]['max_len'])
    print('mean cr: ', stats[key]['mean_compression'])
    print('std cr: ', stats[key]['std_compression'])

print('ChordSymbolTokenizer_m21')
chordSymbolTokenizer = ChordSymbolTokenizer()
print('len(chordSymbolTokenizer.vocab): ', len(chordSymbolTokenizer.vocab))
initialize_stats('ChordSymbolTokenizer', chordSymbolTokenizer)
toks_cs = chordSymbolTokenizer(data_files)
print('example sentence length: ', len(toks_cs['tokens'][0]))
print(toks_cs['tokens'][0])
print(toks_cs['ids'][0])
update_stats('ChordSymbolTokenizer', toks_cs)
print_stats('ChordSymbolTokenizer')

chordSymbolTokenizer.save_pretrained('saved_tokenizers/ChordSymbolTokenizer')
chordSymbolTokenizer.from_pretrained('saved_tokenizers/ChordSymbolTokenizer')
print(chordSymbolTokenizer.vocab)

print('RootTypeTokenizer')
rootTypeTokenizer = RootTypeTokenizer()
print('len(rootTypeTokenizer.vocab): ', len(rootTypeTokenizer.vocab))
initialize_stats('RootTypeTokenizer', rootTypeTokenizer)
toks_rt = rootTypeTokenizer(data_files)
print('example sentence length: ', len(toks_rt['tokens'][0]))
print(toks_rt['tokens'][0])
print(toks_rt['ids'][0])
update_stats('RootTypeTokenizer', toks_rt)
print_stats('RootTypeTokenizer')

rootTypeTokenizer.save_pretrained('saved_tokenizers/RootTypeTokenizer')
rootTypeTokenizer.from_pretrained('saved_tokenizers/RootTypeTokenizer')
print(rootTypeTokenizer.vocab)

print('PitchClassTokenizer')
pitchClassTokenizer = PitchClassTokenizer()
print('len(pitchClassTokenizer.vocab): ', len(pitchClassTokenizer.vocab))
initialize_stats('PitchClassTokenizer', pitchClassTokenizer)
toks_pc = pitchClassTokenizer(data_files)
print('example sentence length: ', len(toks_pc['tokens'][0]))
print(toks_pc['tokens'][0])
print(toks_pc['ids'][0])
update_stats('PitchClassTokenizer', toks_pc)
print_stats('PitchClassTokenizer')

pitchClassTokenizer.save_pretrained('saved_tokenizers/PitchClassTokenizer')
pitchClassTokenizer.from_pretrained('saved_tokenizers/PitchClassTokenizer')
print(pitchClassTokenizer.vocab)

print('RootPCTokenizer')
rootPCTokenizer = RootPCTokenizer()
print('len(rootPCTokenizer.vocab): ', len(rootPCTokenizer.vocab))
initialize_stats('RootPCTokenizer', rootPCTokenizer)
toks_rpc = rootPCTokenizer(data_files)
print('example sentence length: ', len(toks_rpc['tokens'][0]))
print(toks_rpc['tokens'][0])
print(toks_rpc['ids'][0])
update_stats('RootPCTokenizer', toks_rpc)
print_stats('RootPCTokenizer')

rootPCTokenizer.save_pretrained('saved_tokenizers/RootPCTokenizer')
rootPCTokenizer.from_pretrained('saved_tokenizers/RootPCTokenizer')
print(rootPCTokenizer.vocab)

print('GCTRootPCTokenizer')
gctRootPCTokenizer = GCTRootPCTokenizer()
print('len(gctRootPCTokenizer.vocab): ', len(gctRootPCTokenizer.vocab))
initialize_stats('GCTRootPCTokenizer', gctRootPCTokenizer)
toks_gct_rpc = gctRootPCTokenizer(data_files)
print('example sentence length: ', len(toks_gct_rpc['tokens'][0]))
print(toks_gct_rpc['tokens'][0])
print(toks_gct_rpc['ids'][0])
update_stats('GCTRootPCTokenizer', toks_gct_rpc)
print_stats('GCTRootPCTokenizer')

gctRootPCTokenizer.save_pretrained('saved_tokenizers/GCTRootPCTokenizer')
gctRootPCTokenizer.from_pretrained('saved_tokenizers/GCTRootPCTokenizer')
print(gctRootPCTokenizer.vocab)

print('GCTSymbolTokenizer')
gctSymbolTokenizer = GCTSymbolTokenizer()
print('training')
gctSymbolTokenizer.fit( data_files )
print('len(gctSymbolTokenizer.vocab): ', len(gctSymbolTokenizer.vocab))
initialize_stats('GCTSymbolTokenizer', gctSymbolTokenizer)
toks_gct_symb = gctSymbolTokenizer(data_files)
print('example sentence length: ', len(toks_gct_symb['tokens'][0]))
print(toks_gct_symb['tokens'][0])
print(toks_gct_symb['ids'][0])
update_stats('GCTSymbolTokenizer', toks_gct_symb)
print_stats('GCTSymbolTokenizer')

gctSymbolTokenizer.save_pretrained('saved_tokenizers/GCTSymbolTokenizer')
gctSymbolTokenizer.from_pretrained('saved_tokenizers/GCTSymbolTokenizer')
print(gctSymbolTokenizer.vocab)

print('GCTRootTypeTokenizer')
gctRootTypeTokenizer = GCTRootTypeTokenizer()
print('training')
gctRootTypeTokenizer.fit( data_files )
print('len(gctRootTypeTokenizer.vocab): ', len(gctRootTypeTokenizer.vocab))
initialize_stats('GCTRootTypeTokenizer', gctRootTypeTokenizer)
toks_gct_rt = gctRootTypeTokenizer(data_files)
print('example sentence length: ', len(toks_gct_rt['tokens'][0]))
print(toks_gct_rt['tokens'][0])
print(toks_gct_rt['ids'][0])
update_stats('GCTRootTypeTokenizer', toks_gct_rt)
print_stats('GCTRootTypeTokenizer')

gctRootTypeTokenizer.save_pretrained('saved_tokenizers/GCTRootTypeTokenizer')
gctRootTypeTokenizer.from_pretrained('saved_tokenizers/GCTRootTypeTokenizer')
print(gctRootTypeTokenizer.vocab)

print('MelodyPitchTokenizer_m21')
melodyPitchTokenizer = MelodyPitchTokenizer(min_pitch=21, max_pitch=108) #default range, need to adjust
print('len(melodyPitchTokenizer.vocab): ', len(melodyPitchTokenizer.vocab))
initialize_stats('MelodyPitchTokenizer', melodyPitchTokenizer)
toks_cs = melodyPitchTokenizer(data_files)
print('example sentence length: ', len(toks_cs['tokens'][0]))
print(toks_cs['tokens'][0])
print(toks_cs['ids'][0])
update_stats('MelodyPitchTokenizer', toks_cs)
print_stats('MelodyPitchTokenizer')

melodyPitchTokenizer.save_pretrained('saved_tokenizers/MelodyPitchTokenizer')
melodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
print(melodyPitchTokenizer.vocab)

# print stats
tokenizers = ['ChordSymbolTokenizer', 'GCTSymbolTokenizer', \
    'RootTypeTokenizer', 'GCTRootTypeTokenizer',\
    'RootPCTokenizer', 'GCTRootPCTokenizer', \
    'PitchClassTokenizer', 'MelodyPitchTokenizer'
]

results_path = 'vocab_stats_hk_m21.csv' #for hook theory

result_fields = ['Tokenizer_m21', 'vocab_size'] + list( stats['ChordSymbolTokenizer'].keys() )[3:]

with open( results_path, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow( result_fields )

for tok in tokenizers:
    with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [tok] + [stats[tok]['vocab_size']] + list( stats[tok].values() )[3:] )

chordSymbolTokenizer = ChordSymbolTokenizer.from_pretrained('saved_tokenizers/ChordSymbolTokenizer')
rootTypeTokenizer = RootTypeTokenizer.from_pretrained('saved_tokenizers/RootTypeTokenizer')
pitchClassTokenizer = PitchClassTokenizer.from_pretrained('saved_tokenizers/PitchClassTokenizer')
rootPCTokenizer = RootPCTokenizer.from_pretrained('saved_tokenizers/RootPCTokenizer')
gctRootPCTokenizer = GCTRootPCTokenizer.from_pretrained('saved_tokenizers/GCTRootPCTokenizer')
gctSymbolTokenizer = GCTSymbolTokenizer.from_pretrained('saved_tokenizers/GCTSymbolTokenizer')
gctRootTypeTokenizer = GCTRootTypeTokenizer.from_pretrained('saved_tokenizers/GCTRootTypeTokenizer')
melodyPitchTokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')

m_chordSymbolTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, chordSymbolTokenizer, verbose=1)
m_rootTypeTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, rootTypeTokenizer)
m_pitchClassTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, pitchClassTokenizer)
m_rootPCTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, rootPCTokenizer)
m_gctRootPCTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, gctRootPCTokenizer)
m_gctSymbolTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, gctSymbolTokenizer)
m_gctRootTypeTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, gctRootTypeTokenizer)

print(len(chordSymbolTokenizer.vocab))
print(len(melodyPitchTokenizer.vocab))
print(len(m_chordSymbolTokenizer.vocab))