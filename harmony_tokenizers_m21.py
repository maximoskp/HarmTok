# https://huggingface.co/docs/transformers/v4.47.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizer
from tqdm import tqdm 
from transformers import PreTrainedTokenizer
from music21 import converter, harmony, pitch, note, interval
import mir_eval
from copy import deepcopy
import numpy as np
from GCT_functions import get_singe_GCT_of_chord as gct

INT_TO_ROOT_SHARP = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}

MIR_QUALITIES = mir_eval.chord.QUALITIES
EXT_MIR_QUALITIES = deepcopy( MIR_QUALITIES )
for k in list(MIR_QUALITIES.keys()) + ['7(b9)', '7(#9)', '7(#11)', '7(b13)']:
    _, semitone_bitmap, _ = mir_eval.chord.encode( 'C' + (len(k) > 0)*':' + k, reduce_extended_chords=True )
    EXT_MIR_QUALITIES[k] = semitone_bitmap

class HarmonyTokenizerBase(PreTrainedTokenizer):
    def __init__(self):
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.empty_chord = '<emp>'
        self.csl_token = '<s>'
        self.mask_token = '<mask>'
        self.special_tokens = {}
        self.start_harmony_token = '<h>'
        self._added_tokens_encoder = {} # TODO: allow for special tokens
        self.vocab = {
            '<unk>': 0,
            '<pad>': 1,
            '<s>': 2,
            '</s>': 3,
            '<emp>': 4,
            '<mask>': 5,
            '<bar>': 6,
            '<h>': 7
        }
        self.time_quantization = []  # Store predefined quantized times

        # Predefine time quantization tokens for a single measure 1/16th triplets
        max_quarters = 10  # Support up to 10/4 time signatures
        subdivisions = [0, 0.16, 0.25, 0.33, 0.5, 0.66, 0.75, 0.83]
        for quarter in range(max_quarters):
            for subdivision in subdivisions:
                quant_time = round(quarter + subdivision, 3)
                self.time_quantization.append(quant_time)  # Save for later reference
                # Format time tokens with two-digit subdivisions
                quarter_part = int(quant_time)
                subdivision_part = int(round((quant_time - quarter_part) * 100))
                time_token = f'position_{quarter_part}x{subdivision_part:02}'
                self.vocab[time_token] = len(self.vocab)
    # end init

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self._added_tokens_encoder.get(tokens, self.unk_token_id))
        return [self.vocab[token] for token in tokens]
    # end convert_tokens_to_ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens[i] for i in ids]
    # end convert_ids_to_tokens

    def find_closest_quantized_time(self, offset):
        # Find the closest predefined quantized time
        closest_time = min(self.time_quantization, key=lambda t: abs(t - offset))
        quarter = int(closest_time)
        subdivision = int(round((closest_time - quarter) * 100))  # Convert to two-digit integer
        return f'position_{quarter}x{subdivision:02}'  # Format subdivision as two digits
    # end find_closest_quantized_time
    
    def normalize_root_to_sharps(self, root):
        """
        Normalize chord roots to sharp notation, handling special cases like '-' for sharps.
        """
        # Custom mapping for cases like "D-" → "C#"
        special_mapping = {
            'C-': 'B',
            'D-': 'C#',
            'E-': 'D#',
            'F-': 'E',
            'E#': 'F',
            'G-': 'F#',
            'A-': 'G#',
            'B-': 'A#',
            'B#': 'C'
        }

        # Check if the root matches a special case
        if root in special_mapping:
            return special_mapping[root]

        # Use music21 to normalize root to sharp notation otherwise
        pitch_obj = pitch.Pitch(root)
        return pitch_obj.name  # Always return the sharp representation
    # end normalize_root_to_sharps

    def get_closest_mir_eval_symbol(self, chord_symbol):
        # get binary type representation
        # transpose to c major
        ti = interval.Interval( chord_symbol.root(), pitch.Pitch('C') )
        tc = chord_symbol.transpose(ti)
        # make binary
        b = np.zeros(12)
        b[tc.pitchClasses] = 1
        similarity_max = -1
        key_max = '<unk>'
        for k in EXT_MIR_QUALITIES.keys():
            tmp_similarity = np.sum(b == EXT_MIR_QUALITIES[k])
            if similarity_max < tmp_similarity:
                similarity_max = tmp_similarity
                key_max = k
        return key_max
    # end get_closest_mir_eval_symbol

    def normalize_chord_symbol(self, chord_symbol):
        """
        Normalize a music21 chord symbol to match the predefined vocabulary.
        """
        # Normalize root to sharp notation
        root = self.normalize_root_to_sharps(chord_symbol.root().name)  # E.g., "Db" → "C#"
        quality = self.get_closest_mir_eval_symbol( chord_symbol )

        # Return the normalized chord symbol
        return f"{root}", f"{quality}"
    # end normalize_chord_symbol

    def fit(self, corpus):
        raise NotImplementedError()
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        raise NotImplementedError()
    # end fit

    def fit_transform(self, corpus, add_start_harmony_token=True):
        self.fit(corpus)
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class HarmonyTokenizerBase

class MergedMelHarmTokenizer(PreTrainedTokenizer):
    def __init__(self, mel_tokenizer, harm_tokenizer, verbose=0):
        self.mel_tokenizer = mel_tokenizer
        self.harm_tokenizer = harm_tokenizer
        self.verbose = verbose
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.empty_chord = '<emp>'
        self.csl_token = '<s>'
        self.mask_token = '<mask>'
        self.special_tokens = {}
        self._added_tokens_encoder = {} # TODO: allow for special tokens
        # merge vocabularies - start with mel_tokinzer
        self.vocab = mel_tokenizer.vocab
        # add harm_tokenizer on top of that
        if self.verbose > 0:
            print('Merging harmony vocab')
        self.merge_dict_to_vocab( harm_tokenizer.vocab )
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self._added_tokens_encoder.get(tokens, self.unk_token_id))
        return [self.vocab[token] for token in tokens]
    # end convert_tokens_to_ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens[i] for i in ids]
    # end convert_ids_to_tokens

    def merge_dict_to_vocab(self, d):
        for k in d.keys():
            if k not in self.vocab.keys():
                self.vocab[k] = len(self.vocab)
    # end merge_dict_to_vocab

    def fit(self, corpus):
        if self.verbose > 0:
            print('Training melody tokenizer')
        self.mel_tokenizer.fit(corpus)
        if self.verbose > 0:
            print('Merging melody vocab')
        self.merge_dict_to_vocab(self.mel_tokenizer.vocab)
        if self.verbose > 0:
            print('Training harmony tokenizer')
        self.harm_tokenizer.fit(corpus)
        if self.verbose > 0:
            print('Merging harmony vocab')
        self.merge_dict_to_vocab(self.harm_tokenizer.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        # first put melody tokens
        if self.verbose > 0:
            print('Processing melody') #TODO Need proper nested if/else
        mel_toks_ids = self.mel_tokenizer.transform(corpus)
        melody_tokens = mel_toks_ids['tokens'] 
        melody_ids = mel_toks_ids['ids'] 
        # then concatenate harmony tokens
        if self.verbose > 0:
            print('Processing harmony')

        harm_toks_ids = self.harm_tokenizer.transform(corpus, add_start_harmony_token=add_start_harmony_token)
        harmony_tokens = harm_toks_ids['tokens']  
        harmony_ids = harm_toks_ids['ids']   

        # Combine melody and harmony tokens for each file
        combined_tokens = []
        combined_ids = []

        for mel_tok, harm_tok, mel_id, harm_id in zip(melody_tokens, harmony_tokens, melody_ids, harmony_ids):
            combined_tokens.append(mel_tok + harm_tok) 
            combined_ids.append(mel_id + harm_id)    

        return {'tokens': combined_tokens, 'ids': combined_ids}
    # end transform
    

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    # end fit_transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__
# end class MergedMelHarmTokenizer

class ChordSymbolTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(ChordSymbolTokenizer, self).__init__(**kwargs)
        # Generate chord tokens dynamically, forcing sharp notation
        chromatic_roots = []
        for i in range(12):
            pitch_obj = pitch.Pitch(i)
            # Convert flat notation to sharp
            if '-' in pitch_obj.name:  # Check for flats
                pitch_obj = pitch_obj.getEnharmonic()  # Convert to sharp
            chromatic_roots.append(pitch_obj.name)  # Use sharp representation

        qualities = list(EXT_MIR_QUALITIES.keys())

        for root in chromatic_roots:
            for quality in qualities:
                    chord_token = f'{root}:{quality}'
                    #print(chord_token)
                    self.vocab[chord_token] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    chord_token = root_token + ':' + type_token
                    if chord_token in self.vocab:
                        harmony_tokens.append(chord_token)
                        harmony_ids.append(self.vocab[chord_token])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1  

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform
# end class ChordSymbolTokenizer

class RootTypeTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(RootTypeTokenizer, self).__init__(**kwargs)
        # Generate chord tokens dynamically, forcing sharp notation
        chromatic_roots = []
        for i in range(12):
            pitch_obj = pitch.Pitch(i)
            # Convert flat notation to sharp
            if '-' in pitch_obj.name:  # Check for flats
                pitch_obj = pitch_obj.getEnharmonic()  # Convert to sharp
            chromatic_roots.append(pitch_obj.name)  # Use sharp representation

        qualities = list(EXT_MIR_QUALITIES.keys())

        for root in chromatic_roots:
            self.vocab[ root ] = len(self.vocab)
        for quality in qualities:
                quality_token = f'{quality}'
                #print(chord_token)
                self.vocab[quality_token] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def fit(self, corpus):
        pass
    # end fit
    
    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if root_token in self.vocab:
                        harmony_tokens.append(root_token)
                        harmony_ids.append(self.vocab[root_token])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1
                    if type_token in self.vocab:
                        harmony_tokens.append(type_token)
                        harmony_ids.append(self.vocab[type_token])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1  

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__
# end class RootTypeTokenizer

class PitchClassTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(PitchClassTokenizer, self).__init__(**kwargs)
        # chord pitch classes
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        for pc in pcs:
                            tmp_token = 'chord_pc_' + str(pc)
                            harmony_tokens.append( tmp_token )
                            harmony_ids.append(self.vocab[ tmp_token ])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__
# end class PitchClassTokenizer

class RootPCTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(RootPCTokenizer, self).__init__(**kwargs)
        # chord root and pitch classes
        for root in range(12):
            self.vocab['chord_root_' + str(root)] = len(self.vocab)
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        tmp_token = 'chord_root_' + str(root_pc)
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        for pc in pcs:
                            if pc != root_pc:
                                tmp_token = 'chord_pc_' + str(pc)
                                harmony_tokens.append( tmp_token )
                                harmony_ids.append(self.vocab[ tmp_token ])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform
# end class RootPCTokenizer

class GCTRootPCTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(GCTRootPCTokenizer, self).__init__(**kwargs)
        # chord root and pitch classes
        for root in range(12):
            self.vocab['chord_root_' + str(root)] = len(self.vocab)
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        # get gct
                        g = gct( pcs )
                        # get root pc
                        tmp_token = 'chord_root_' + str( g[0] )
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get pitch classes from mir_eval
                        for pc in g[2:]:
                            tmp_token = 'chord_pc_' + str((pc+g[0])%12)
                            harmony_tokens.append( tmp_token )
                            harmony_ids.append(self.vocab[ tmp_token ])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform
# end class GCTRootPCTokenizer

class GCTSymbolTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(GCTSymbolTokenizer, self).__init__(**kwargs)
    # end init

    def fit(self, corpus):
        for file_path in tqdm(corpus, desc="Processing Files"):
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        # get gct
                        g = gct( pcs )
                        tmp_token = str(g)
                        if tmp_token not in self.vocab.keys():
                            self.vocab[tmp_token] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        # get gct
                        g = gct( pcs )
                        tmp_token = str(g)
                        if tmp_token not in self.vocab.keys():
                            tmp_token = '<unk>'
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform
# end class GCTSymbolTokenizer

class GCTRootTypeTokenizer(HarmonyTokenizerBase):
    def __init__(self, **kwargs):
        super(GCTRootTypeTokenizer, self).__init__(**kwargs)
        # chord root and pitch classes
        for root in range(12):
            self.vocab['chord_root_' + str(root)] = len(self.vocab)
    # end init

    def fit(self, corpus):
        for file_path in tqdm(corpus, desc="Processing Files"):
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        # get gct
                        g = gct( pcs )
                        tmp_token = str(g[1:])
                        if tmp_token not in self.vocab.keys():
                            self.vocab[tmp_token] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            if add_start_harmony_token:
                harmony_tokens = [self.start_harmony_token]
                harmony_ids = [self.vocab[self.start_harmony_token]]
            else:
                harmony_tokens = [self.bos_token]
                harmony_ids = [self.vocab[self.bos_token]]

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('<bar>')
                harmony_ids.append(self.vocab['<bar>'])

                # Get all chord symbols within the current measure
                chords_in_measure = [
                    h for h in harmony_stream if measure_offset <= h.offset < measure_offset + quarter_length
                ]

                # If the measure is empty, continue to the next measure
                if not chords_in_measure:
                    continue

                # Process each chord in the current measure
                for h in chords_in_measure:
                    # Quantize time relative to the measure
                    quant_time = h.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    harmony_tokens.append(time_token)
                    harmony_ids.append(self.vocab[time_token])

                    # Normalize and add the chord symbol
                    root_token, type_token = self.normalize_chord_symbol(h)
                    if type_token in EXT_MIR_QUALITIES:
                        root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
                        pcs = (root_pc + np.where(bmap > 0)[0])%12
                        # get gct
                        g = gct( pcs )
                        # get gct root
                        tmp_token = 'chord_root_' + str(g[0])
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get gct type
                        tmp_token = str(g[1:])
                        if tmp_token not in self.vocab.keys():
                            tmp_token = '<unk>'
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('<unk>')
                        harmony_ids.append(self.vocab['<unk>'])
                        unk_count += 1

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(harmony_tokens + [self.eos_token])
            ids.append(harmony_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform
# end class GCTRootTypeTokenizer

class MelodyPitchTokenizer(PreTrainedTokenizer):
    def __init__(self, min_pitch=21, max_pitch=108):
        """
        Initialize the melody tokenizer with a configurable pitch range.
        """
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        self.special_tokens = {}
        self.csl_token = '<s>'
        self.min_pitch = min_pitch  # Minimum MIDI pitch value (e.g., 21 for A0)
        self.max_pitch = max_pitch  # Maximum MIDI pitch value (e.g., 108 for C8)
        self.vocab = {
            '<unk>': 0,
            '<pad>': 1,
            '<s>': 2,
            '</s>': 3,
            '<rest>': 4,
            '<mask>': 5,
            '<bar>': 6
        }
        self.time_quantization = []  # Store predefined quantized times

        # Predefine pitch tokens for the allowed range
        for midi_pitch in range(self.min_pitch, self.max_pitch + 1):
            pitch_token = f'P:{midi_pitch}'
            self.vocab[pitch_token] = len(self.vocab)

        # Predefine time quantization tokens
        max_quarters = 10  # Support up to 10/4 time signatures
        subdivisions = [0, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9]
        for quarter in range(max_quarters):
            for subdivision in subdivisions:
                quant_time = round(quarter + subdivision, 3)
                self.time_quantization.append(quant_time)  # Save for later reference
                # Format time tokens with two-digit subdivisions
                quarter_part = int(quant_time)
                subdivision_part = int(round((quant_time - quarter_part) * 100))
                time_token = f'position_{quarter_part}x{subdivision_part:02}'
                self.vocab[time_token] = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end init

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self._added_tokens_encoder.get(tokens, self.unk_token_id))
        return [self.vocab[token] for token in tokens]
    # end convert_tokens_to_ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens[i] for i in ids]
    # end convert_ids_to_tokens

    def fit(self, corpus):
        pass
    # end fit

    def find_closest_quantized_time(self, offset):
        # Find the closest predefined quantized time
        closest_time = min(self.time_quantization, key=lambda t: abs(t - offset))
        quarter = int(closest_time)
        subdivision = int(round((closest_time - quarter) * 100))  # Convert to two-digit integer
        return f'position_{quarter}x{subdivision:02}'  # Format subdivision as two digits

    def transform(self, corpus):
        """
        Transform a list of MusicXML files into melody tokens and IDs.
        """
        tokens = []
        ids = []

        # Use tqdm to show progress when processing files
        for file_path in tqdm(corpus, desc="Processing Melody Files"):
            unk_count = 0  # Counter to track '<unk>' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume single melody line
            measures = list(part.getElementsByClass('Measure'))
            melody_stream = part.flat.notesAndRests

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            melody_tokens = [self.bos_token]
            melody_ids = [self.vocab[self.bos_token]]

            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                melody_tokens.append('<bar>')
                melody_ids.append(self.vocab['<bar>'])

                # Get all valid events (notes/rests) within the current measure
                events_in_measure = [
                    e for e in melody_stream 
                    if measure_offset <= e.offset < measure_offset + quarter_length 
                    and isinstance(e, (note.Note, note.Rest))
                ]

                # If the measure is empty, add a "Rest" token and continue
                if not events_in_measure:
                    melody_tokens.append('<rest>')
                    melody_ids.append(self.vocab['<rest>'])
                    continue

                # Process each event in the current measure
                for e in events_in_measure:
                    # Quantize time relative to the measure
                    quant_time = e.offset - measure_offset
                    time_token = self.find_closest_quantized_time(quant_time)

                    melody_tokens.append(time_token)
                    melody_ids.append(self.vocab[time_token])

                    # Handle pitch or rest
                    if isinstance(e, note.Note):
                        # Add pitch token if within range
                        midi_pitch = e.pitch.midi
                        if self.min_pitch <= midi_pitch <= self.max_pitch:
                            pitch_token = f'P:{midi_pitch}'
                            melody_tokens.append(pitch_token)
                            melody_ids.append(self.vocab[pitch_token])
                        else:
                            # Out-of-range pitch is treated as '<unk>'
                            melody_tokens.append('<unk>')
                            melody_ids.append(self.vocab['<unk>'])
                            unk_count += 1  

                    elif isinstance(e, note.Rest):
                        # Add rest token
                        melody_tokens.append('<rest>')
                        melody_ids.append(self.vocab['<rest>'])
                    else:
                        # Unknown event type is treated as '<unk>'
                        melody_tokens.append('<unk>')
                        melody_ids.append(self.vocab['<unk>'])
                        unk_count += 1  

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

            tokens.append(melody_tokens + [self.eos_token])
            ids.append(melody_ids + [self.vocab[self.eos_token]])

        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class MelodyPitchTokenizer