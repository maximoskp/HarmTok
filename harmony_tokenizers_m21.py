# https://huggingface.co/docs/transformers/v4.47.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from music21 import converter, harmony, pitch, note, interval, stream, meter, chord, duration
import mir_eval
from copy import deepcopy
import numpy as np
from GCT_functions import get_singe_GCT_of_chord as gct
import os
import json
import ast
from copy import deepcopy
import random

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

TOKEN_TO_MUSIC21_QUALITY = {
    'maj': '',
    'min': 'm',
    'dim': 'dim',
    'aug': '+',
    'maj7': 'M7',
    'min7': 'm7',
    '7': '7',
    'dim7': 'dim7',
    'hdim7': 'm7b5',
    'min6': 'm6',
    'maj6': '6',
    '9': '9',
    'min9': 'm9',
    'maj9': 'M9',
    '7(b9)': '7b9',
    '7(#9)': '7#9',
    '7(#11)': '7#11',
    '7(b13)': '7b13',
}

MIR_QUALITIES = mir_eval.chord.QUALITIES
EXT_MIR_QUALITIES = deepcopy( MIR_QUALITIES )
for k in list(MIR_QUALITIES.keys()) + ['7(b9)', '7(#9)', '7(#11)', '7(b13)']:
    _, semitone_bitmap, _ = mir_eval.chord.encode( 'C' + (len(k) > 0)*':' + k, reduce_extended_chords=True )
    EXT_MIR_QUALITIES[k] = semitone_bitmap

class HarmonyTokenizerBase(PreTrainedTokenizer):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.empty_chord = '<emp>'
        self.csl_token = '<s>'
        self.mask_token = '<mask>'
        self.special_tokens = {}
        self.start_harmony_token = '<h>'
        self.construct_basic_vocab()
        if vocab is not None:
            self.vocab = vocab
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self._added_tokens_encoder = {}
        else:
            self.special_tokens = {} # not really needed in this implementation
            self._added_tokens_encoder = {} # TODO: allow for special tokens
    # end init

    def construct_basic_vocab(self):
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
      
        self.update_ids_to_tokens()
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 5
    # end construct_basic_vocab

    def update_ids_to_tokens(self):
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end update_ids_to_tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
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
            'B#': 'C',
            'C##': 'D',
            'D##': 'E',
            'E##': 'F#',
            'F##': 'G',
            'G##': 'A',
            'A##': 'B',
            'B##': 'C#',
            'C--': 'A#',
            'D--': 'C',
            'E--': 'D',
            'F--': 'D#',
            'G--': 'F',
            'A--': 'G',
            'B--': 'A'
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

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        raise NotImplementedError()
    # end handle_chord_symbol

    def decode_chord_symbol(self, harmony_tokens):
        raise NotImplementedError()
    # end handle_chord_symbol

    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []
        for file_path in tqdm(corpus, desc="Processing Files"):
            encoded = self.encode(file_path, add_start_harmony_token=add_start_harmony_token)
            harmony_tokens = encoded['input_tokens']
            harmony_ids = encoded['input_ids']
            tokens.append(harmony_tokens)
            ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def randomize_score(self, score, note_remove_pct=0., chord_remove_pct=0., note_change_pct=0.):
        """
        Modifies a music21 score by:
        1. Removing a percentage of melody notes.
        2. Removing a percentage of harmony chords.
        3. Shifting a percentage of melody notes by a given number of semitones.
        
        Parameters:
        - score (music21.stream.Score): The input score.
        - note_remove_pct (float): Percentage of melody notes to remove.
        - chord_remove_pct (float): Percentage of harmony chords to remove.
        - note_change_pct (float): Percentage of melody notes to shift.
        - shift_semitones (int): Number of semitones to shift the notes.
        
        Returns:
        - Modified music21.stream.Score
        """
        # Get the first part
        part = score.parts[0]
        
        # Separate notes and chord symbols
        notes = [n for n in part.notes if isinstance(n, note.Note)]
        chords = [c for c in part.notes if isinstance(c, harmony.ChordSymbol)]
        
        # Remove random notes
        num_notes_remove = int(len(notes) * note_remove_pct)
        notes_to_remove = random.sample(notes, num_notes_remove)
        
        for note in notes_to_remove:
            part.remove(note)
        
        # Remove random chord symbols
        num_chords_remove = int(len(chords) * chord_remove_pct)
        chords_to_remove = random.sample(chords, num_chords_remove)
        
        for chord in chords_to_remove:
            part.remove(chord)
        
        # Shift random notes by n semitones
        num_notes_change = int(len(notes) * note_change_pct/2)
        notes_to_change = random.sample(notes, num_notes_change)
        
        for note in notes_to_change:
            shift_semitones = np.random.randint(-3,3)
            note.transpose(shift_semitones, inPlace=True)
        
        return score
    # end randomize_score

    def encode(self, file_path, add_start_harmony_token=True, max_length=None, verbose=0, \
            pad_to_max_length=False, padding_side='right', add_eos_token=True, \
            num_bars=None, randomization_rate=0):
        score = converter.parse(file_path)
        if randomization_rate > 0:
            score = self.randomize_score(score, randomization_rate,randomization_rate,randomization_rate)
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
                self.handle_chord_symbol(h, harmony_tokens, harmony_ids)
        
        attention_mask = [1]*len(harmony_ids)

        # Print a message if unknown tokens were generated for the current file
        if verbose > 0 and harmony_tokens.count(self.unk_token) > 0:
            unk_count = harmony_tokens.count(self.unk_token)
            print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")

        if num_bars is not None:
            # get indexes of '<bar>'
            bar_idxs = np.where( np.array(harmony_tokens) == '<bar>' )[0]
            # check if bars number exceed current number of bars
            if bar_idxs.size > num_bars+1:
                bar_idx = bar_idxs[num_bars+1]
                harmony_tokens = harmony_tokens[:bar_idx]
                harmony_ids = harmony_ids[:bar_idx]
                attention_mask = attention_mask[:bar_idx]

        if max_length is not None:
            if max_length > len(harmony_tokens) and pad_to_max_length:
                if padding_side == 'right':
                    harmony_tokens = harmony_tokens + add_eos_token*[self.eos_token] + (max_length-add_eos_token-len(harmony_tokens))*[self.pad_token]
                    harmony_ids = harmony_ids + add_eos_token*[self.vocab[self.eos_token]] + (max_length-add_eos_token-len(harmony_ids))*[self.vocab[self.pad_token]]
                    attention_mask = attention_mask + (max_length-len(attention_mask))*[0]
                else:
                    harmony_tokens =  (max_length-add_eos_token-len(harmony_tokens))*[self.pad_token] + harmony_tokens + add_eos_token*[self.eos_token]
                    harmony_ids = (max_length-add_eos_token-len(harmony_ids))*[self.vocab[self.pad_token]] + harmony_ids + add_eos_token*[self.vocab[self.eos_token]]
                    attention_mask = (max_length-add_eos_token-len(attention_mask))*[0] + attention_mask + add_eos_token*[1]
            else:
                harmony_tokens = harmony_tokens[:max_length]
                harmony_ids = harmony_ids[:max_length]
                attention_mask = [1]*len(harmony_ids)
                if add_eos_token:
                    harmony_tokens[-1] = self.eos_token
                    harmony_ids[-1] = self.vocab[self.eos_token]
        # TODO: return overflowing tokens
        return {
            'input_tokens': harmony_tokens,
            'input_ids': harmony_ids,
            'attention_mask': attention_mask
        }
    # end encode

    def fit_transform(self, corpus, add_start_harmony_token=True):
        self.fit(corpus)
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save special tokens and configuration
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {"special_tokens": self.special_tokens}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    # end save_pretrained

    @classmethod
    def from_pretrained(cls, load_directory):
        # Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load special tokens and configuration
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        special_tokens = config.get("special_tokens", {})
        
        # Create a new tokenizer instance
        return cls(vocab, special_tokens)
    # end from_pretrained

# end class HarmonyTokenizerBase

class MergedMelHarmTokenizer(PreTrainedTokenizer):
    def __init__(self, mel_tokenizer, harm_tokenizer, verbose=0):
        '''
        There is only one way to initialize this tokenizer:
        By providing two tokenizer objects that have been loaded beforehand.
        There is no save_pretrained or load_pretrained.
        '''
        self.melody_tokenizer = mel_tokenizer
        self.harmony_tokenizer = harm_tokenizer
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
        self.vocab = deepcopy(mel_tokenizer.vocab)
        # add harm_tokenizer on top of that
        if self.verbose > 0:
            print('Merging harmony vocab')
        self.merge_and_update_dict_to_vocab( harm_tokenizer.vocab )
        self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 5
    # end init

    def update_ids_to_tokens(self):
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.melody_tokenizer.update_ids_to_tokens()
        self.harmony_tokenizer.update_ids_to_tokens()
    # end update_ids_to_tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[token] for token in tokens]
    # end convert_tokens_to_ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens[i] for i in ids]
    # end convert_ids_to_tokens

    def merge_and_update_dict_to_vocab(self, d):
        for k in d.keys():
            if k not in self.vocab.keys():
                self.vocab[k] = len(self.vocab)
                # update incoming vocab
            d[k] = self.vocab[k]
    # end merge_and_update_dict_to_vocab

    def fit(self, corpus):
        if self.verbose > 0:
            print('Training melody tokenizer')
        self.melody_tokenizer.fit(corpus)
        if self.verbose > 0:
            print('Merging melody vocab')
        self.merge_and_update_dict_to_vocab(self.melody_tokenizer.vocab)
        if self.verbose > 0:
            print('Training harmony tokenizer')
        self.harmony_tokenizer.fit(corpus)
        if self.verbose > 0:
            print('Merging harmony vocab')
        self.merge_and_update_dict_to_vocab(self.harmony_tokenizer.vocab)
        self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end fit

    def encode(self, file_path, add_start_harmony_token=True, max_length=None, verbose=0,\
            pad_to_max_length=False, pad_melody=False, padding_side='right', \
            num_bars=None, randomization_rate=0):
        # first put melody tokens
        if self.verbose > 0:
            print('Processing melody') #TODO Need proper nested if/else
        mel_encoded = self.melody_tokenizer.encode(file_path,\
            max_length=None if max_length is None else max_length//2,\
            pad_to_max_length=pad_melody, num_bars=num_bars)
        melody_tokens = mel_encoded['input_tokens'] 
        melody_ids = mel_encoded['input_ids']
        melody_attention_mask = mel_encoded['attention_mask']
        # then concatenate harmony tokens
        if self.verbose > 0:
            print('Processing harmony')

        harm_encoded = self.harmony_tokenizer.encode(file_path,\
            add_start_harmony_token=add_start_harmony_token,\
            max_length=None if max_length is None else max_length-len(melody_tokens),\
            pad_to_max_length=pad_to_max_length, num_bars=num_bars, \
            randomization_rate=randomization_rate)
        harmony_tokens = harm_encoded['input_tokens']
        harmony_ids = harm_encoded['input_ids']
        harmony_attention_mask = harm_encoded['attention_mask']

        # Combine melody and harmony tokens for each file
        combined_tokens = melody_tokens + harmony_tokens
        combined_ids = melody_ids + harmony_ids
        combined_attention_mask = melody_attention_mask + harmony_attention_mask

        return {
            'input_tokens': combined_tokens,
            'input_ids': combined_ids,
            'attention_mask': combined_attention_mask
        }
    # end encode

    def decode(self, token_sequence, output_format='text', output_path='test.mxl'):

        # Step 1: Strip <pad> tokens and split into melody and harmony parts
        token_sequence = [token for token in token_sequence if token != self.pad_token]  # Remove all <pad> tokens

        try:
            split_index = token_sequence.index('<h>')
        except ValueError:
            raise ValueError("The token sequence does not contain an '<h>' token to separate melody and harmony.")

        melody_tokens = token_sequence[:split_index + 1]
        harmony_tokens = token_sequence[split_index + 1:]  # Exclude the <h> token

        # Step 2: Decode melody
        melody_part = self.melody_tokenizer.decode(melody_tokens)
        melody_for_midi = deepcopy(melody_part)
        # create a part for chords in midi format
        chords_part = stream.Part()
        chords_measure = None
        chords_in_measure = []
        # create a score that will hold both parts
        score = stream.Score()
        measures = melody_part.getElementsByClass('Measure')  # Retrieve all measures from the melody

        # Step 3: Decode harmony and align with measures
        current_measure_index = -1  # Track the measure being processed. -1 to fit the logic with the first bar
        quantized_time = 0  # Track the time position in the current measure

        # Auxilliary function to calculate durations for Chord objects
        def get_current_measure(measures, current_measure_index):
            if current_measure_index < len(measures):
                return measures[current_measure_index]
            else:
                # Default to the last measure if index exceeds melody measures
                return measures[-1]
        
        i = 0
        while i < len(harmony_tokens):
            token = harmony_tokens[i]
            if token in [self.bos_token, '<h>']:
                i += 1
                continue
            elif token == self.eos_token:
                break
            elif token == '<bar>':
                # At measure boundary: finalize previous measure chords
                if chords_measure is not None:
                    current_measure = get_current_measure(measures, current_measure_index)
                    ts = current_measure.getTimeSignatures(returnDefault=True)[0]
                    measure_length = ts.barDuration.quarterLength

                    for idx, (ch_offset, ch_obj) in enumerate(chords_in_measure):
                        if idx < len(chords_in_measure) - 1:
                            next_offset = chords_in_measure[idx + 1][0]
                            chord_duration = next_offset - ch_offset
                        else:
                            chord_duration = measure_length - ch_offset
                        if chord_duration <= 0:
                            print(f"Invalid duration ({chord_duration}), adjusted to minimal duration.")
                            chord_duration = 0.25
                        ch_obj.duration = duration.Duration(chord_duration)
                        chords_measure.insert(ch_offset, ch_obj)

                # Initialize next measure clearly here:
                current_measure_index += 1
                chords_measure = stream.Measure(number=current_measure_index + 1)
                current_measure = get_current_measure(measures, current_measure_index)
                if current_measure.timeSignature is not None:  # if there is a new TS in the melody measure
                    chords_measure.timeSignature = deepcopy(current_measure.timeSignature)

                chords_part.append(chords_measure)
                chords_in_measure = []
                quantized_time = 0
                i += 1
                continue

            elif token.startswith('position_'):
                pos = token.split('_')[1]
                quarter_part, subdivision_part = map(int, pos.split('x'))
                quantized_time = quarter_part + subdivision_part / 100
                i += 1
                continue

            else:
                # Collect chord tokens
                chord_tokens = [token]
                i += 1
                while (i < len(harmony_tokens) and 
                        'bar' not in harmony_tokens[i] and
                        'position' not in harmony_tokens[i] and
                        harmony_tokens[i] != '</s>'):
                    chord_tokens.append(harmony_tokens[i])
                    i += 1

                try:
                    chord_symbol_obj, chord_obj = self.harmony_tokenizer.decode_chord_symbol(chord_tokens)
                except:
                    print(f'cannot decode tokens: {chord_tokens}')

                if chord_symbol_obj and chord_obj:
                    if current_measure_index < len(measures):
                        measure = measures[current_measure_index]
                        chord_symbol_obj.offset = quantized_time
                        measure.insert(quantized_time, chord_symbol_obj)

                    # Store chord offset and object to calculate duration later
                    chords_in_measure.append((quantized_time, chord_obj))

        # AFTER LOOP, handle any remaining chords in the final measure
        if chords_measure is not None and chords_in_measure:
            melody_measure = get_current_measure(measures, current_measure_index)
            if melody_measure.timeSignature is not None: 
                chords_measure.timeSignature = deepcopy(melody_measure.timeSignature)


            ts = melody_measure.getTimeSignatures(returnDefault=True)[0]
            measure_length = ts.barDuration.quarterLength

            for idx, (ch_offset, ch_obj) in enumerate(chords_in_measure):
                if idx < len(chords_in_measure) - 1:
                    next_offset = chords_in_measure[idx + 1][0]
                    chord_duration = next_offset - ch_offset
                else:
                    chord_duration = measure_length - ch_offset
                if chord_duration <= 0:
                    print(f"Invalid duration ({chord_duration}), adjusted to minimal duration.")
                    chord_duration = 0.25
                ch_obj.duration = duration.Duration(chord_duration)
                chords_measure.insert(ch_offset, ch_obj)

        # Assuming chords_part is your music21 stream.Part
        # for measure in chords_part.getElementsByClass('Measure'):
        #    print(f"Measure {measure.measureNumber}:")
        #    for elem in measure.notesAndRests:
        #        print(f"  Offset {elem.offset}: Chord {elem}, Duration {elem.duration.quarterLength}")

        score.insert(0, melody_for_midi)
        score.insert(0, chords_part)
        # Step 4: Display or save the result
        if output_format == 'text':
            # melody_part.show('text')
            score.show('text')
        elif output_format == 'file':
            score.write('musicxml', output_path)
            # melody_part.write('musicxml', output_path)
            print('Saved as', output_path)
    # end decode

    def transform(self, corpus, add_start_harmony_token=True):
        # first put melody tokens
        if self.verbose > 0:
            print('Processing melody') #TODO Need proper nested if/else
        mel_toks_ids = self.melody_tokenizer.transform(corpus)
        melody_tokens = mel_toks_ids['tokens'] 
        melody_ids = mel_toks_ids['ids'] 
        # then concatenate harmony tokens
        if self.verbose > 0:
            print('Processing harmony')

        harm_toks_ids = self.harmony_tokenizer.transform(corpus, add_start_harmony_token=add_start_harmony_token)
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
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(ChordSymbolTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
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
                        chord_token = root + (len(quality) > 0)*':' + quality
                        #print(chord_token)
                        self.vocab[chord_token] = len(self.vocab)
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        # Normalize and add the chord symbol
        root_token, type_token = self.normalize_chord_symbol(h)
        chord_token = root_token + (len(type_token) > 0)*':' + type_token
        if chord_token in self.vocab:
            harmony_tokens.append(chord_token)
            harmony_ids.append(self.vocab[chord_token])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a trivial 1-element list with the token
        token = tokens[0]
        chord_symbol = None
        c = None
        try:
            if token[-1] == ':':
                token = token[:-1]
            r, t, _ = mir_eval.chord.encode( token, reduce_extended_chords=True )
            pcs = r + np.where( t > 0 )[0] + 48
            c = chord.Chord( pcs.tolist() )
            chord_symbol = harmony.chordSymbolFromChord( c )
        except:
            print('unknown chord symbol token: ', token)
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class ChordSymbolTokenizer

class RootTypeTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(RootTypeTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
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
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        root_token, type_token = self.normalize_chord_symbol(h)
        if root_token in self.vocab:
            harmony_tokens.append(root_token)
            harmony_ids.append(self.vocab[root_token])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
        if type_token in self.vocab:
            harmony_tokens.append(type_token)
            harmony_ids.append(self.vocab[type_token])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a 2-element list with the root and quality tokens
        if len(tokens) > 1:
            token = tokens[0] + ':' + tokens[1]
        else:
            token = None
        chord_symbol = None
        c = None
        try:
            r, t, _ = mir_eval.chord.encode( token, reduce_extended_chords=True )
            pcs = r + np.where( t > 0 )[0] + 48
            c = chord.Chord( pcs.tolist() )
            chord_symbol = harmony.chordSymbolFromChord( c )
        except:
            print('unknown chord symbol token: ', token)
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class RootTypeTokenizer

class PitchClassTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(PitchClassTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
            # chord pitch classes
            for pc in range(12):
                self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        # Normalize and add the chord symbol
        root_token, type_token = self.normalize_chord_symbol(h)
        if type_token in EXT_MIR_QUALITIES:
            root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
            pcs = (root_pc + np.where(bmap > 0)[0])%12
            pcs.sort()
            for pc in pcs:
                tmp_token = 'chord_pc_' + str(pc)
                harmony_tokens.append( tmp_token )
                harmony_ids.append(self.vocab[ tmp_token ])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a list of pitch classes
        pcs = [int(pc_token.split('_')[-1]) + 48 for pc_token in tokens ]
        c = None
        chord_symbol = None
        try:
            c = chord.Chord( pcs )
            chord_symbol = harmony.chordSymbolFromChord( c )
        except:
            print(f'pcs not recognized: {pcs}')
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class PitchClassTokenizer

class RootPCTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(RootPCTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
            # chord root and pitch classes
            for root in range(12):
                self.vocab['chord_root_' + str(root)] = len(self.vocab)
            for pc in range(12):
                self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        # Normalize and add the chord symbol
        root_token, type_token = self.normalize_chord_symbol(h)
        if type_token in EXT_MIR_QUALITIES:
            root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
            pcs = (root_pc + np.where(bmap > 0)[0])%12
            tmp_token = 'chord_root_' + str(root_pc)
            harmony_tokens.append( tmp_token )
            harmony_ids.append(self.vocab[ tmp_token ])
            pcs.sort()
            for pc in pcs:
                if pc != root_pc:
                    tmp_token = 'chord_pc_' + str(pc)
                    harmony_tokens.append( tmp_token )
                    harmony_ids.append(self.vocab[ tmp_token ])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a list of pitch classes - we don't care about the root for decoding
        pcs = [int(pc_token.split('_')[-1]) + 48 for pc_token in tokens ]
        c = None
        chord_symbol = None
        try:
            c = chord.Chord( pcs )
            chord_symbol = harmony.chordSymbolFromChord( c )
        except:
            print(f'pcs not recognized: {pcs}')
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class RootPCTokenizer

class GCTRootPCTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(GCTRootPCTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
            # chord root and pitch classes
            for root in range(12):
                self.vocab['chord_root_' + str(root)] = len(self.vocab)
            for pc in range(12):
                self.vocab['chord_pc_' + str(pc)] = len(self.vocab)
            self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
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
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a list of pitch classes - we don't care about the root for decoding
        pcs = [int(pc_token.split('_')[-1]) + 48 for pc_token in tokens ]
        c = chord.Chord( pcs )
        chord_symbol = harmony.chordSymbolFromChord( c )
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class GCTRootPCTokenizer

class GCTSymbolTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(GCTSymbolTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
        # Normalize and add the chord symbol
        root_token, type_token = self.normalize_chord_symbol(h)
        if type_token in EXT_MIR_QUALITIES:
            root_pc, bmap, _ = mir_eval.chord.encode( root_token + (len(type_token) > 0)*':' + type_token, reduce_extended_chords=True )
            pcs = (root_pc + np.where(bmap > 0)[0])%12
            # get gct
            g = gct( pcs )
            tmp_token = str(g)
            if tmp_token not in self.vocab.keys():
                tmp_token = self.unk_token
            harmony_tokens.append( tmp_token )
            harmony_ids.append(self.vocab[ tmp_token ])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

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
        self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end fit

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have a trivial 1-element list with the token
        gct_list = ast.literal_eval(tokens[0])
        pcs = gct_list[0] + np.array( gct_list[1:] ) + 48
        c = chord.Chord( pcs.tolist() )
        chord_symbol = harmony.chordSymbolFromChord( c )
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class GCTSymbolTokenizer

class GCTRootTypeTokenizer(HarmonyTokenizerBase):
    def __init__(self, vocab=None, special_tokens=None, **kwargs):
        super(GCTRootTypeTokenizer, self).__init__(vocab=vocab, special_tokens=special_tokens, **kwargs)
        # if vocab is not None, the vocabulary should be already ok
        if vocab is None:
            # chord root and pitch classes
            for root in range(12):
                self.vocab['chord_root_' + str(root)] = len(self.vocab)
        self.total_vocab_size = len(self.vocab)
    # end init

    def handle_chord_symbol(self, h, harmony_tokens, harmony_ids):
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
                tmp_token = self.unk_token
            harmony_tokens.append( tmp_token )
            harmony_ids.append(self.vocab[ tmp_token ])
        else:
            # Handle unknown chords
            harmony_tokens.append(self.unk_token)
            harmony_ids.append(self.vocab[self.unk_token])
    # end handle_chord_symbol

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
        self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end fit

    def decode_chord_symbol(self, tokens):
        """
        Decode a tokenized chord symbol into a music21.harmony.ChordSymbol object using a predefined mapping.
        """
        # here we should have two tokens, for root and GCT-type
        r = int( tokens[0].split('_')[-1] )
        gct_type = ast.literal_eval(tokens[0])
        pcs = r + np.array( gct_type ) + 48
        c = chord.Chord( pcs.tolist() )
        chord_symbol = harmony.chordSymbolFromChord( c )
        return chord_symbol, c
    # end decode_chord_symbol

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

# end class GCTRootTypeTokenizer

class MelodyPitchTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, special_tokens=None, min_pitch=21, max_pitch=108):
        """
        Initialize the melody tokenizer with a configurable pitch range.
        """
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        self.csl_token = '<s>'
        self.min_pitch = min_pitch  # Minimum MIDI pitch value (e.g., 21 for A0)
        self.max_pitch = max_pitch  # Maximum MIDI pitch value (e.g., 108 for C8)
        self.construct_basic_vocab()
        if vocab is not None:
            self.vocab = vocab
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self._added_tokens_encoder = {}
        else:
            self.special_tokens = {} # not really needed in this implementation
            self._added_tokens_encoder = {}
        self.total_vocab_size = len(self.vocab)
    # end init

    def construct_basic_vocab(self):
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
        self.time_signatures = []  # Store most common time signatures

        # Predefine pitch tokens for the allowed range
        for midi_pitch in range(self.min_pitch, self.max_pitch + 1):
            pitch_token = f'P:{midi_pitch}'
            self.vocab[pitch_token] = len(self.vocab)

        # Predefine time quantization tokens
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
        self.update_ids_to_tokens()
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 5

        # Compute and store most popular time signatures coming from predefined time tokens
        self.time_signatures = self.infer_time_signatures_from_quantization(self.time_quantization, max_quarters)

        # Add time signature tokens to the vocabulary
        for num, denom in self.time_signatures:
            ts_token = f"ts_{num}x{denom}"
            self.vocab[ts_token] = len(self.vocab)
    # end construct_basic_vocab

    def infer_time_signatures_from_quantization(self, time_quantization, max_quarters=10):
        """
        Calculate time signatures based on the quantization scheme. Only x/4 and x/8 are
        included. Removing duplicates like 2/4 and 4/8 keeping the simplest denominator.
        """
        inferred_time_signatures = set()

        for measure_length in range(1, max_quarters + 1):
            # Extract tokens within the current measure
            measure_tokens = [t for t in time_quantization if int(t) < measure_length]

            # Add `x/4` time signatures (number of quarters in the measure)
            inferred_time_signatures.add((measure_length, 4))

            # Validate all valid groupings for `x/8`
            for numerator in range(1, measure_length * 2 + 1):  # Up to 2 eighths per quarter
                eighth_duration = 0.5  # Fixed duration for eighth notes
                valid_onsets = [i * eighth_duration for i in range(numerator)]
                
                # Check if measure_tokens contains a valid subset matching the onsets
                if all(any(abs(t - onset) < 0.01 for t in measure_tokens) for onset in valid_onsets):
                    inferred_time_signatures.add((numerator, 8))
        
        # Remove equivalent time signatures. Separate x/4 and x/8 time signatures
        quarter_signatures = {num for num, denom in inferred_time_signatures if denom == 4}
        cleaned_signatures = [] 
        
        for num, denom in inferred_time_signatures:
            # Keep x/4 time signatures
            if denom == 4:
                cleaned_signatures.append((num, denom))
            # Keep x/8 only if there's no equivalent x/4
            elif denom == 8 and num / 2 not in quarter_signatures:
                cleaned_signatures.append((num, denom))              

        # Return sorted time signatures
        return sorted(cleaned_signatures)
    # end infer_time_signatures_from_quantization

    def update_ids_to_tokens(self):
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end update_ids_to_tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
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

    def encode(self, file_path, max_length=None, verbose=0, pad_to_max_length=False,\
               padding_side='right', num_bars=None):
        unk_count = 0  # Counter to track '<unk>' tokens for the current file
        score = converter.parse(file_path)
        part = score.parts[0]  # Assume single melody line
        measures = list(part.getElementsByClass('Measure'))
        melody_stream = part.flat.notesAndRests

        # Create a mapping of measures to their quarter lengths
        measure_map = {m.offset: (m.quarterLength, m.timeSignature) for m in measures}

        melody_tokens = [self.bos_token]
        melody_ids = [self.vocab[self.bos_token]]

        for measure_offset, (quarter_length, time_signature) in sorted(measure_map.items()):
            # Add a "bar" token for each measure
            melody_tokens.append('<bar>')
            melody_ids.append(self.vocab['<bar>'])

            # Add a time signature token if it exists
            if time_signature:
                ts_token = f"ts_{time_signature.numerator}x{time_signature.denominator}"
                if ts_token in self.vocab:
                    melody_tokens.append(ts_token)
                    melody_ids.append(self.vocab[ts_token])
                else:
                    # Check for equivalent time signature in vocabulary
                    reduced_numerator = time_signature.numerator
                    reduced_denominator = time_signature.denominator
                    while reduced_denominator % 2 == 0 and reduced_numerator % 2 == 0:
                        reduced_numerator //= 2
                        reduced_denominator //= 2
                    equivalent_ts_token = f"ts_{reduced_numerator}x{reduced_denominator}"
                    if equivalent_ts_token in self.vocab:
                        melody_tokens.append(equivalent_ts_token)
                        melody_ids.append(self.vocab[equivalent_ts_token])
                    else:
                        # Default to 4/4 and issue a warning
                        default_ts_token = "ts_4x4"
                        melody_tokens.append(default_ts_token)
                        melody_ids.append(self.vocab[default_ts_token])
                        if verbose > 0:
                            print(f"Warning: Time signature not found in vocab. Defaulting to 4/4.")

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

        attention_mask = [1]*len(melody_ids)

        # Print a message if unknown tokens were generated for the current file
        if verbose > 0 and unk_count > 0:
            print(f"File '{file_path}' generated {unk_count} '<unk>' tokens.")
        
        if num_bars is not None:
            # get indexes of '<bar>'
            bar_idxs = np.where( np.array(melody_tokens) == '<bar>' )[0]
            # check if bars number exceed current number of bars
            if bar_idxs.size > num_bars+1:
                bar_idx = bar_idxs[num_bars+1]
                melody_tokens = melody_tokens[:bar_idx]
                melody_ids = melody_ids[:bar_idx]
                attention_mask = attention_mask[:bar_idx]
        
        if max_length is not None:
            melody_tokens = melody_tokens[:max_length]
            melody_ids = melody_ids[:max_length]
            attention_mask = [1]*len(melody_ids)
            if max_length > len(melody_tokens) and pad_to_max_length:
                if padding_side == 'right':
                    melody_tokens = melody_tokens + (max_length-len(melody_tokens))*[self.pad_token]
                    melody_ids = melody_ids + (max_length-len(melody_ids))*[self.vocab[self.pad_token]]
                    attention_mask = attention_mask + (max_length-len(attention_mask))*[0]
                else:
                    melody_tokens =  (max_length-len(melody_tokens))*[self.pad_token] + melody_tokens
                    melody_ids = (max_length-len(melody_ids))*[self.vocab[self.pad_token]] + melody_ids
                    attention_mask = (max_length-len(attention_mask))*[0] + attention_mask
        # TODO: return overflowing tokens
        return {
            'input_tokens': melody_tokens,
            'input_ids': melody_ids,
            'attention_mask': attention_mask
        }
    # end encode

    def decode(self, tokens):
        """
        Decode a sequence of tokens into a music21 Part for the melody, considering time signatures.
        """
        melody_part = stream.Part()

        current_measure = None
        current_time_signature = None
        bar_length = None

        quantized_time = 0  # Track time position within the measure
        last_position = None  # To calculate durations for notes/rests

        # Track measure numbers
        measure_number = 1

        def finalize_measure(current_measure, bar_length, last_position):
            """
            Finalize the measure by adjusting the duration of the last element
            to fill the remaining time in the bar.
            """
            if current_measure is not None and last_position is not None and len(current_measure.elements) > 0:
                remaining_duration = bar_length - last_position
                if remaining_duration > 0:
                    current_measure[-1].quarterLength = remaining_duration
            return current_measure

        for token in tokens:
            if token == self.bos_token:
                continue
            elif token == '<h>': # harmony token
                # Finalize the last measure and stop
                current_measure = finalize_measure(current_measure, bar_length, last_position)
                if current_measure is not None and len(current_measure.elements) > 0:
                    melody_part.append(current_measure)
                    current_measure = None  # Reset after appending
                break
            elif token == '<bar>':
                # Finalize the current measure and start a new one
                current_measure = finalize_measure(current_measure, bar_length, last_position)
                if current_measure is not None and len(current_measure.elements) > 0:
                    melody_part.append(current_measure)
                    current_measure = None  # Reset after appending
                # Create a new measure instance and assign a measure number
                current_measure = stream.Measure(number=measure_number)
                measure_number += 1
                quantized_time = 0  # Reset time for the new measure
                last_position = None  # Reset last position
                # Add default time signature only if none exists
                if current_time_signature is None:
                    current_time_signature = meter.TimeSignature("4/4")
                    bar_length = current_time_signature.barDuration.quarterLength
                    current_measure.append(current_time_signature)
            elif token.startswith('ts_'):
                # Time signature token, update bar length
                ts_values = token.split('_')[1]
                num, denom = ts_values.split('x')
                new_time_signature = meter.TimeSignature(f"{num}/{denom}")
                bar_length = new_time_signature.barDuration.quarterLength
                if current_measure is None:
                    current_measure = stream.Measure(number=measure_number)
                    measure_number += 1
                # Replace default time signature if it exists
                if current_time_signature != new_time_signature:
                    current_measure.timeSignature = new_time_signature
                current_time_signature = new_time_signature
            elif token.startswith('position_'):
                # Update the quantized time position
                position = token.split('_')[1]
                quarter_part, subdivision_part = map(int, position.split('x'))
                quantized_time = quarter_part + subdivision_part / 100
                if last_position is not None and len(current_measure.elements) > 0:
                    # Calculate the duration for the previous note/rest
                    duration = quantized_time - last_position
                    if duration > 0:
                        current_measure[-1].quarterLength = duration
                last_position = quantized_time
            elif token.startswith('P:'):
                # Create a note based on the pitch
                midi_pitch = int(token.split(':')[1])
                note_obj = note.Note(midi_pitch)
                note_obj.offset = quantized_time
                if current_measure is None:
                    current_measure = stream.Measure(number=measure_number)
                    measure_number += 1
                current_measure.append(note_obj)
            elif token == '<rest>':
                # Create a rest
                rest_obj = note.Rest()
                rest_obj.offset = quantized_time
                if current_measure is None:
                    current_measure = stream.Measure(number=measure_number)
                    measure_number += 1
                current_measure.append(rest_obj)

        # Add the final measure to the part
        if current_measure is not None and len(current_measure.elements) > 0:
            melody_part.append(current_measure)
            current_measure = None  # Reset after appending

        return melody_part
    # end decode

    def transform(self, corpus):
        """
        Transform a list of MusicXML files into melody tokens and IDs.
        """
        tokens = []
        ids = []

        # Use tqdm to show progress when processing files
        for file_path in tqdm(corpus, desc="Processing Melody Files"):
            encoded = self.encode(file_path)
            melody_tokens = encoded['input_tokens']
            melody_ids = encoded['input_ids']
            tokens.append(melody_tokens)
            ids.append(melody_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save special tokens and configuration
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {"special_tokens": self.special_tokens}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    # end save_pretrained

    @classmethod
    def from_pretrained(cls, load_directory):
        # Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load special tokens and configuration
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        special_tokens = config.get("special_tokens", {})
        
        # Create a new tokenizer instance
        return cls(vocab, special_tokens)
    # end from_pretrained

# end class MelodyPitchTokenizer