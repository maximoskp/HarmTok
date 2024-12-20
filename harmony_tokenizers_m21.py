from tqdm import tqdm 
from transformers import PreTrainedTokenizerBase
from music21 import converter, harmony, pitch, note


class ChordSymbolTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        current_token_id = 6
        self.time_quantization = []  # Store predefined quantized times

        # Predefine time quantization tokens for a single measure
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
                self.vocab[time_token] = current_token_id
                current_token_id += 1

        # Generate chord tokens dynamically, forcing sharp notation
        chromatic_roots = []
        for i in range(12):
            pitch_obj = pitch.Pitch(i)
            # Convert flat notation to sharp
            if '-' in pitch_obj.name:  # Check for flats
                pitch_obj = pitch_obj.getEnharmonic()  # Convert to sharp
            chromatic_roots.append(pitch_obj.name)  # Use sharp representation

        qualities = [
            'maj', 'min', 'aug', 'dim', 'sus4', 'sus2', '7', 'maj7', 'min7', 'minmaj7',
            'maj6', 'min6', '6', 'dim7', 'hdim7', 'maj9', 'min9', '9', 'min11', '11', 'maj13',
            'min13', '13', '1', '5'#, ''
        ]

        for root in chromatic_roots:
            for quality in qualities:
                    chord_token = f'{root}:{quality}'
                    #print(chord_token)
                    self.vocab[chord_token] = current_token_id
                    current_token_id += 1

    def fit(self, corpus):
        pass
    # end fit


    def find_closest_quantized_time(self, offset):
        # Find the closest predefined quantized time
        closest_time = min(self.time_quantization, key=lambda t: abs(t - offset))
        quarter = int(closest_time)
        subdivision = int(round((closest_time - quarter) * 100))  # Convert to two-digit integer
        return f'position_{quarter}x{subdivision:02}'  # Format subdivision as two digits
    
    def normalize_root_to_sharps(self, root):
        """
        Normalize chord roots to sharp notation, handling special cases like '-' for sharps.
        """
        # Custom mapping for cases like "D-" → "C#"
        special_mapping = {
            'D-': 'C#',
            'E-': 'D#',
            'G-': 'F#',
            'A-': 'G#',
            'B-': 'A#',
        }

        # Check if the root matches a special case
        if root in special_mapping:
            return special_mapping[root]

        # Use music21 to normalize root to sharp notation otherwise
        pitch_obj = pitch.Pitch(root)
        return pitch_obj.name  # Always return the sharp representation


    def normalize_chord_symbol(self, chord_symbol):
        """
        Normalize a music21 chord symbol to match the predefined vocabulary.
        """
        # Normalize root to sharp notation
        root = self.normalize_root_to_sharps(chord_symbol.root().name)  # E.g., "Db" → "C#"

        # Handle 'sus' chords explicitly
        if 'sus' in chord_symbol.figure:
            # Get intervals of the chord
            intervals = [
                (p.midi - chord_symbol.root().midi) % 12 for p in chord_symbol.pitches
            ]

            # Check for sus2 (M2 interval) or sus4 (P4 interval)
            if 2 in intervals:
                quality = 'sus2'
            elif 5 in intervals:
                quality = 'sus4'
            else:
                quality = 'sus4'  # Default to 'sus4' if unclear
        else:
            # Mapping of music21 qualities to predefined chord qualities
            type_mapping = [
                ('maj7', 'maj7'),
                #('min7', 'min7'),
                ('m7', 'min7'),
                ('7', '7'),
                ('dim7', 'dim7'),
                ('hdim7', 'hdim7'),
                ('minmaj7', 'minmaj7'),
                ('maj6', 'maj6'),
                ('m6', 'min6'),
                ('6', '6'),
                ('maj9', 'maj9'),
                ('M9', 'maj9'),
                ('m9', 'min9'),
                ('power', '5'),
                ('9', '9'),
                ('11', '11'),
                ('maj13', 'maj13'),
                ('m13', 'min13'),
                ('13', '13'),
                ('dim', 'dim'),
                ('aug', 'aug'),
                ('m', 'min'),
                ('', 'maj')  # Default to major
            ]

            # Match figure with type mapping
            for keyword, normalized_type in type_mapping:
                if keyword in chord_symbol.figure:
                    quality = normalized_type
                    break
            else:
                quality = 'unk'  # unknown

        # Return the normalized chord symbol
        return f"{root}:{quality}"

    def transform(self, corpus):
        tokens = []
        ids = []

        for file_path in tqdm(corpus, desc="Processing Files"):
            unk_count = 0  # Counter to track 'unk' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume lead sheet
            measures = list(part.getElementsByClass('Measure'))
            harmony_stream = part.flat.getElementsByClass(harmony.ChordSymbol)

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            harmony_tokens = []
            harmony_ids = []

            # Ensure every measure (even empty ones) generates tokens
            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                harmony_tokens.append('bar')
                harmony_ids.append(self.vocab['bar'])

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
                    chord_token = self.normalize_chord_symbol(h)
                    if chord_token in self.vocab:
                        harmony_tokens.append(chord_token)
                        harmony_ids.append(self.vocab[chord_token])
                    else:
                        # Handle unknown chords
                        harmony_tokens.append('unk')
                        harmony_ids.append(self.vocab['unk'])
                        unk_count += 1  

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} 'unk' tokens.")

            tokens.append(harmony_tokens)
            ids.append(harmony_ids)

        return {'tokens': tokens, 'ids': ids}
    

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class ChordSymbolTokenizer


class MelodyPitchTokenizer(PreTrainedTokenizerBase):
    def __init__(self, min_pitch=21, max_pitch=108):
        """
        Initialize the melody tokenizer with a configurable pitch range.
        """
        self.min_pitch = min_pitch  # Minimum MIDI pitch value (e.g., 21 for A0)
        self.max_pitch = max_pitch  # Maximum MIDI pitch value (e.g., 108 for C8)
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'Rest': 4, 
            'bar': 5
        }
        self.time_quantization = []  # Store predefined quantized times
        current_token_id = 6

        # Predefine pitch tokens for the allowed range
        for midi_pitch in range(self.min_pitch, self.max_pitch + 1):
            pitch_token = f'P:{midi_pitch}'
            self.vocab[pitch_token] = current_token_id
            current_token_id += 1

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
                self.vocab[time_token] = current_token_id
                current_token_id += 1

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
            unk_count = 0  # Counter to track 'unk' tokens for the current file
            score = converter.parse(file_path)
            part = score.parts[0]  # Assume single melody line
            measures = list(part.getElementsByClass('Measure'))
            melody_stream = part.flat.notesAndRests

            # Create a mapping of measures to their quarter lengths
            measure_map = {m.offset: (m.measureNumber, m.quarterLength) for m in measures}

            melody_tokens = []
            melody_ids = []

            for measure_offset, (measure_number, quarter_length) in sorted(measure_map.items()):
                # Add a "bar" token for each measure
                melody_tokens.append('bar')
                melody_ids.append(self.vocab['bar'])

                # Get all valid events (notes/rests) within the current measure
                events_in_measure = [
                    e for e in melody_stream 
                    if measure_offset <= e.offset < measure_offset + quarter_length 
                    and isinstance(e, (note.Note, note.Rest))
                ]

                # If the measure is empty, add a "Rest" token and continue
                if not events_in_measure:
                    melody_tokens.append('Rest')
                    melody_ids.append(self.vocab['Rest'])
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
                            # Out-of-range pitch is treated as 'unk'
                            melody_tokens.append('unk')
                            melody_ids.append(self.vocab['unk'])
                            unk_count += 1  

                    elif isinstance(e, note.Rest):
                        # Add rest token
                        melody_tokens.append('Rest')
                        melody_ids.append(self.vocab['Rest'])
                    else:
                        # Unknown event type is treated as 'unk'
                        melody_tokens.append('unk')
                        melody_ids.append(self.vocab['unk'])
                        unk_count += 1  

            # Print a message if unknown tokens were generated for the current file
            if unk_count > 0:
                print(f"File '{file_path}' generated {unk_count} 'unk' tokens.")

            tokens.append(melody_tokens)
            ids.append(melody_ids)

        return {'tokens': tokens, 'ids': ids}
    

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class MelodyPitchTokenizer