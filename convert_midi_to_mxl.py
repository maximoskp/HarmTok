import os
from tqdm import tqdm
from music21 import converter, stream, chord, harmony, metadata, meter, tempo, key, clef, pitch

'''Script to convert midis to mxl using music21'''

flat_dict = {'C#': 'D-',
              'D#': 'E-',
              'F#': 'G-',
              'G#': 'A-',
              'A#': 'B-'}

sharp_dict = {'D-': 'C#',
             'E-': 'D#',
             'G-': 'F#',
             'A-': 'G#',
             'B-': 'A#'}

def fix_accidentals(e, scale):
    
    if scale == 'major':
        setAccidental = 'sharp'
        acc_dict = sharp_dict
    else:
        setAccidental = 'flat'
        acc_dict = flat_dict
        
    new_pitches = []
    
    for p in e.pitches:
        try:
            if p.accidental.name != setAccidental:    
                #create new pitch with conversion
                new_p = pitch.Pitch(acc_dict[p.name])
                new_pitches.append(new_p)
            else:
                new_pitches.append(p)
        except AttributeError:
            new_pitches.append(p)
            continue
        
    newChord = chord.Chord(new_pitches)
    
    return newChord


def check_chord_validation(chord_symbol, scale, e):
    
    #check if it is undedefined chord
    if 'Chord' in chord_symbol:
        #refix chord
        new_e = fix_accidentals(e, scale)  
        #try again
        chord_symbol = harmony.chordSymbolFigureFromChord(new_e)
        if 'Chord' in chord_symbol:
            #try the opposite
            if scale == 'major':
                new_scale = 'minor'
            else:
                new_scale = 'major'
            new_ee = fix_accidentals(e, new_scale)  
            #try last one
            chord_symbol = harmony.chordSymbolFigureFromChord(new_ee)

        
    return chord_symbol


# Subclass ChordSymbol to force valid inversions
class FixedChordSymbol(harmony.ChordSymbol):
    def __init__(self, figure=None, inversion=0, **kwargs):
        super().__init__(figure, **kwargs)
        self.forcedInversion = inversion  # Store valid inversion

    def toXml(self):
        # Call the original toXml method
        xmlElement = super().toXml()
        
        # Ensure valid inversion in the MusicXML output
        inversionElements = xmlElement.findall('.//inversion')
        for inv in inversionElements:
            inv.text = str(max(0, self.forcedInversion))  # Ensure non-negative inversion
        
        return xmlElement


def process_midi_file(midi_file, output_file, band_name, song_name, part_name):
    try:
        # Parse the MIDI file
        midi = converter.parse(midi_file)
        
        # Detect Time Signature
        time_signatures = midi.getTimeSignatures()
        time_signature = time_signatures[0] if time_signatures else meter.TimeSignature('4/4')
        
        # Detect Key Signature
        akey = midi.analyze('key')
        key_signature = key.KeySignature(akey.sharps)  # Convert key analysis to KeySignature object
        
        # Detect Tempo
        tempos = midi.flat.getElementsByClass(tempo.MetronomeMark)
        tempo_mark = tempos[0] if tempos else tempo.MetronomeMark(number=120)  # Default tempo: 120 bpm
        
        # Separate melody and chords tracks
        melody_part = midi.parts[0]  # Assume the first track is melody
        chords_part = midi.parts[1]  # Assume the second track is chords
        
        # Create a new score
        lead_sheet = stream.Score()
        
        # Add metadata
        lead_sheet.metadata = metadata.Metadata()
        lead_sheet.metadata.title = f"{song_name} {part_name}"  # Replace '-' with ' '
        lead_sheet.metadata.composer = band_name.replace('-', ' ')  # Replace '-' with ' '
        
        # Create a Part for the lead sheet
        melody_with_chords = stream.Part()
        
        # Insert time signature, key signature, and tempo at the beginning
        melody_with_chords.append(clef.TrebleClef())  # Add clef
        melody_with_chords.insert(0, time_signature)
        melody_with_chords.insert(0, key_signature)
        melody_with_chords.insert(0, tempo_mark)
        
        # Add melody notes
        for element in melody_part.flat.notesAndRests:
            melody_with_chords.append(element)
        
        # Add chord symbols as harmony objects
        for element in chords_part.flat.notesAndRests:
            if isinstance(element, chord.Chord):
                # Generate chord symbol from the chord
                chord_symbol_text = harmony.chordSymbolFigureFromChord(element)
                # Fix undendified chords
                chord_symbol_text = check_chord_validation(chord_symbol_text, akey.mode, element)
        
                # Calculate inversion
                bass_pitch = element.bass()  # Get the bass note
                inversion = 0  # Default to root position
                for idx, p in enumerate(element.pitches):
                    if p == bass_pitch:
                        inversion = idx
                        break
                # Create FixedChordSymbol and set valid inversion
                chord_symbol = FixedChordSymbol(figure=chord_symbol_text, inversion=max(0, inversion))
                chord_symbol.offset = element.offset  # Align with melody
                melody_with_chords.insert(chord_symbol.offset, chord_symbol)
        
        # Group into measures for better formatting
        formatted_part = melody_with_chords.makeMeasures()
        
        # Add the part to the lead sheet
        lead_sheet.append(formatted_part)
        
        # Check if the score is well-formed
        if not lead_sheet.isWellFormedNotation():
            print("The score is not well-formed. Check the structure.")
            print(lead_sheet.show('text'))  # Debugging: Show the textual representation
        
        # Save as MusicXML
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists
        lead_sheet.write("musicxml", fp=output_file)
        return True  # Successful conversion
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return False  # Failed conversion    


def process_folder(base_folder):
    total_conversions = 0  # Counter for successful conversions
    midi_files = []  # List of MIDI file paths for tqdm
    output_base_folder = os.path.join(os.path.dirname(base_folder), "xmls")  # Create "xmls" folder next to "midis"

    # Collect all MIDI files and their paths
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append((root, file))
    
    # Process each file
    for root, file in tqdm(midi_files, desc="Processing MIDI files"):
        relative_path = os.path.relpath(root, base_folder)
        parts = relative_path.split(os.sep)
        if len(parts) < 3:
            print(f"Skipping {os.path.join(root, file)}: Invalid folder structure.")
            continue
        _, band_name, song_name = parts
        part_name = os.path.splitext(file)[0].replace('-', ' ')
        
        # Define input/output paths
        midi_file = os.path.join(root, file)
        output_file = os.path.join(output_base_folder, relative_path, file.replace('.mid', '.xml'))
        
        # Process the MIDI file
        if process_midi_file(midi_file, output_file, band_name, song_name.replace('-', ' '), part_name):
            total_conversions += 1

    print(f"Total conversions: {total_conversions}")


base_folder = "./hooktheory_dataset/midis"  # Replace with the actual path to your "midis" folder
process_folder(base_folder)