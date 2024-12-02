import numpy as np
from transformers import PreTrainedTokenizerBase
import mir_eval
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

class ChordSymbolTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        '''
        0: unk
        1: padding
        2: beginning of sequence
        3: end of sequence
        4: empty chord
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        # self.root_offset = 5
        # self.type_offset = self.root_offset + 12
        # self.max_num_segments = max_num_segments
        # self.segment_offset = self.root_offset + 12
        # self.pad_to_length = pad_to_length
        # self.vocab_size = self.segment_offset + max_num_segments
        # self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        # positions
        current_token_id = 6
        # up to 8/4 time signatures
        for quart in range(8):
            for sixteenth in ['0', '5']:
                self.vocab['position_' + str(quart) + 'x' + sixteenth] = current_token_id
                current_token_id += 1
        # chords
        for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
            for quality in mir_eval.chord.QUALITIES.keys():
                self.vocab[ root + ':' + quality ] = current_token_id
                current_token_id += 1
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # corpus is a list of muspy pieces
        tokens =[]
        ids = []
        for piece in corpus:
            resolution = piece.resolution
            barlines = piece.barlines
            for track in piece.tracks:
                if len(track.harmony) > 0:
                    harmony = track.harmony
                    # adjust time in each chord symbol
                    bar_idx = 0
                    bar_time = barlines[bar_idx].time

                    # in parallel, create tokens list
                    harmony_tokens = ['bar']
                    harmony_ids = [self.vocab['bar']]

                    for h in harmony:
                        # check if chord time exceeds next bar time
                        if bar_idx+1 < len(barlines):
                            while h.time >= barlines[bar_idx + 1].time:
                                bar_idx += 1
                                bar_time = barlines[bar_idx].time
                                harmony_tokens.append( 'bar' )
                                harmony_ids.append(self.vocab['bar'])
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # append chord symbol with one of eleven sharp-only roots
                        tmp_root = INT_TO_ROOT_SHARP[h.root_pc]
                        if ':' in h.chord_symbol_mir_eval:
                            tmp_type = h.chord_symbol_mir_eval.split(':')[1]
                        else:
                            tmp_type = ''
                        tmp_token = tmp_root + ':' + tmp_type
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                    tokens.append(harmony_tokens)
                    ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class ChordSymbolTokenizer

class RootTypeTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        '''
        0: unk
        1: padding
        2: beginning of sequence
        3: end of sequence
        4: empty chord
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        # self.root_offset = 5
        # self.type_offset = self.root_offset + 12
        # self.max_num_segments = max_num_segments
        # self.segment_offset = self.root_offset + 12
        # self.pad_to_length = pad_to_length
        # self.vocab_size = self.segment_offset + max_num_segments
        # self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        # positions
        current_token_id = 6
        # up to 8/4 time signatures
        for quart in range(8):
            for sixteenth in ['0', '5']:
                self.vocab['position_' + str(quart) + 'x' + sixteenth] = current_token_id
                current_token_id += 1
        # chords
        for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
            self.vocab[ root ] = current_token_id
            current_token_id += 1
        for quality in mir_eval.chord.QUALITIES.keys():
            self.vocab[ quality ] = current_token_id
            current_token_id += 1
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # corpus is a list of muspy pieces
        tokens =[]
        ids = []
        for piece in corpus:
            resolution = piece.resolution
            barlines = piece.barlines
            for track in piece.tracks:
                if len(track.harmony) > 0:
                    harmony = track.harmony
                    # adjust time in each chord symbol
                    bar_idx = 0
                    bar_time = barlines[bar_idx].time

                    # in parallel, create tokens list
                    harmony_tokens = ['bar']
                    harmony_ids = [self.vocab['bar']]

                    for h in harmony:
                        # check if chord time exceeds next bar time
                        if bar_idx+1 < len(barlines):
                            while h.time >= barlines[bar_idx + 1].time:
                                bar_idx += 1
                                bar_time = barlines[bar_idx].time
                                harmony_tokens.append( 'bar' )
                                harmony_ids.append(self.vocab['bar'])
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # append chord symbol with one of eleven sharp-only roots
                        tmp_root = INT_TO_ROOT_SHARP[h.root_pc]
                        harmony_tokens.append( tmp_root )
                        harmony_ids.append(self.vocab[ tmp_root ])
                        if ':' in h.chord_symbol_mir_eval:
                            tmp_type = h.chord_symbol_mir_eval.split(':')[1]
                        else:
                            tmp_type = ''
                        harmony_tokens.append( tmp_type )
                        harmony_ids.append(self.vocab[ tmp_type ])
                    tokens.append(harmony_tokens)
                    ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class RootTypeTokenizer

class PitchClassTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        '''
        0: unk
        1: padding
        2: beginning of sequence
        3: end of sequence
        4: empty chord
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        # self.root_offset = 5
        # self.type_offset = self.root_offset + 12
        # self.max_num_segments = max_num_segments
        # self.segment_offset = self.root_offset + 12
        # self.pad_to_length = pad_to_length
        # self.vocab_size = self.segment_offset + max_num_segments
        # self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        # positions
        current_token_id = 7
        # up to 8/4 time signatures
        for quart in range(8):
            for sixteenth in ['0', '5']:
                self.vocab['position_' + str(quart) + 'x' + sixteenth] = current_token_id
                current_token_id += 1
        # chord pitch classes
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = current_token_id
            current_token_id += 1
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # corpus is a list of muspy pieces
        tokens =[]
        ids = []
        for piece in corpus:
            resolution = piece.resolution
            barlines = piece.barlines
            for track in piece.tracks:
                if len(track.harmony) > 0:
                    harmony = track.harmony
                    # adjust time in each chord symbol
                    bar_idx = 0
                    bar_time = barlines[bar_idx].time

                    # in parallel, create tokens list
                    harmony_tokens = ['bar']
                    harmony_ids = [self.vocab['bar']]

                    for h in harmony:
                        # check if chord time exceeds next bar time
                        if bar_idx+1 < len(barlines):
                            while h.time >= barlines[bar_idx + 1].time:
                                bar_idx += 1
                                bar_time = barlines[bar_idx].time
                                harmony_tokens.append( 'bar' )
                                harmony_ids.append(self.vocab['bar'])
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get pitch classes from mir_eval
                        pcs = (h.root_pc + np.where(h.binary_mir_eval > 0)[0])%12
                        for pc in pcs:
                            tmp_token = 'chord_pc_' + str(pc)
                            harmony_tokens.append( tmp_token )
                            harmony_ids.append(self.vocab[ tmp_token ])
                    tokens.append(harmony_tokens)
                    ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class PitchClassTokenizer

class RootPCTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        '''
        0: unk
        1: padding
        2: beginning of sequence
        3: end of sequence
        4: empty chord
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        # self.root_offset = 5
        # self.type_offset = self.root_offset + 12
        # self.max_num_segments = max_num_segments
        # self.segment_offset = self.root_offset + 12
        # self.pad_to_length = pad_to_length
        # self.vocab_size = self.segment_offset + max_num_segments
        # self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        # positions
        current_token_id = 7
        # up to 8/4 time signatures
        for quart in range(8):
            for sixteenth in ['0', '5']:
                self.vocab['position_' + str(quart) + 'x' + sixteenth] = current_token_id
                current_token_id += 1
        # chord root and pitch classes
        for root in range(12):
            self.vocab['chord_root_' + str(root)] = current_token_id
            current_token_id += 1
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = current_token_id
            current_token_id += 1
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # corpus is a list of muspy pieces
        tokens =[]
        ids = []
        for piece in corpus:
            resolution = piece.resolution
            barlines = piece.barlines
            for track in piece.tracks:
                if len(track.harmony) > 0:
                    harmony = track.harmony
                    # adjust time in each chord symbol
                    bar_idx = 0
                    bar_time = barlines[bar_idx].time

                    # in parallel, create tokens list
                    harmony_tokens = ['bar']
                    harmony_ids = [self.vocab['bar']]

                    for h in harmony:
                        # check if chord time exceeds next bar time
                        if bar_idx+1 < len(barlines):
                            while h.time >= barlines[bar_idx + 1].time:
                                bar_idx += 1
                                bar_time = barlines[bar_idx].time
                                harmony_tokens.append( 'bar' )
                                harmony_ids.append(self.vocab['bar'])
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get root pc
                        tmp_token = 'chord_root_' + str(h.root_pc)
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get pitch classes from mir_eval
                        pcs = (h.root_pc + np.where(h.binary_mir_eval > 0)[0])%12
                        for pc in pcs:
                            if pc != h.root_pc:
                                tmp_token = 'chord_pc_' + str(pc)
                                harmony_tokens.append( tmp_token )
                                harmony_ids.append(self.vocab[ tmp_token ])
                    tokens.append(harmony_tokens)
                    ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class RootPCTokenizer

class GCTRootPCTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        '''
        0: unk
        1: padding
        2: beginning of sequence
        3: end of sequence
        4: empty chord
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.unk_token = 'unk'
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.empty_chord = 'emp'
        # self.root_offset = 5
        # self.type_offset = self.root_offset + 12
        # self.max_num_segments = max_num_segments
        # self.segment_offset = self.root_offset + 12
        # self.pad_to_length = pad_to_length
        # self.vocab_size = self.segment_offset + max_num_segments
        # self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'unk': 0,
            'pad': 1,
            'bos': 2,
            'eos': 3,
            'emp': 4,
            'bar': 5
        }
        # positions
        current_token_id = 7
        # up to 8/4 time signatures
        for quart in range(8):
            for sixteenth in ['0', '5']:
                self.vocab['position_' + str(quart) + 'x' + sixteenth] = current_token_id
                current_token_id += 1
        # chord root and pitch classes
        for root in range(12):
            self.vocab['chord_root_' + str(root)] = current_token_id
            current_token_id += 1
        for pc in range(12):
            self.vocab['chord_pc_' + str(pc)] = current_token_id
            current_token_id += 1
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # corpus is a list of muspy pieces
        tokens =[]
        ids = []
        for piece in corpus:
            resolution = piece.resolution
            barlines = piece.barlines
            for track in piece.tracks:
                if len(track.harmony) > 0:
                    harmony = track.harmony
                    # adjust time in each chord symbol
                    bar_idx = 0
                    bar_time = barlines[bar_idx].time

                    # in parallel, create tokens list
                    harmony_tokens = ['bar']
                    harmony_ids = [self.vocab['bar']]

                    for h in harmony:
                        # check if chord time exceeds next bar time
                        if bar_idx+1 < len(barlines):
                            while h.time >= barlines[bar_idx + 1].time:
                                bar_idx += 1
                                bar_time = barlines[bar_idx].time
                                harmony_tokens.append( 'bar' )
                                harmony_ids.append(self.vocab['bar'])
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get gct
                        g = gct( (h.root_pc + np.where(h.binary_mir_eval > 0)[0])%12 )
                        # get root pc
                        tmp_token = 'chord_root_' + str( g[0] )
                        harmony_tokens.append( tmp_token )
                        harmony_ids.append(self.vocab[ tmp_token ])
                        # get pitch classes from mir_eval
                        for pc in g[2:]:
                            tmp_token = 'chord_pc_' + str((pc+g[0])%12)
                            harmony_tokens.append( tmp_token )
                            harmony_ids.append(self.vocab[ tmp_token ])
                    tokens.append(harmony_tokens)
                    ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class GCTRootPCTokenizer