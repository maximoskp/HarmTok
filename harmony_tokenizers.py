import numpy as np
from transformers import PreTrainedTokenizerBase
import mir_eval

class ChordSymbolTokenizer(PreTrainedTokenizerBase):
    def __init__(self, max_num_segments=0, pad_to_length=0, model_max_length=0):
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
        for root in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            for alteration in ['', '#', 'b', '##', 'bb']:
                for quality in mir_eval.chord.QUALITIES.keys():
                    self.vocab[ root + alteration + ':' + quality ] = current_token_id
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
                                harmony_ids = [self.vocab['bar']]
                                if bar_idx+1 >= len(barlines):
                                    break
                        # quantize to 0.5
                        tmp_time = h.time - bar_time
                        quant_time = round(2*tmp_time/resolution)/2
                        # replace '.' with 'x' so that word level HF tokenizers don't split it
                        tmp_token = 'position_' + str(quant_time).replace('.', 'x')
                        harmony_tokens.append( tmp_token )
                        harmony_ids = [self.vocab[ tmp_token ]]
                        # append chord symbol
                        tmp_token = h.chord_symbol_mir_eval
                        harmony_tokens.append( tmp_token )
                        harmony_ids = [self.vocab[ tmp_token ]]
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
# end class GCTSerialChromaTokenizer