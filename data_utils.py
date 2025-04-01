import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from harmony_tokenizers_m21 import MergedMelHarmTokenizer
import random
import os
import numpy as np
from transformers import DataCollatorForSeq2Seq
import torch.nn.functional as F

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy

class MergedMelHarmDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, return_harmonization_labels=False,\
                num_bars=8):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
        self.return_harmonization_labels = return_harmonization_labels
        self.randomization_rate = 0
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars,\
                                randomization_rate = self.randomization_rate)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        if self.return_harmonization_labels:
            input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
            # Generate labels: mask the question part
            sep_token_idx = (input_ids == self.merged_tokenizer.vocab['<h>']).nonzero(as_tuple=True)[0]
            labels = input_ids.clone()
            labels[:sep_token_idx + 1] = -100  # Ignore question tokens and <h> in loss computation
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        elif self.return_attention_mask:
            return {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
            }
        else:
            return torch.tensor(encoded['input_ids'], dtype=torch.long)
    # end getitem
# end class dataset

class SeparatedMelHarmDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, num_bars=8):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        # separate melody from harmony
        labels = torch.tensor(encoded['input_ids']).clone()
        start_harmony_position = np.where( np.array(encoded['input_ids']) == self.merged_tokenizer.vocab[self.merged_tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
        input_ids = torch.tensor(encoded['input_ids'][:start_harmony_position], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'][:start_harmony_position], dtype=torch.long)
        labels = labels[start_harmony_position:]  # Ignore question tokens and <h> in loss computation
        labels[ labels == self.merged_tokenizer.pad_token_id ] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    # end getitem
# end class dataset

class MLMCollator:
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    # end init

    def __call__(self, batch):
        input_ids = torch.stack(batch)
        labels = input_ids.clone()
        
        # Create mask
        rand = torch.rand(input_ids.shape)
        mask = (rand < self.mask_prob) & (input_ids != self.tokenizer.vocab[self.tokenizer.pad_token])
        
        # Apply mask
        for i in range(input_ids.shape[0]):
            mask_idx = torch.where(mask[i])[0]
            for idx in mask_idx:
                prob = random.random()
                if prob < 0.8:
                    input_ids[i, idx] = self.mask_token_id  # 80% <mask>
                elif prob < 0.9:
                    input_ids[i, idx] = random.randint(0, len(self.tokenizer.vocab) - 1)  # 10% random
                # 10% unchanged (do nothing)

        # Replace labels of non-mask tokens with -100
        labels[~mask] = -100
        
        return {"input_ids": input_ids, "labels": labels}
    # end call

# end class MLMCollator

class GenCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
    # end init

    def __call__(self, batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class GenCollator

class PureGenCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
    # end init

    def __call__(self, batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class PureGenCollator

class MaskedGenCollator:
    def __init__(self, tokenizer, mask_prob=0.2, bar_id=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id  # Mask token ID
        self.bar_id = bar_id
    # end init

    def __call__(self, batch):
        # Apply masking before padding
        masked_input_ids = []
        for item in batch:
            input_ids = item["input_ids"].clone()
            for i in range(len(input_ids)):
                if input_ids[i] != self.bar_id and random.random() < self.mask_prob:
                    input_ids[i] = self.mask_token_id  # Replace with <mask>
            masked_input_ids.append(input_ids)

        input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class MaskedGenCollator

class MaskedDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, mask_prob=0.2, bar_id=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.bar_id = bar_id  # Token that should not be masked
    # end init

    def __call__(self, features):
        batch = super().__call__(features)  # Get the default behavior
        
        # Apply masking only to the encoder input (input_ids)
        masked_input_ids = batch["input_ids"].clone()
        for i in range(masked_input_ids.shape[0]):  # Iterate over batch
            for j in range(masked_input_ids.shape[1]):  # Iterate over sequence
                if masked_input_ids[i, j] != self.bar_id and random.random() < self.mask_prob:
                    masked_input_ids[i, j] = self.mask_token_id

        batch["input_ids"] = masked_input_ids  # Replace with masked version
        return batch
    # end call
# end class MaskedDataCollatorForSeq2Seq