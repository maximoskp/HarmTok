import torch
from torch.utils.data import Dataset
from harmony_tokenizers_m21 import MergedMelHarmTokenizer
import random
import os

class MergedMelodyHarmonyDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, return_attention_mask=False):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length, pad_to_max_length=True)
        if self.return_attention_mask:
            return {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
            }
        else:
            return torch.tensor(encoded['input_ids'], dtype=torch.long)
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
