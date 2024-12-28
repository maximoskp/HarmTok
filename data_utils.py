import torch
from torch.utils.data import Dataset
from MergedMelHarmTokenizer

class MergedMelodyHarmonyDataset(Dataset):
    def __init__(self, root_dir, melody_tokenizer, harmony_tokenizer, max_length):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.melody_tokenizer = melody_tokenizer
        self.harmony_tokenizer = harmony_tokenizer
        self.merged_tokenizer = MergedMelHarmTokenizer(self.melody_tokenizer, self.harmony_tokenizer)
        self.max_length = max_length
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length)
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long)
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
        }
    # end getitem
