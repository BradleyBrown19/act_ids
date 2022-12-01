import pdb
import random
from itertools import permutations
from typing import Optional, Set

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from torch.utils.data import IterableDataset,Dataset

class MathDataModule(LightningDataModule):
    def __init__(
        self, 
        dataset, 
        train_batch_size,
        val_batch_size,
        train_shuffle,
        num_workers,
        pin_memory,
        persistent_workers
    ):
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prepare_data_per_node = False # Yelled at me without this
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.dataset.get_train_dataset()
        self.val_dataset = self.dataset.get_val_dataset()

    def _log_hyperparams(self):
        return {
            "train_batch_size": self.train_batch_size,
            "p": self.dataset.p,
            "frac_train": self.dataset.frac_train
        }
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=self.train_shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

class MathDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, index):
        return self.examples[index]
    
    def __len__(self): 
        return len(self.examples)

class MathDatasetCreator():
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float, fix_val_pct: float):
        self.frac_train = frac_train
        self.fix_val_pct = fix_val_pct
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ['o', '='] + list(self.group_elements1.union(self.group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(self.group_elements1.union(self.group_elements2))
        idxs = list(range(len(self.group_elements1)*len(self.group_elements2)))
        random.Random(9).shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs)*self.frac_train)], idxs[int(len(idxs)*self.frac_train):]
      
        if fix_val_pct != -1:
            assert fix_val_pct + frac_train <= 1
            self.val_pairs = idxs[-int(len(idxs)*self.fix_val_pct):]
    
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]
    
    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]
    
    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]
    
    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return {"input_ids": torch.tensor(self.encode(equation[:-1])), "labels": torch.tensor(self.vocab2idx[c]-2)}
    
    def get_val_dataset(self):
        return MathDataset([self.fetch_example(self.val_pairs[i]) for i in range(len(self.val_pairs))])
    
    def get_train_dataset(self):
        return MathDataset([self.fetch_example(self.train_pairs[i]) for i in range(len(self.train_pairs))])


class ModSumDataset(MathDatasetCreator):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a + b) % self.p

class ModSubtractDataset(MathDatasetCreator):
    def __init__(self, p, frac_train):
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a - b) % self.p

class ModDivisionDataset(MathDatasetCreator):
    def __init__(self, p, frac_train, fix_val_pct):
        super(ModDivisionDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train, fix_val_pct)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a * pow(b, self.p-2, self.p)) % self.p

class PermutationGroup(MathDatasetCreator):
    def __init__(self, k, frac_train):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])