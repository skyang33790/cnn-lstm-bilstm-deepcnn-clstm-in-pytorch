# torchtext/data.py
# Minimal torchtext(0.2.x)-style shim (pure python) for legacy projects.
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


class Vocab:
    def __init__(self, counter: Counter, min_freq: int = 1, specials: Optional[List[str]] = None):
        specials = specials or []
        # keep specials in front, unique
        seen = set()
        itos = []
        for sp in specials:
            if sp not in seen:
                itos.append(sp)
                seen.add(sp)

        # add tokens by freq then alpha (stable-ish)
        words = [w for w, c in counter.items() if c >= min_freq and w not in seen]
        words.sort(key=lambda w: (-counter[w], w))
        itos.extend(words)

        self.itos: List[str] = itos
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)


class Pipeline:
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)


class Field:
    def __init__(
        self,
        sequential: bool = True,
        use_vocab: bool = True,
        lower: bool = False,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        batch_first: bool = False,
        include_lengths: bool = False,
    ):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.lower = lower
        self.tokenize = tokenize or (lambda s: s.split())
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.batch_first = batch_first
        self.include_lengths = include_lengths

        self.vocab: Optional[Vocab] = None

    def preprocess(self, x: Any) -> Any:
        # allow legacy: field.preprocessing = data.Pipeline(fn)
        fn = getattr(self, "preprocessing", None)
        if fn is not None:
            x = fn(x)

        if not self.sequential:
            return str(x).strip()

        s = str(x)
        if self.lower:
            s = s.lower()
        return self.tokenize(s)


    def build_vocab(self, *args, min_freq: int = 1):
        counter = Counter()

        for a in args:
            # dataset
            if hasattr(a, "examples") and hasattr(a, "fields"):
                ds = a
                # find our field name in dataset.fields
                my_names = [n for n, f in ds.fields.items() if f is self]
                if not my_names:
                    continue
                name = my_names[0]
                for ex in ds.examples:
                    val = getattr(ex, name)
                    if self.sequential:
                        counter.update(val)
                    else:
                        counter.update([val])
            else:
                # iterable
                for item in a:
                    if self.sequential:
                        counter.update(item)
                    else:
                        counter.update([item])

        specials = []
        if self.unk_token is not None:
            specials.append(self.unk_token)
        if self.pad_token is not None and self.sequential:
            specials.append(self.pad_token)

        self.vocab = Vocab(counter, min_freq=min_freq, specials=specials)

    def numericalize(self, arr: Any) -> Any:
        if not self.use_vocab:
            return arr
        if self.vocab is None:
            raise RuntimeError("Vocab not built. Call field.build_vocab(...) first.")
        if not self.sequential:
            return self.vocab.stoi.get(arr, self.vocab.stoi.get(self.unk_token, 0))
        unk = self.vocab.stoi.get(self.unk_token, 0)
        return [self.vocab.stoi.get(tok, unk) for tok in arr]


class Example:
    @classmethod
    def fromlist(cls, data_list: List[Any], fields: List[Tuple[str, Field]]):
        ex = cls()
        for (name, field), val in zip(fields, data_list):
            setattr(ex, name, field.preprocess(val))
        return ex


class Dataset:
    def __init__(self, examples: List[Example], fields: List[Tuple[str, Field]]):
        self.examples = examples
        self.fields: Dict[str, Field] = dict(fields)

    def __len__(self):
        return len(self.examples)
    def __getattr__(self, name: str):
        # legacy torchtext: dataset.text / dataset.label => list of each example's field value
        if "fields" in self.__dict__ and name in self.fields:
            return [getattr(ex, name) for ex in self.examples]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


@dataclass
class Batch:
    # dynamic attributes will be attached (text, label, etc.)
    pass


class Iterator:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        device: int = -1,
        shuffle: bool = False,
        repeat: bool = False,
        sort: bool = False,
        sort_key: Optional[Callable[[Example], Any]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.repeat = repeat
        self.sort = sort
        self.sort_key = sort_key

        self._device = torch.device("cpu") if device == -1 else torch.device(f"cuda:{device}")

    @classmethod
    def splits(
        cls,
        datasets: Tuple[Dataset, Dataset, Dataset],
        batch_sizes: Tuple[int, int, int],
        device: int = -1,
        shuffle: bool = True,
        repeat: bool = False,
        sort: bool = False,
        sort_key: Optional[Callable[[Example], Any]] = None,
    ):
        train, dev, test = datasets
        bs_train, bs_dev, bs_test = batch_sizes
        return (
            cls(train, bs_train, device=device, shuffle=shuffle, repeat=repeat, sort=sort, sort_key=sort_key),
            cls(dev, bs_dev, device=device, shuffle=False, repeat=False, sort=sort, sort_key=sort_key),
            cls(test, bs_test, device=device, shuffle=False, repeat=False, sort=sort, sort_key=sort_key),
        )

    def __iter__(self):
        while True:
            idxs = list(range(len(self.dataset.examples)))
            if self.sort and self.sort_key is not None:
                idxs.sort(key=lambda i: self.sort_key(self.dataset.examples[i]))
            if self.shuffle:
                random.shuffle(idxs)

            for start in range(0, len(idxs), self.batch_size):
                batch_ids = idxs[start : start + self.batch_size]
                examples = [self.dataset.examples[i] for i in batch_ids]
                yield self._batchify(examples)

            if not self.repeat:
                break

    def _batchify(self, examples: List[Example]) -> Batch:
        b = Batch()
        b.batch_size = len(examples)
        for name, field in self.dataset.fields.items():
            vals = [getattr(ex, name) for ex in examples]

            if field.sequential:
                # numericalize
                nums = [field.numericalize(v) for v in vals]
                lengths = torch.tensor([len(x) for x in nums], dtype=torch.long)
                max_len = int(lengths.max().item()) if len(nums) > 0 else 0

                pad_idx = 0
                if field.use_vocab and field.vocab is not None and field.pad_token is not None:
                    pad_idx = field.vocab.stoi.get(field.pad_token, 0)

                padded = []
                for x in nums:
                    x = list(x)
                    if len(x) < max_len:
                        x = x + [pad_idx] * (max_len - len(x))
                    padded.append(x)

                # torchtext default: [seq_len, batch]
                tensor = torch.tensor(padded, dtype=torch.long)  # [batch, seq]
                if not field.batch_first:
                    tensor = tensor.t().contiguous()  # [seq, batch]

                tensor = tensor.to(self._device)
                lengths = lengths.to(self._device)

                if field.include_lengths:
                    setattr(b, name, (tensor, lengths))
                else:
                    setattr(b, name, tensor)
            else:
                # label etc.
                nums = [field.numericalize(v) for v in vals]
                tensor = torch.tensor(nums, dtype=torch.long).to(self._device)
                setattr(b, name, tensor)

        return b
