import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceLabelDataset(Dataset):


  def __init__(self, df, tokenizer, n_tags, label_pad_value=-1):
    self.df = df
    self.markup = []
    self.n_tags = n_tags
    self.label_pad_value = label_pad_value
    self.tokenizer = tokenizer
    # make span markup with tokenizer
    for id, row in self.df.iterrows():
      spans = []
      for span in row.markup:
        s = span['start']
        e = span['end']
        t = span['tag']
        new_s = len(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(row.text[:s]))
        # new_s = self.tokenizer(row.text[:s], add_special_tokens=False, return_length=True)['length'][0]
        new_e = new_s + len(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(row.text[s:e]))
        # new_e = new_s + self.tokenizer(row.text[s:e], add_special_tokens=False, return_length=True)['length'][0]
        spans.append({'start': new_s, 'end': new_e, 'tag': t})
      self.markup.append(spans)

  def __len__(self):
    return len(self.df)
  
  def get_tokens(self, idx):
    text = self.df['text'].iloc[idx]
    tokens = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    tokens = [w for w, _ in tokens]
    return tokens
  
  def get_labels(self, idx, length):
    spans = self.markup[idx]
    labels = np.zeros((length, self.n_tags))
    for span in spans:
      s = span['start']
      e = span['end']
      t = span['tag']
      labels[s:e, t] = 1

    return labels

  def __getitem__(self, idx):
    text = self.get_tokens(idx)
    labels = self.get_labels(idx, length=len(text))

    return text, labels


class TransformersCollator:


  def __init__(self, tokenizer, tokenizer_kwargs, label_padding_value):
    self.tokenizer = tokenizer
    self.tokenizer_kwargs = tokenizer_kwargs
    self.label_padding_value = label_padding_value

  def __call__(self, batch):
    tokens, labels = zip(*batch)

    tokens = self.tokenizer(tokens, **self.tokenizer_kwargs)
    labels = self.encode_labels(tokens, labels, self.label_padding_value)

    tokens.pop("offset_mapping")

    return tokens, labels

  @staticmethod
  def encode_labels(tokens, labels, label_padding_value):
    encoded_labels = []

    for doc_labels, doc_offset in zip(labels, tokens.offset_mapping):

      doc_enc_labels = np.full(
        (len(doc_offset), doc_labels.shape[1]), 
        fill_value=label_padding_value, 
        dtype=int
      )
      arr_offset = np.array(doc_offset)[..., None]
      # TODO: make it like human with vectorization
      mask = (arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)
      mask = mask[:, 0]
      for i in range(doc_enc_labels.shape[1]):
        doc_enc_labels[:, i][mask] = doc_labels[:mask.sum(), i]
      encoded_labels.append(doc_enc_labels.tolist())

    return torch.LongTensor(encoded_labels)