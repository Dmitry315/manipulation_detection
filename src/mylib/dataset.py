import torch
from torch.utils.data import DataLoader, Dataset


class SequenceLabelDataset(Dataset):
  def __init__(self, df, tokenizer, n_tags, max_len=512):
    self.df = df
    self.markup = []
    self.n_tags = n_tags
    self.max_len = max_len
    self.tokenizer = tokenizer
    # make span markup with tokenizer
    for id, row in self.df.iterrows():
      spans = []
      for span in row.markup:
        s = span['start']
        e = span['end']
        t = span['tag']
        new_s = self.tokenizer(row.text[:s], add_special_tokens=False, return_length=True)['length'][0]
        new_e = new_s + self.tokenizer(row.text[s:e], add_special_tokens=False, return_length=True)['length'][0]
        spans.append({'start': new_s, 'end': new_e, 'tag': t})
      self.markup.append(spans)

  def __len__(self):
    return len(self.df)

  def get_tokens_from_text(self, text):
    tokens = self.tokenizer(text,
                       padding='max_length', max_length = self.max_len, truncation=True,
                       return_tensors="pt")
    return tokens
  
  def get_labels_from_markup(self, text, markup):
    labels = torch.zeros((self.max_len, self.n_tags), dtype=torch.float32)
    for span in markup:
      labels[span['start'] + 1:span['end'] + 1, span['tag']] = 1
    return labels

  def __getitem__(self, idx):
    text = self.df.text.iloc[idx]
    tokens = self.get_tokens_from_text(text)
    markup = self.markup[idx]
    labels = self.get_labels_from_markup(text, markup)
    return tokens, labels