import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from torch.autograd import Variable
import torch
import torch.nn as nn

from torchtext.legacy import data

def normalise_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    text = text.str.replace(r"\n"," ")
    text = text.str.replace(r"IMG-S+","IMAGE")
    text = text.str.strip()
    return text

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

with open('dat.npy','rb') as f:
    T = np.load(f)

T = np.hstack((T,np.ones((T.shape[0],1))))
tr = pd.DataFrame(T)

tr.columns = ['text','target']

tr["text"]=normalise_text(tr["text"])
t = tr.shape[0]
for i in range(0,t):
    s = t-i-1
    if len(tr['text'][s])==0:
        tr = tr.drop(s,axis = 0)



tr_df, valid_df = train_test_split(tr,train_size = tr.shape[0]-1)

SEED = 42

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


fields = [('text',TEXT), ('label',LABEL)]
train_ds = DataFrameDataset.splits(fields, train_df=tr_df)[0]

MAX_VOCAB_SIZE = 100000

print(train_ds)

TEXT.build_vocab(train_ds,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.200d',
                 unk_init = torch.Tensor.zero_)

LABEL.build_vocab(train_ds)




class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim*3)

        self.fc2 = nn.Linear(hidden_dim*3, hidden_dim*2)

        self.fc3 = nn.Linear(hidden_dim*2,1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu())

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        #unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
        output = self.dropout(self.fc3(output))

        #hidden = [batch size, hid dim * num directions]

        return output

model = torch.load('model.pth')
device = torch.device('cpu')

model.eval()
"""
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)


model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
"""
test_sen1 = 'lol'
tes = np.expand_dims(np.array(test_sen1),axis = 0)
tes = np.hstack(([tes],np.ones((1,1))))

tes = pd.DataFrame(tes)
tes.columns = ['text','target']
test_sen2 = 'that can be proved using induction'
tes2 = np.expand_dims(np.array(test_sen2),axis = 0)
tes2 = np.hstack(([tes2],np.ones((1,1))))

tes2 = pd.DataFrame(tes2)
tes2.columns = ['text','target']

tes,tes2 = DataFrameDataset.splits(fields, train_df=tes, val_df  = tes2)



tes,tes2 = data.BucketIterator.splits(
    (tes,tes2),
    batch_size = 1,
    sort_within_batch = True,
    device = device)

predictions = None
for batch in tes:
    text, len = batch.text
    predictions = model(text,len).squeeze(1)


print(predictions)
