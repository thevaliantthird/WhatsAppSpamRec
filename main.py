import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from torch.autograd import Variable
import torch
import torch.nn as nn

from torchtext.legacy import data

class Text (object) :

    def __init__ (self, fileName):
        self.FileRead = open(fileName, "r",encoding = "charmap")
        self.Text = []
        self.Counter = 0
        self.ifEnd = False



    def IfStart (self) :

        K = self.FileRead.tell()
        readUp = self.FileRead.read(20)
        res = False
        Name = ""
        if len(readUp) == 20 and readUp[2] == '/' and readUp[5] == '/' and readUp[10] == ','  \
        and readUp[11] == ' ' and readUp[14] == ':' and readUp[17] == ' ' and readUp[18] == '-' \
        and readUp[19] == ' ' and readUp[0:2].isnumeric() and readUp[3:5].isnumeric() and readUp[6:10].isnumeric() \
        and readUp[12:14].isnumeric() and readUp[15:17].isnumeric() :
            res  = True
            str = self.FileRead.read(1)
            while str[0] != ':' :
                readUp+=str
                str = self.FileRead.read(1)
            Name = readUp

        else:
            self.FileRead.seek(K)

        return (res,Name)

    def ReadAndStore (self):
        ifPrevTerm  = True
        Start = True
        Sender = ""
        Msg = ""
        while len(self.FileRead.read(1)) != 0:
            self.FileRead.seek(self.FileRead.tell()-1)
            if ifPrevTerm:
                #print(self.FileRead.tell())
                resu = self.IfStart()
                if resu[0]:
                    if Start:
                        Sender = resu[1]
                        Start = False
                    else:
                         self.Text.append((Sender,Msg))
                    #     self.Print()
                         Msg = ""
                         Sender = resu[1]
                else:
                    if len(Sender) == 0:
                        print("There is some inconsistency with the text!")
                        break
                    Msg+=self.FileRead.read(1)
            else:
                buff = self.FileRead.read(1)
                if buff == '\n':
                    ifPrevTerm = True
                Msg+=buff
        if len(Sender) != 0:
            self.Text.append((Sender,Msg))


    def Print(self):
        for (a,b) in self.Text:
            print(a+b)

    def GetText (self) :
        return self.Text


fi = input("What's the name of the file you wanna process?")
myText = Text(fi)
myText.ReadAndStore()


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
j = 0
for i in range(0,t):
    s = t-i-1
    if len(tr['text'][s])==0:
        tr = tr.drop(s,axis = 0)
    else:
        tr['target'][s] = str(j)
        j+=1




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

MAX_VOCAB_SIZE = 20000

#print(train_ds)

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
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),enforce_sorted=False)

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

texti = myText.GetText()
ToPrint = []

tes = []

for (a,b) in texti:
    tes.append(b)

tes = np.vstack((np.array(tes),np.arange(len(tes))))
tes = tes.T

tes = pd.DataFrame(tes)
tes.columns = ['text','target']

tes['text'] = normalise_text(tes['text'])
#print(type(tes['text'][0]))
for i in range(0,tes.shape[0]):

    if len(tes['text'][i])==0:
        tes.drop(i,axis = 0)

test_sen2 = 'that can be proved using induction'
tes2 = np.expand_dims(np.array(test_sen2),axis = 0)
tes2 = np.hstack(([tes2],np.ones((1,1))))

tes2 = pd.DataFrame(tes2)
tes2.columns = ['text','target']
#print(tes)
tes,tes2 = DataFrameDataset.splits(fields, train_df=tes, val_df  = tes2)

ToKeep = []
tes,tes2 = data.BucketIterator.splits(
         (tes,tes2),
         batch_size = 1,
         sort_within_batch = False,
         device = device)

#print(typeLABEL.vocab.stoi['6901'])

for batch in tes:


    text, len = batch.text
    # print('Done!')
    # print(text)
    sha = list(text.size())
    if sha[0]!= 0 and sha[1]!=0:
        predictions = model(text,len).squeeze(1)
        if predictions > 0.2:
            ToKeep.append(int(batch.label))

#print(ToKeep)

with open("(PROCESSED) "+fi,"w") as f:
    for x in ToKeep:
        h = int(LABEL.vocab.itos[x])

        f.write(texti[h][0]+':'+texti[h][1])

f.close()
