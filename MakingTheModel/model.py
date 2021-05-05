# import os
# import shutil
# import numpy as np
# import tensorflow as tf
# #import tensorflow_text as text
# #from official.nlp import optimization
#
# #1 means keep, 0 means remove!
#
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64
#
# with open("pos.npy","rb") as f:
#     posi = np.load(f)
#
# with open("neg.npy","rb") as f:
#     negi = np.load(f)
#
# X = np.ones(posi.shape)
# Y = np.zeros(negi.shape)
#
# T = np.hstack((X,posi))
# T = np.vstack((T,np.hstack((Y,negi))))
#
# np.random.shuffle(T)
#
# print(T.shape)
#
#
#
# TrainLabel = T[:,0].astype('float64')
# TrainData = T[:,1]
#
#
#
# encoder = tf.keras.layers.experimental.preprocessing.TextVectorization()
# encoder.adapt(TrainData(lambda text, label: text))
#
# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         # Use masking to handle the variable sequence lengths
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation = 'relu'),
#     tf.keras.layers.Dense(1, activation = 'sigmoid')
# ])
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])
#
# history = model.fit(TrainData,TrainLabel, epochs=10,
#                     validation_data=Val_dataset,
#                     validation_steps=30)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import torch
import torch.nn as nn

from torchtext.legacy import data

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

fields = [('text',TEXT), ('label',LABEL)]
train_ds = DataFrameDataset.splits(fields, train_df=tr_df)[0]

MAX_VOCAB_SIZE = 100000

print(train_ds)

TEXT.build_vocab(train_ds,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.200d',
                 unk_init = torch.Tensor.zero_)


with open("pos.npy","rb") as f:
    posi = np.load(f)

with open("neg.npy","rb") as f:
    negi = np.load(f)

X = np.ones(posi.shape)
Y = np.zeros(negi.shape)

T = np.hstack((posi,X))
T = np.vstack((T,np.hstack((negi,Y))))

np.random.shuffle(T)

np.random.shuffle(T)

train = pd.DataFrame(T)


train.columns = ['text','target']



train["text"]=normalise_text(train["text"])
t = train.shape[0]

for i in range(0,t):
    s = t-i-1
    if len(train['text'][s])==0:
        train = train.drop(s,axis = 0)



train_df, valid_df = train_test_split(train,train_size = 1534)





train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)

MAX_VOCAB_SIZE = 20000


LABEL.build_vocab(train_ds)

BATCH_SIZE = 256

device = torch.device('cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_ds, val_ds),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)


# Hyperparameters
num_epochs = 60
learning_rate = 0.001

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 6
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding

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


model = LSTM_net(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)


pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)


model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)



model.to(device) #CNN to GPU


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.text

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator):

    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label)

            epoch_acc += acc.item()

    return epoch_acc / len(iterator)


t = time.time()
loss=[]
acc=[]
val_acc=[]

for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evaluate(model, valid_iterator)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Acc: {valid_acc*100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)

torch.save(model, 'model.pth')

print(f'time:{time.time()-t:.3f}')

plt.xlabel("runs")
plt.ylabel("normalised measure of loss/accuracy")
x_len=list(range(len(acc)))
plt.axis([0, max(x_len), 0, 1])
plt.title('result of LSTM')
loss=np.asarray(loss)/max(loss)
plt.plot(x_len, loss, 'r.',label="loss")
plt.plot(x_len, acc, 'b.', label="accuracy")
plt.plot(x_len, val_acc, 'g.', label="val_accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)
plt.show
X = input("Continue?")
