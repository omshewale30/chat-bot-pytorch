import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import Brain
with open('intents.json', 'r') as f:
    body = json.load(f)

all_words = []
tags = []
xy = []
for intent in body['intents']:
    tag = intent['tag']
    tags.append(tag)
    for p in intent['patterns']:
        tokenized_sentence = tokenize(p)
        all_words.extend(tokenized_sentence)  # adding all the tokenized words to the all words array
        xy.append((tokenized_sentence, tag))

ignore_words = ['?', '!', ".", ","'']

all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (tokenized_sentence, tag) in xy:
    X_train.append(bag_of_words(tokenized_sentence, all_words))

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x = X_train
        self.y = Y_train

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.n_samples
#hyperparams

output_size= len(tags)
input_size = len(all_words)
hidden_size=8
batch_size= 8
learning_rate = 0.001
epochs = 1000


dataset = ChatDataSet()
train_loader = DataLoader(dataset, batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
brain = Brain(input_size=input_size, hidden_size=hidden_size,output_size=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(brain.parameters(),lr=learning_rate)

for epoch in range(epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)


        #forward propagation
        outputs= brain(words)
        loss = criterion(outputs,labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss:{loss.item():.4f}')

print(f'final loss {loss.item():.4f}')

