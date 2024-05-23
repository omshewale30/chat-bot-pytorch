import random
import json
import torch
from model import Brain
from nltk_utils import stem,bag_of_words,tokenize

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    body = json.load(f)

FILE = 'data.pth'

data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words= data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

brain = Brain(input_size,hidden_size=hidden_size,output_size=output_size)

brain.load_state_dict(model_state)
brain.eval()

bot_name = "Om"
print("Lets chat! type 'quit' to exit")

while True:
    sentence = input('You:')
    if sentence=='quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output= brain(X)
    _,pred = torch.max(output,dim=1) #getting the prediction with maximum probability
    tag = tags[pred.item()]

    probabilities= torch.softmax(output,dim=1)
    prob = probabilities[0][pred.item()]
    if prob>0.75:
        for intent in body["intents"]:
            if intent["tag"]==tag:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: I do not understand, please reframe your text')


