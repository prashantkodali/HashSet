import numpy as np
import json 
import os
import pickle
import pandas as pd
import torch
import torch.optim as optim

## shell call to download datasets
#os.sys('pip3 install datasets transformers seqeval conllu wandb')

from tqdm import tqdm
from conllu.models import TokenList, Token

file = open("NER_datasets/conllpp_train.txt","r+") 
lines = file.readlines()
## Get training and testing data
sentence = ''
labels = ''
sentence_sep = ''
count = 0
x_fin = 219553
x = 0
ner = []
for i,line in enumerate(lines):

    x += 1
    if x == x_fin:
        break
    if line == '-DOCSTART- -X- -X- O\n':
        continue
    if line != '\n':
        word = line.split()[0]
        sentence += word
        sentence_sep += '_' + word
        label = line.split()[-1].strip('\n').split('-')[0]
        if label == 'B':
            labels += 'B'
            for i in range(len(word)-1):
                labels += 'B'
        elif label == 'I':
            for i in range(len(word)):
                labels += 'I'
        elif label == 'O':
            for i in range(len(word)):
                labels += 'O'
        count += 1
    else:
        if count != 0:
            ner.append((sentence, labels, count))
            # separated.append((sentence_sep[1:], labels))
            sentence = ''
            labels = ''
            count = 0


'''
converting sampels to conll format and writing to conll file
'''
allsamples = []

for sample in tqdm(ner):

  assert len(list(sample[0])) == len(list(sample[1]))

  tl = TokenList()

  for ind,(c,a) in enumerate(zip (list(sample[0].lower()),list(sample[1]))):
    tl.append(Token(id =ind,form=c,tag= a))

  allsamples.append(tl)

with open('NER_datasets/test.conll', 'w') as f:
  f.writelines([sentence.serialize() + "\n" for sentence in allsamples])


## create index based dictonary for vocab(map tokens to numbers.)
##list1 contains on 
vocab_dict = {}
tag_map = {'B' : 0, 'I': 1,'O':2} ## Tag map for labels/annotations
list1 = [ str(x) for x in range(0,10)]
import string
for c in string.ascii_lowercase:
    list1.append(c)
#print(list1)
for idx,num in enumerate(list1):
    vocab_dict[num] = idx
num = len(vocab_dict) 
vocab_dict['PAD'] = num
#print(vocab_dict) ## Add vocabulary for numbers and all alphabets, rest can be added on the fly. 
vocab_len = len(vocab_dict.keys())



### convert text tokens to numbers using vocab dictionary.
ner_converted = []
for x in ner:
    string = x[0].lower()
    label = x[1]
    num = x[2]
    #print(string)
    conv_str = []
    for let in string:
        try:
            conv_str.append(vocab_dict[let])
        except:
            vocab_dict[let] = vocab_len
            vocab_len +=1
            conv_str.append(vocab_dict[let])
    conv_label = []
    for let in label:
        conv_label.append(tag_map[let])
    ner_converted.append((conv_str,conv_label,num))
ner_converted

max_len = 0
for a in ner_converted:
    max_len = max(len(a[0]),max_len)


from torch.autograd import Variable

batch_max_len = max([len(s[0]) for s in ner])

#prepare a numpy array with the data, initializing the data with 'PAD' 
#and all labels with -1; initializing labels to -1 differentiates tokens 
#with tags from 'PAD' tokens
batch_data = vocab_dict['PAD']*np.ones((len(ner_converted), batch_max_len))
batch_labels = -1*np.ones((len(ner_converted), batch_max_len))

#copy the data to the numpy array
for j in range(len(ner_converted)):
    cur_len = len(ner_converted[j][0])
    batch_data[j][:cur_len] = ner_converted[j][0]
    batch_labels[j][:cur_len] = ner_converted[j][1]

#since all data are indices, we convert them to torch LongTensors
batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

#convert Tensors to Variables
batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)


# Hyperparameter Dictionary
params = {}
params['vocab_size'] = len(vocab_dict)
params['embedding_dim'] = 6
params['lstm_hidden_dim'] = 1
params['number_of_tags']  = 3
params['num_epochs'] = 50
params['input_dim'] = 400


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #print('hello world')
        #maps each token to an embedding_dim vector
        
        self.embedding = nn.Embedding(params['vocab_size'], params['embedding_dim']) #vocab_size is len of dictionary, params.embedding_size = hyperparameter
        #print(self.embedding)
        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params['embedding_dim'], params['lstm_hidden_dim'], batch_first=True)

        #fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(params['lstm_hidden_dim'], params['number_of_tags'])
    
    def forward(self, s):
        #apply the embedding layer that maps each token to its embedding
        
    
        s = s.view(-1)
        #print(s)
        s = self.embedding(s)# dim: batch_size x batch_max_len x embedding_dim
        s = s.unsqueeze(0)
        
        #run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                

        #s, _ = self.lstm(s)
        #reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        #apply the fully connected layer and obtain the output for each token
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags

def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    
    #print(outputs.size(),labels.size())
    labels = labels.unsqueeze(0)
    labels = labels.view(-1)  

    #print('Labels shape',labels.shape)
    #print(labels)
    #mask out 'PAD' tokens
    mask = (labels >= 0 ).float()
    #print('Mask',mask)
    #the number of tokens is the sum of elements in mask

    num_tokens = int(torch.sum(mask).item())
    
    #pick the values corresponding to labels and multiply by mask
    #print('Outputs Shape',outputs.shape)
    xyz = range(outputs.shape[0])
    #print(xyz)
    outputs = outputs[xyz,labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


import matplotlib.pyplot as plt
def plot_loss(epoch_list,loss_list,title):
    plt.plot(epoch_list,loss_list)
    plt.title(title)
    plt.savefig('Plots/loss_curve.jpeg')



model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    #inputs = prepare_sequence(training_data[0][0], word_to_ix)
    inputs = batch_data[0]

    #print(inputs.size())
    #inputs = inputs.unsqueeze(0)
    tag_scores = model(inputs)
    #print(tag_scores)


num_epochs = params['num_epochs']


print('Starting Training..')
e_list = []
l_list = []
for epoch in range(params['num_epochs']): ## runnning for 50 epochs.
    #print('Epoch',epoch)
    for i in range(len(ner_converted)):
    #for i in range(0,30):
        sentence = batch_data[i]
        #sentense = sentence.unsqueeze(0)
        
        tags = batch_labels[i]
        #tags = tags.unsqueeze(0)
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        #print(tag_scores)
        loss = loss_fn(tag_scores, tags)
        loss.backward()
        optimizer.step()

    ## Get Accuracy
    #print(tag_scores.shape)
    #tag_scores = (tag_scores>0.5).float()
    #correct = (tag_scores == labels).float().sum()
    #print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/x.shape[0]))
    f = open('report.txt','a+')
    f.write("Epoch {}/{}, Loss: {:.3f},".format(epoch+1,num_epochs, loss.item()))
    f.write("\n")
    f.close()
    print("Epoch {}/{}, Loss: {:.3f},".format(epoch+1,num_epochs, loss.item()))
    e_list.append(epoch)
    l_list.append(loss.item())

plot_loss(epoch_list=e_list,loss_list=l_list,title = 'Loss vs Epochs')

# See what the scores are after training
with torch.no_grad():
    #inputs = prepare_sequence(training_data[0][0], word_to_ix)
    inputs = batch_data[0]
    tag_scores = model(inputs)

