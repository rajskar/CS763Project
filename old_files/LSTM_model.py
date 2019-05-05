#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:41:41 2019

@author: rajs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
INPUT_DIM = 2048
HIDDEN_DIM = 1024

######################################################################
# Create the model:

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        
        # The linear layer that maps from hidden state space to output space
        self.hidden2linear = nn.Linear(hidden_dim, 512, bias=True)
        self.dropout       = nn.Dropout(0.5)
        self.linear2output = nn.Linear(512, tagset_size, bias=True)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Train the model:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


######################################################################
# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the example above, each word had an embedding, which served as the
# inputs to our sequence model. Let's augment the word embeddings with a
# representation derived from the characters of the word. We expect that
# this should help significantly, since character-level information like
# affixes have a large bearing on part-of-speech. For example, words with
# the affix *-ly* are almost always tagged as adverbs in English.
#
# To do this, let :math:`c_w` be the character-level representation of
# word :math:`w`. Let :math:`x_w` be the word embedding as before. Then
# the input to our sequence model is the concatenation of :math:`x_w` and
# :math:`c_w`. So if :math:`x_w` has dimension 5, and :math:`c_w`
# dimension 3, then our LSTM should accept an input of dimension 8.
#
# To get the character level representation, do an LSTM over the
# characters of a word, and let :math:`c_w` be the final hidden state of
# this LSTM. Hints:
#
# * There are going to be two LSTM's in your new model.
#   The original one that outputs POS tag scores, and the new one that
#   outputs a character-level representation of each word.
# * To do a sequence model over characters, you will have to embed characters.
#   The character embeddings will be the input to the character LSTM.
#
