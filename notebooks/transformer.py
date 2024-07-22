#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from MyTransformer import Classifier
import pandas as pd
import numpy as np
from preprocessing import preprocess
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# In[2]:


max_tokens = 128 #67
emb_size = 512


# In[7]:


model = Classifier(vocab_size=23174, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=67)
rand_input = torch.randint(0, 21000, (16,67))
out = model(rand_input)
out


# In[25]:


def build_filter_vocab(data, min_count=5):
    vocab = {}
    for sentence in data:
        for word in sentence.split():
            vocab[word] = vocab.get(word, 0) + 1
    filtered_vocab = {word: count for word, count in vocab.items() if count >= min_count}
    return filtered_vocab

def build_tokenized_vocab(vocab:dict):
    voc = {word: idx for idx, (word, _) in enumerate(vocab.items())}
    voc['<UNK>'] = len(voc)
    voc['<PAD>'] = len(voc)
    return voc

def pad_lists_in_df_column(df, column_name, desired_length, padding_value):
    """
    Pads lists in a specified column of a DataFrame to a desired length.

    Parameters:
    - df: The DataFrame to process.
    - column_name: The name of the column containing lists to pad.
    - desired_length: The desired length of the lists.
    - padding_value: The value to use for padding shorter lists.

    Returns:
    - A new DataFrame with padded lists in the specified column.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Pad each list in the specified column
    df_copy[column_name] = df_copy[column_name].apply(
        lambda x: x + [padding_value] * (desired_length - len(x)) if len(x) < desired_length else x
    )
    
    return df_copy

class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Assuming the DataFrame has two columns: 'features' and 'labels'
        # Adjust this method based on the actual structure of your DataFrame
        features = self.dataframe.iloc[idx, :-1].values # All columns except the last one
        label = self.dataframe.iloc[idx, -1] # Last column
        if label == 0:
            label = [1, 0]
        else:
            label = [0, 1]
        return torch.tensor(features, dtype=torch.int), torch.tensor(label, dtype=torch.float)

# Step 2: Function to create a DataLoader from a DataFrame
def create_dataloaders_from_df(dataframe, test_size=0.2, batch_size=32, shuffle=True):
    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(dataframe, test_size=test_size)
    
    # Create datasets
    train_dataset = DataFrameDataset(train_df)
    test_dataset = DataFrameDataset(test_df)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader

def train_model(model, train_loader, test_loader, n_epochs, loss_fn=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)):
    # Check if CUDA is available and move the model to GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            inputs, labels = batch
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs)  # Forward pass
            #
            # print('output dim:', outputs.shape, 'label dim:', labels.shape)
            loss = loss_fn(outputs.squeeze(), labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            total_loss += loss.item()
        print(evaluate_model(model, test_loader))
        
        
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


# In[21]:


df_pos = pd.read_csv('preprocessed_pos.csv')
df_neg = pd.read_csv('preprocessed_neg.csv')
df_pos['label'] = 1
df_neg['label'] = 0
neg_tweets=df_neg['hashtags_written_out'].values
pos_tweets=df_pos['hashtags_written_out'].values
all_tweets = np.concatenate((neg_tweets, pos_tweets), axis=0)


# In[22]:


vocab = build_filter_vocab(all_tweets, 4)
tokenized_vocab = build_tokenized_vocab(vocab)
print('vocab size:', len(tokenized_vocab))
token_tweets_neg = []
for tweet in neg_tweets:
    work_tweet = []
    for word in tweet.split():
        if word in tokenized_vocab:
            work_tweet.append(tokenized_vocab[word])
        else:
            work_tweet.append(tokenized_vocab['<UNK>'])
    token_tweets_neg.append(work_tweet)
token_tweets_pos = []
for tweet in pos_tweets:
    work_tweet = []
    for word in tweet.split():
        if word in tokenized_vocab:
            work_tweet.append(tokenized_vocab[word])
        else:
            work_tweet.append(tokenized_vocab['<UNK>'])
    token_tweets_pos.append(work_tweet)


# In[23]:


df_token_neg = pd.DataFrame(token_tweets_neg)
df_token_pos = pd.DataFrame(token_tweets_pos)
df_token_neg['label'] = 0
df_token_pos['label'] = 1
df_token = pd.concat([df_token_neg, df_token_pos])
df_token = df_token.fillna(tokenized_vocab['<PAD>'])
cols = [col for col in df_token.columns if col != 'label'] + ['label']
df_token = df_token[cols]
print(df_token)
train_loader, test_loader = create_dataloaders_from_df(df_token)


# In[26]:


train_model(model, train_loader, test_loader, 10)


# In[18]:




