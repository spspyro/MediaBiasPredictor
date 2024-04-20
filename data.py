import nltk
import json
import time
import torch
import random
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import nltk
nltk.download('punkt')

class ArticleBiasDS(Dataset):
    def __init__(self, datapath, size):
        '''
        Costume dataset class for Media Bias Prediction dataset.
        Args:
            datapath:
                pathway to the dataset.
            size:
                size limit of the dataset.
                google colab runs out the RAMafter loading in 
                120,000 datapoints.
        '''
        super().__init__()
        self.data = []
        self.label = []
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.cuda()
        for params in self.bert.parameters():
            params.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        file_list = [json_file for json_file in datapath.glob("*.json")]
        random.shuffle(file_list)
        for idx in tqdm(range(size)):
            json_file = file_list[idx]
            with open(json_file) as file:
                data_pt = json.load(file)
                content = self._process_content(data_pt['content'])
                content = {k:v.cuda() for k, v in content.items()}
                token_embedding = self.bert(**(content))[0]
                attention_mask = content['attention_mask']
                token_embedding = token_embedding[attention_mask == 1].cpu()
                self.label.append(data_pt['bias'])
                self.data.append(token_embedding)

    def _process_title(self, title):
        '''
        Turn title into vector embedding.
        Args:
            title:
                dictionary of title, transformed by bert-tokenizer.
        Return:
            encoded input representing the title.
        '''
        encoded_input = self.tokenizer(
            title,
            padding=True,
            truncation=True,
            return_tensors="pt")
        return encoded_input

    def _process_content(self, content):
        '''
        Turn content into vector embedding.
        Args:
            content:
                dictionary of content, transformed by bert-tokenizer.
        Return:
            encoded input representing the content.
        '''
        content = sent_tokenize(content)
        encoded_input = self.tokenizer(
            content,
            padding=True,
            truncation=True,
            return_tensors="pt")
        return encoded_input

    def __getitem__(self, idx):
        '''
        needed for dataset class declaration.
        '''
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        '''
        needed for dataset class declaration.
        '''
        return len(self.data)

def split_dataset(dataset):
    '''
    Split dataset into training, validation, and testing.
    Args:
        dataset:
            ArticleBiasDS that contain all the datapoints.
    Return:
        dataset, dataset, dataset: train, val, test dataset
    '''
    train = 0.6
    val = 0.2
    dataset_len = len(dataset)
    partition = [int(dataset_len*0.6), int(dataset_len*0.2), dataset_len-int(dataset_len*0.6)-int(dataset_len*0.2)]
    train, val, test = random_split(dataset, partition)
    return train, val, test

def collate_fn(batch):
    '''
    Costumize batching function for the dataloader.
    Args:
        batch:
            list of datapoints
    Return:
        list, tensor: list of tensor representing the content, tensor of labels
    '''
    batch_data = []
    batch_target = []
    for data in batch:
        batch_data.append(data[0])
        batch_target.append(data[1])
    return batch_data, torch.tensor(batch_target)

def get_loader(dataset):
    '''
    Convert dataset into training, validation, and testing loader.
    Args:
        dataset:
            ArticleBiasDS that contain all the datapoints.
    Return:
        dataloader, dataloader, dataloader: train, val, test dataloader
    '''
    train, val, test = split_dataset(dataset)
    train_loader = DataLoader(train, batch_size=32, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=32, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
