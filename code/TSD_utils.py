
from transformers import AutoTokenizer, BatchEncoding, AutoModelForTokenClassification, Trainer, TrainingArguments
from typing import List, Dict
#from sklearn.metrics import f1_score, precision_score, recall_score
import os
import torch


from torch import nn
import numpy as np
import ast
import pandas as pd
import itertools
from statistics import mean



# INITIALIZING MODEL AND TOKENIZER

def create_model_tokenizer_folder(model_checkpoint:str, destination_folder:str, id2label:Dict):
    """Loads a tokenizer and model for tokenclassification, and creates a folder in which the trained model is to be saved.
    Param str model_checkpoint: the model checkpoint with which the model can be imported from the huggingface library.
        works at least for the following: facebook/xlm-v-base, xlm-roberta-base, bert-base-multilingual-cased, distilbert-base-multilingual-cased, ai-forever/mGPT
    Param str destination_folder: the folder in which we want the model to be saved..
    id2label: a dictionary mapping ids to labels"""

    #Create label2id, the revesre of id2label
    label2id = {v: k for k, v in id2label.items()}

    #load tokenizer and add pad token
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #load model
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                            num_labels = len(id2label),
                                                            id2label=id2label,
                                                            label2id=label2id,)

    destination_folder = f"{destination_folder}{model_checkpoint.replace('/', '_')}_trained_for_TSD"
    #Create the destination folder for this model, which gets the name of the model checkpoint followed by 'trained_for_TSD'
    try:
        os.mkdir(destination_folder)
    except FileExistsError:
        pass

    return model, tokenizer, destination_folder


#PREPROCESSING

class TSDdataset(torch.utils.data.Dataset):
    #inspired by https://huggingface.co/transformers/v4.1.1/custom_datasets.html
    #Creates a dataset object that holds encodings as well as gold labels
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def align_tokens_and_annotation_labels(tokenized: BatchEncoding, annotations, pad_token=-100, max_len=512):
    #inspired by https://github.com/LightTag/sequence-labeling-with-transformers/blob/master/notebooks/how-to-align-notebook.ipynb
    """Aligning tokens with annotation labels (I/O), given a BatchEncoding (a tokenized text) and a list of character offsets.
    Param BatchEncoding tokenized: a tokenized sentence that has been tokenized by a FastTokenizer.
    Param annotations: a list or string that indicates the character indices that are toxic spans. """
    #create aligned_labels as a list of 0's, length equals the number of tokens
    aligned_labels = [0] * len([ids for ids in tokenized.type_ids if id != 0])
    #convert annotation from str to list
    if type(annotations) == str:
        spanlist = ast.literal_eval(annotations)

    #iterate over indices in the span list
    for char_ix in spanlist:
        #Find the corresponding token index
        token_ix = tokenized.char_to_token(char_ix)
        #Change the value in aligned_labels to 1 (I)
        if token_ix is not None: # White spaces have no token and will return None
          aligned_labels[token_ix] = 1
    #aligned_labels now looks like a list of 0s and 1s, of equal length as the tokens
    #we add the pad_token to the list until the list has length max_len
    n_pad_tokens = max_len-len(aligned_labels)
    aligned_labels += [pad_token]*n_pad_tokens

    return aligned_labels

def preprocess_TSD (file_dict:Dict, tokenizer:AutoTokenizer, max_len = 512):
    """Preprocesses a list of .csv files into a dictionary TSDdataset objects.
    Param files: a list of .csv files to be preprocessed
    tokenizer: an Autotokenizer (the output of create_model_tokenizer_folder)"""
    TSD_datasetdict = {}

    #open file and extract the list of texts and list of spans
    for data_type, file in file_dict.items():
        df = pd.read_csv(file)
        texts = list(df['text'])
        spans = list(df['spans'])

        #tokenize the texts so that we can create a list of gold labels that is aligned with the tokens
        raw_encodings = tokenizer(texts)
        all_labels = []
        #enumerate over spans, find the corresponding encoding and create a list of gold labels on a token level
        for i, span in enumerate(spans):
          encoding = raw_encodings[i]
          aligned_labels = align_tokens_and_annotation_labels(encoding, span)
          all_labels.append(aligned_labels)

        #create encodings, this time with truncation and padding
        encodings = tokenizer(texts, truncation = True, max_length = max_len, padding = 'max_length')
        #create a TSDdataset with the encodings and matched align_tokens_and_annotation_labels
        TSD_datasetdict[data_type] = TSDdataset(encodings, all_labels)

    return TSD_datasetdict


#### EVALUATION




def convert_token_predictions_to_spans(binary_predictions, test_datasetdict, pad_token=-100):
    all_spans = []
    for tweet_idx, predictions in enumerate(binary_predictions):
        tweet_spans = []
        for token_idx, pred_token in enumerate(predictions):
          if test_datasetdict.labels[tweet_idx][token_idx] != pad_token:
            if pred_token == 1:
              token_span = test_datasetdict.encodings.token_to_chars(tweet_idx, token_idx)
              tweet_spans += ([idx for idx in range(token_span.start, token_span.end)])
          else:
            break
        all_spans.append(tweet_spans)

    return all_spans






def calculate_evaluation_metrics(gold, pred):
    all_precision = []
    all_recall = []
    all_f1 = []

    for tweet_gold, tweet_pred in zip(gold,pred):
        if type(tweet_gold) == str:
            tweet_gold = ast.literal_eval(tweet_gold)


        if tweet_gold == [] and tweet_pred == []:
            precision, recall, f1 = 1,1,1
        elif tweet_gold == [] or tweet_pred == []:
            precision, recall, f1 = 0,0,0

        else:
            TP = len([char for char in tweet_pred if char in tweet_gold])
            FP = len([char for char in tweet_pred if char not in tweet_gold])
            FN = len([char for char in tweet_gold if char not in tweet_pred])

        precision_I = (TP/(TP+FP))
        recall_I = (TP/(TP+FN))
        try:
            f1 = 2*precision*recall/(precision+recall)
        except ZeroDivisionError:
            f1= 0

    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)

    return {
          'precision': mean(all_precision),
          'recall': mean(all_recall),
          'f1': mean(all_f1),
          }
