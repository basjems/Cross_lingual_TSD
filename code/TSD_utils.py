
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

def preprocess_TSD (file_dict:Dict, tokenizer:AutoTokenizer, max_len = 512, pad_token=-100):
    """Preprocesses a dict of .csv files into a dictionary TSDdataset objects.
    Param files: a dict holding paths to .csv files to be preprocessed. Keys should be 'train' 'dev' and/or 'test', values should be paths to respective data file.
    tokenizer: an AutoTokenizer with which we want to preprocess the data."""
    TSD_datasetdict = {}

    #open file and extract the list of texts and list of spans
    for data_type, file in file_dict.items():
        df = pd.read_csv(file)
        texts = list(df['text'])
        spans = list(df['spans'])

        #tokenize the texts so that we can create a list of gold labels that is aligned with the tokens
        encodings = tokenizer(texts, truncation = True, max_length = max_len, padding = 'max_length')
        labels = [align_tokens_and_annotation_labels(tokenized, annotation, max_len, pad_token) for tokenized, annotation in zip(tokenizer(texts).encodings, spans)]
        TSD_datasetdict[data_type] = TSDdataset(encodings, labels)

    return TSD_datasetdict


#### EVALUATION




def convert_token_predictions_to_spans(binary_predictions, test_datasetdict, text_list, pad_token=-100):

  all_spans = []
  for tweet_idx, predictions in enumerate(binary_predictions):
    cur_tweet = text_list[tweet_idx]
    tweet_spans = []
    toxic_words = []
    for token_idx, pred_token in enumerate(predictions):
      if test_datasetdict.labels[tweet_idx][token_idx] != pad_token:
        if pred_token == 1:
          cur_tweet = text_list[tweet_idx]

          token_span = test_datasetdict.encodings.token_to_chars(tweet_idx, token_idx)
          tweet_spans += [idx for idx in range(token_span.start, token_span.end) if cur_tweet[idx] != ' ']

      else:
        break

    all_spans.append(tweet_spans)


  return all_spans




def calculate_evaluation_metrics(gold, pred):
    """Calculates averaged f1, precision and recall for TSD, via the metric defined by Pavlopoulos et al's "SemEval-2021 Task 5: Toxic Spans Detection" (2021)
    Param gold: the gold labels (list of lists of character indices)
    Param pred: system predictions (list of lists of character indices) 
    Returns: a dictionary holding precision, recall and f1"""
    all_precision = []
    all_recall = []
    all_f1 = []

    #iterate over all sublists in the gold and pred lists
    for tweet_gold, tweet_pred in zip(gold,pred):
        #gold may hold strings instead of sublists. If so, convert to list.
        if type(tweet_gold) == str:
            tweet_gold = ast.literal_eval(tweet_gold)

        #If there are no toxic spans and none predicted, set precision, recall and f1 to 1
        if tweet_gold == [] and tweet_pred == []:
            precision, recall, f1 = 1,1,1
        #else, if either gold or pred holds no spans, set precision, recall and f1 to 0
        elif tweet_gold == [] or tweet_pred == []:
            precision, recall, f1 = 0,0,0

        #else, count the number of true positives, false positives, false negatives
        else:
            TP = len([char for char in tweet_pred if char in tweet_gold])
            FP = len([char for char in tweet_pred if char not in tweet_gold])
            FN = len([char for char in tweet_gold if char not in tweet_pred])

        #calculate precision, recall, f1
        precision = (TP/(TP+FP))
        recall = (TP/(TP+FN))
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
