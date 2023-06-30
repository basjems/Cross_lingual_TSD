# Cross_lingual_TSD

This repository contains the code and data used for training models for cross-lingual Toxic Spans Detection, as part of a thesis project for the master Text Mining at Vrije Universiteit Amsterdam.

Folder 'data' holds:
- annotated_test_set_hashtags_removed.csv: A Dutch Twitter test set annotated for toxic spans, without hashtags. This test set is a subset of the DALC corpus (https://aclanthology.org/2021.woah-1.6/)
- annotated_test_set_with_hashtags.csv: the same test set, only before removing hashtags.
- individual_annotator_results.csv: holds the annotations of individual annotators for the same test set.

Folder 'code' holds:
- fine_tune_TSD.ipynb: a notebook for fine-tuning models for Toxic Spans Detection.
- Predict_TSD.ipynb: a notebook for predicting the test data and evaluating the predictions.
Both notebooks are designed for use in Google Colab and require that the user gives access to Google Drive.  
