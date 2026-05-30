# Model card

## Purpose
Predict synthetic support ticket priority: low, medium or high.

## Inputs
Text, channel, customer tier, product area, sentiment score and account age.

## Model
TF-IDF for text, one-hot encoding for categories, scaling for numeric features, Logistic Regression classifier.

## Metrics
Baseline macro F1: 0.195

Final model macro F1: 0.9419

## Limitations
The dataset is synthetic. The model demonstrates an engineering pipeline, not production readiness.
