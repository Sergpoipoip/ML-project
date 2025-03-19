# Emotion Detection in Movie-Dialogues: A Comparative Analysis of Pre-trained Models

## Abstract
This project evaluates and compares three pre-trained transformer-based models for emotion detection in movie dialogues: DistilBERT Go Emotions, Bhadresh's DistilBERT Emotion, and DistilRoBERTa Emotion. Using utterances from the Cornell Movie-Dialogs Corpus, the study analyzes each model's performance through manual annotation and comparative metrics. The findings reveal significant differences in accuracy, with RoBERTa demonstrating superior overall performance (66.4%) compared to DistilBERT Go Emotions (38.8%) and Bhadresh's model (25.6%). This report discusses implications for emotion recognition in conversational text and highlights the strengths and weaknesses of each approach.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Dataset](#dataset)
   - [Pre-trained Models](#pre-trained-models)
   - [Data Preprocessing](#data-preprocessing)
   - [Evaluation Approach](#evaluation-approach)
3. [Implementation](#implementation)
4. [Results and Analysis](#results-and-analysis)
   - [Overall Performance](#overall-performance)
   - [Per-emotion Performance](#per-emotion-performance)
   - [Confusion Matrix Analysis](#confusion-matrix-analysis)
   - [Model-specific Findings](#model-specific-findings)
5. [Discussion](#discussion)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction
Emotion detection in text is a challenging natural language processing task with applications across human-computer interaction, sentiment analysis, and conversational AI. This project investigates how effectively pre-trained transformer models can identify emotions in movie dialogues, which present unique challenges due to their dramatic nature, implied context, and varied emotional expressions.

The project compares three different transformer-based models, each trained on different emotion taxonomies and datasets. The comparison aims to determine which approach is most effective for identifying emotions in conversational text from movies and to understand the strengths and limitations of each model.

## Methodology

### Dataset
The project utilized the Cornell Movie-Dialogs Corpus, comprising 220,579 conversational exchanges between characters from 617 movies. This dataset was chosen for its rich emotional content and conversational structure. The dataset contains a large number of utterances representing natural dialogue in dramatic contexts.

### Pre-trained Models
Three pre-trained transformer models were selected for comparison:

1. **DistilBERT Go Emotions** (`joeddav/distilbert-base-uncased-go-emotions-student`): A model fine-tuned on the Google's GoEmotions dataset, which covers 28 emotion categories including more nuanced emotions like "curiosity" and "realization."

2. **Bhadresh's DistilBERT Emotion** (`bhadresh-savani/distilbert-base-uncased-emotion`): A model fine-tuned for emotion classification with 6 basic emotion categories: anger, fear, joy, love, sadness, and surprise. The model does not include a "neutral" category.

3. **DistilRoBERTa Emotion** (`j-hartmann/emotion-english-distilroberta-base`): A DistilRoBERTa-based model fine-tuned for emotion detection with 7 categories: anger, disgust, fear, joy, neutral, sadness, and surprise.

These models were chosen because they represent different approaches to emotion classification, with varying emotion taxonomies and underlying architectures.

### Data Preprocessing
The preprocessing pipeline included:

1. Extracting utterances from the corpus
2. Fixing contractions for better tokenization
3. Removing special characters while preserving punctuation
4. Tokenizing and truncating to the maximum allowed length for each model (510 tokens)
5. Removing empty utterances
6. Normalizing the dataset to ensure all models were evaluated on the same set of utterances

### Evaluation Approach
The evaluation followed these steps:

1. Processing the entire corpus with each model to generate emotion predictions
2. Creating a stratified sample of 250 utterances for manual annotation, ensuring representation of different emotions and confidence levels
3. Manual annotation of the sample to create a gold standard, where a "neutral" category was added by the researcher when annotating utterances that did not fit into any of Bhadresh's model's 6 emotion categories
4. Computing evaluation metrics including accuracy, precision, recall, and F1-score
5. Generating per-emotion performance statistics
6. Creating confusion matrices to visualize classification errors

## Implementation
The implementation was carried out in Python using the Hugging Face Transformers library. The key components included:

1. Loading and preprocessing the Cornell Movie-Dialogs Corpus
2. Setting up emotion classification pipelines for each model
3. Processing utterances in batches to improve efficiency
4. Saving model predictions for subsequent analysis
5. Generating stratified samples for human annotation
6. Computing evaluation metrics using scikit-learn
7. Visualizing results using matplotlib and seaborn

GPU acceleration was utilized when available to speed up processing, and batch processing was implemented to handle the large dataset efficiently.

## Results and Analysis

### Overall Performance
The overall accuracy results revealed substantial differences between the three models:

- RoBERTa: 66.40% accuracy
- DistilBERT Go Emotions: 38.80% accuracy
- Bhadresh's DistilBERT: 25.60% accuracy

RoBERTa significantly outperformed the other models, achieving more than double the accuracy of Bhadresh's model. When excluding neutral utterances, Bhadresh's model performance improved to 49.23%, indicating particular difficulty with the neutral category that was not part of its original training taxonomy but was added during manual annotation.

### Per-emotion Performance
Examining per-emotion accuracy reveals interesting patterns across models:

**DistilBERT Go Emotions (top 5):**
- amusement: 100.0% (3 samples)
- grief: 100.0% (3 samples)
- realization: 100.0% (2 samples)
- disgust: 100.0% (3 samples)
- remorse: 85.7% (7 samples)

**Bhadresh's DistilBERT (top 5):**
- joy: 75.8% (33 samples)
- anger: 60.7% (28 samples)
- sadness: 50.0% (24 samples)
- love: 36.4% (11 samples)
- fear: 28.6% (14 samples)

**RoBERTa (top 5):**
- disgust: 100.0% (6 samples)
- surprise: 85.0% (20 samples)
- neutral: 69.6% (125 samples)
- fear: 64.3% (14 samples)
- joy: 63.2% (38 samples)

These results demonstrate that models have varying strengths for different emotions. DistilBERT Go Emotions performs well on specific nuanced emotions but struggles with more common ones. RoBERTa shows more balanced performance across emotion categories.

### Confusion Matrix Analysis
The confusion matrices provide deeper insights into classification patterns:

1. **DistilBERT Go Emotions**: Shows many misclassifications across its 28 emotion categories. The model particularly struggles with correctly identifying neutral utterances (only 4.8% accuracy), often misclassifying them as other emotions. However, it performs well on several specific emotions like amusement, grief, and disgust.

2. **Bhadresh's DistilBERT**: Demonstrates a strong bias toward classifying utterances within its limited taxonomy (6 emotions), with a particular tendency to misclassify many utterances as "joy". The model is forced to assign one of its 6 emotion categories to utterances that were manually annotated as "neutral", which explains its poor performance when evaluated against the manual annotations that include a neutral category. This taxonomic mismatch is a significant factor in its low overall accuracy.

3. **RoBERTa**: Shows the most balanced performance across emotion categories. It achieves strong performance on neutral utterances (69.6%), which constitutes a large portion of the dataset. The model shows fewer extreme misclassifications compared to the other models.

### Model-specific Findings

**DistilBERT Go Emotions**:
- Strengths: Fine-grained emotion detection; high accuracy for specific emotions
- Weaknesses: Poor overall accuracy; struggles with neutral utterances; possibly overtrained on certain emotion categories

**Bhadresh's DistilBERT**:
- Strengths: Good performance on joy and anger; improves substantially when neutral utterances are excluded
- Weaknesses: Lowest overall accuracy; complete failure on neutral class (which is not part of its original taxonomy); strong bias toward misclassifying utterances as joy

**RoBERTa**:
- Strengths: Highest overall accuracy; balanced performance across emotion categories; handles neutral utterances well
- Weaknesses: Still shows some confusion between related emotions (e.g., anger and disgust)

## Discussion
The significant performance differences between models can be attributed to several factors:

1. **Emotion taxonomy**: The DistilBERT Go Emotions model's fine-grained taxonomy (28 categories) may create more opportunities for confusion compared to the more limited category systems of the other models. While this provides more nuanced classifications, it appears to reduce overall accuracy on general emotional content.

2. **Training data**: RoBERTa's superior performance suggests its training data may be more relevant to conversational text or may include content similar to movie dialogues.

3. **Neutral detection**: The handling of neutral utterances significantly impacted performance comparisons. Bhadresh's model was evaluated on a category ("neutral") that it was not trained to recognize, which explains its 0% accuracy for this class. Meanwhile, DistilBERT Go Emotions rarely predicts neutral at all despite having the category in its taxonomy.

4. **Architectural differences**: The underlying differences between RoBERTa and DistilBERT architectures may contribute to performance differences, with RoBERTa potentially providing better contextual understanding.

The results highlight the challenge of emotion detection in conversational text, particularly for more nuanced or subtle emotional expressions. The evaluation methodology of adding a "neutral" category during manual annotation for Bhadresh's model (which lacks this category) impacts the comparability of results but reflects real-world application needs where texts without clear emotional content must be handled.

## Conclusion
This project provides valuable insights into the performance of pre-trained transformer models for emotion detection in movie dialogues. RoBERTa emerges as the most effective model with 66.4% accuracy, significantly outperforming both DistilBERT variants.

For applications requiring emotion detection in conversational text, the RoBERTa model (`j-hartmann/emotion-english-distilroberta-base`) appears to be the most reliable choice among those tested. However, if detection of specific nuanced emotions is required, the DistilBERT Go Emotions model may be valuable despite its lower overall accuracy.

Future work could explore ensemble approaches combining the strengths of multiple models, fine-tuning these models specifically on movie dialogue data, or investigating the impact of conversational context on emotion detection accuracy. Additionally, addressing the discrepancy between model taxonomies and evaluation needs (particularly regarding neutral utterances) would be valuable for more standardized comparisons.

This comparative analysis demonstrates that while pre-trained transformer models show promising capabilities for emotion detection, their performance varies significantly depending on their architecture, training data, and emotion taxonomy. Careful model selection based on specific application requirements remains essential for effective emotion detection in conversational AI systems.

## References
1. [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
2. [Hugging Face Transformers Library](https://huggingface.co/docs/transformers/index)
3. [Bhadresh-Savani, "DistilBERT-base-uncased-emotion"](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
4. [J. Hartmann, "Emotion-English-DistilRoBERTa-base"](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
5. [Joe Davison, "DistilBERT-base-uncased-go-emotions-student"](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)