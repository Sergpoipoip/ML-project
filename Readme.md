# Emotion Detection in Movie-Dialogues
An individual project carried out as part of the final examination of "Machine Learning for the Arts & Humanities" course (master programme "Digital Humanities and Digital Knowledge", University of Bologna) held by Giovanni Colavizza.

## Project Overview
This project evaluates and compares three pre-trained transformer-based models for emotion detection in movie dialogues:
- DistilBERT Go Emotions
- Bhadresh's DistilBERT Emotion
- DistilRoBERTa Emotion

Using utterances from the Cornell Movie-Dialogs Corpus, the study analyzes each model's performance through manual annotation and comparative metrics. The findings show significant differences in accuracy, with RoBERTa demonstrating superior overall performance (66.4%) compared to DistilBERT Go Emotions (38.8%) and Bhadresh's model (25.6%).

## Project Structure

The project is organized as follows:

```
ML-Colab-1/
├── Roberta/
│   ├── roberta_emotion_utterances.csv                  # Raw predictions from RoBERTa model; zip in GitHub repo
│   ├── roberta_emotion_utterances_normalized.csv       # Normalized predictions; zip in GitHub repo
│   ├── roberta_emotion_annotation_spreadsheet.csv      # Data used for manual annotation
│   ├── roberta_emotion_annotation_spreadsheet_v1.csv   # Annotated data with ground truth
│   ├── roberta_confusion_matrix.png                    # Visualization of model performance
│   └── roberta_evaluation_metrics.txt                  # Detailed performance metrics
├── Distilbert/
│   ├── distilbert_go_emotions_utterances.csv           # Raw predictions from DistilBERT Go Emotions; zip in GitHub repo
│   ├── distilbert_go_emotions_utterances_normalized.csv# Normalized predictions; zip in GitHub repo
│   ├── distilbert_emotion_annotation_spreadsheet.csv   # Data used for manual annotation
│   ├── distilbert_emotion_annotation_spreadsheet_v1.csv # Annotated data with ground truth
│   ├── distilbert_confusion_matrix.png                 # Visualization of model performance
│   └── distilbert_evaluation_metrics.txt               # Detailed performance metrics
├── Bhadresh/
│   ├── bhadresh_emotion_utterances.csv                 # Raw predictions from Bhadresh's DistilBERT; zip in GitHub repo
│   ├── bhadresh_emotion_utterances_normalized.csv      # Normalized predictions; zip in GitHub repo
│   ├── bhadresh_emotion_annotation_spreadsheet.csv     # Data used for manual annotation
│   ├── bhadresh_emotion_annotation_spreadsheet_v1.csv  # Annotated data with ground truth
│   ├── bhadresh_confusion_matrix.png                   # Visualization of model performance
│   └── bhadresh_evaluation_metrics.txt                 # Detailed performance metrics
├── movie-corpus/
│   └── [Cornell Movie-Dialogs Corpus files]            # Original dataset files; this folder with all related files will be created the first time the code is run
└── Report.md                                           # Final project report with findings
```

## Implementation Details

This project was implemented in Google Colab, making use of its GPU acceleration capabilities to process the large Cornell Movie-Dialogs Corpus efficiently. The implementation uses the Hugging Face Transformers library to access the pre-trained models:

1. **DistilBERT Go Emotions** (`joeddav/distilbert-base-uncased-go-emotions-student`): Covers 28 emotion categories including nuanced emotions like "curiosity" and "realization."

2. **Bhadresh's DistilBERT Emotion** (`bhadresh-savani/distilbert-base-uncased-emotion`): Includes 6 basic emotion categories: anger, fear, joy, love, sadness, and surprise.

3. **DistilRoBERTa Emotion** (`j-hartmann/emotion-english-distilroberta-base`): Covers 7 categories: anger, disgust, fear, joy, neutral, sadness, and surprise.

## Data Files

Each model directory contains the following files:

- **emotion_utterances.csv**: Raw output from running the model on the movie dialogue corpus
- **emotion_utterances_normalized.csv**: Processed dataset with standardized emotion labels
- **emotion_annotation_spreadsheet.csv**: Sample of utterances selected for manual annotation
- **emotion_annotation_spreadsheet_v1.csv**: Manually annotated data with ground truth labels
- **confusion_matrix.png**: Visual representation of the model's classification performance
- **evaluation_metrics.txt**: Detailed metrics including accuracy, precision, recall, and F1-score

## Key Findings

- **RoBERTa** achieved the highest overall accuracy (66.4%) and showed balanced performance across emotion categories
- **DistilBERT Go Emotions** performed well on specific nuanced emotions but struggled with common emotions and neutral utterances
- **Bhadresh's DistilBERT** showed a strong bias toward classifying utterances as "joy" and struggled with neutral utterances (which were not part of its original taxonomy)

## How to Use

To reproduce this analysis:

1. Clone this repository
2. Upload the notebook files to Google Colab
3. Connect to a GPU runtime in Google Colab
4. Run the cells in order to process the Cornell Movie-Dialogs Corpus with each model
5. The code will generate the CSV files, confusion matrices, and evaluation metrics found in each model's directory

## Requirements

- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Transformers (Hugging Face)
- PyTorch

## Acknowledgements

- [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Hugging Face Transformers Library](https://huggingface.co/docs/transformers/index)
- Models used:
  - [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
  - [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
  - [joeddav/distilbert-base-uncased-go-emotions-student](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)
