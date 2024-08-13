# NLP_Spam_Detection

## Project Description
This project is an implementation of a spam detection system using Natural Language Processing (NLP) techniques. The system is designed to classify SMS messages as either "spam" or "ham" (not spam) by leveraging machine learning algorithms. The project primarily focuses on the use of the Naive Bayes classifier, a common algorithm for text classification tasks.

## Dataset
The dataset used in this project is the SMS Spam Collection Dataset, which can be found on Kaggle. It consists of a collection of SMS messages that are labeled as either "spam" or "ham."

- **Source:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Shape:** The dataset contains 5,572 SMS messages.

## Features
- **Text Preprocessing:** The raw text data is cleaned by removing noise such as special characters, numbers, and unnecessary spaces.
- **Vectorization:** The cleaned text data is transformed into numerical vectors using techniques like Count Vectorization and TF-IDF (Term Frequency-Inverse Document Frequency).
- **Modeling:** A Naive Bayes classifier is trained on the vectorized text data to classify the messages.

## Usage
1. **Import Dependencies:** The notebook begins with importing necessary libraries such as Pandas, NumPy, scikit-learn, and others.
2. **Data Loading:** The dataset is loaded from Google Drive into a Pandas DataFrame.
3. **Data Preprocessing:** Text data is cleaned and vectorized.
4. **Model Training:** The Naive Bayes model is trained using the training data.
5. **Evaluation:** The model's performance is evaluated using metrics like accuracy, confusion matrix, and classification report.

## Results
The project demonstrates the effectiveness of using Naive Bayes for spam detection, with a high accuracy rate achieved during evaluation.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Spacy
- Seaborn
- Matplotlib

## How to Run
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook.

## Conclusion
This project provides a hands-on approach to building a spam detection system using NLP techniques. It showcases the importance of text preprocessing, feature extraction, and model selection in achieving high classification accuracy.
