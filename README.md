# Sentiment Analysis with Logistic Regression, SVM, and Random Forest

This project implements a sentiment analysis pipeline using Logistic Regression, Support Vector Machines (SVM), and Random Forest models. The dataset used contains text data, along with some additional categorical features.

## Project Structure

- **SentimentAnalysis_LogisticRegression.ipynb**: The main Jupyter notebook that contains the code for data preprocessing, model training, evaluation, and hyperparameter tuning.
- **sentimentdata.csv.xls**: The dataset used in this analysis.
- **README.md**: This file, providing an overview of the project.

## Project Workflow

### 1. Data Loading and Preprocessing
- The dataset is loaded from Google Drive and cleaned, including removing URLs and non-word characters.
- Missing values in the 'sentiment' column are handled by dropping rows with NaN values.
- Categorical features such as 'Time of Tweet,' 'Age of User,' and 'Country' are encoded using `LabelEncoder`.

### 2. Feature Extraction
- Text data is converted into numerical features using the `TfidfVectorizer`.
- Categorical features are combined with the TF-IDF matrix to form the final feature set.

### 3. Model Training and Evaluation
- Three models are trained:
  - **Logistic Regression**: Used as a baseline model.
  - **Support Vector Machine (SVM)**: Trained with a linear kernel.
  - **Random Forest**: Trained with 100 estimators initially, and then optimized using GridSearchCV.
  
- Each model's performance is evaluated using accuracy, precision, recall, and F1-score metrics.

### 4. Hyperparameter Tuning
- The Random Forest model undergoes hyperparameter tuning using `GridSearchCV` to find the best combination of parameters.

## Results
- **Random Forest** achieved the best overall accuracy and balanced performance across different sentiment classes after hyperparameter tuning.
- **Logistic Regression** and **SVM** performed slightly worse but still provided valuable insights, with SVM having the lowest accuracy.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- re
- google-colab

### Installation
To run the project locally, ensure you have the necessary packages installed:

```bash
pip install pandas scikit-learn google-colab
```

### Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
```

2. Navigate to the project directory:

```bash
cd sentiment-analysis
```

3. Open the Jupyter notebook and run all cells to see the results:

```bash
jupyter notebook SentimentAnalysis_LogisticRegression.ipynb
```
