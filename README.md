# Job Listing Integrity Investigation  
#### "Leveraging NLP and machine learning to detect fraudulent job postings and protect platform trust"

## About  
This project applies Natural Language Processing and machine learning techniques to distinguish between genuine and fraudulent job postings. By analyzing both text content and metadata from ~18K postings, it offers a repeatable, data‑driven approach for enhancing online job platform integrity.

## Data Source  
- **Dataset Name:** [Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
- **Provider:** University of the Aegean – Laboratory of Information & Communication Systems Security  
- **Records:** 17,880 job postings, 18 features (textual + categorical)  
- **Target Variable:** `fraudulent` (binary)

## Purpose & Business Context  
Online job platforms face growing challenges with fraudulent postings, eroding trust among seekers and recruiters. Even a small fraction of false postings can damage credibility and user engagement. This project addresses the need for an automated, scalable pipeline to identify fake listings, reduce user risk, and maintain platform reliability.

## Solution Overview  
We designed a pipeline that:  
- Cleans and preprocesses structured and unstructured posting data.  
- Uses NLP (tokenization, stopword removal, lemmatization) to prepare textual features.  
- Applies **Word2Vec** embeddings with dimensionality reduction (PCA).  
- Integrates non‑text features for holistic model input.  
- Handles severe class imbalance using **SMOTE**.  
- Trains multiple classification models (Logistic Regression, Random Forest, XGBoost, SVC, Stacking Classifier) with hyperparameter tuning.  
- Evaluates performance to identify the most effective detection method.

## Tech Stack & Tools  

| Tool / Library     | Purpose                         | Why Chosen                         |
|--------------------|---------------------------------|-------------------------------------|
| Python             | Data cleaning, modelling        | Flexibility, strong ML ecosystem   |
| Pandas, NumPy      | Data wrangling & manipulation   | Industry standard for tabular data |
| scikit-learn       | ML modelling, evaluation        | Robust classification algorithms   |
| Gensim (Word2Vec)  | Text vectorization               | Captures semantic meaning          |
| Matplotlib, Seaborn| Data visualization              | Clear, publication‑ready charts    |
| Jupyter Notebook   | Development environment         | Interactive EDA & documentation    |

## Data Pipeline / Workflow  

1. **Data Cleaning:** Handle missing values, standardize categorical values, and parse location metadata.  
2. **Text Preprocessing:** Tokenization, lowercasing, stopword removal, stemming, lemmatization.  
3. **Vectorization:** Word2Vec embeddings for semantic representation.  
4. **Dimensionality Reduction:** PCA to retain 95% variance while reducing computational cost.  
5. **Feature Integration:** Combine textual and non‑textual features.  
6. **Class Balancing:** Apply SMOTE to address minority fraudulent class (4.8% of postings).  
7. **Model Training:** Compare Logistic Regression, Random Forest, SVC, XGBoost, and Stacking.  
8. **Evaluation:** Balanced accuracy, precision, recall, F1‑score.

## Business Impact & Key Results  

- **Fraudulent Listing Detection:** Achieved **~80% balanced accuracy** with the Stacking Classifier on unseen test data.  
- **High Precision:** Ability to flag suspicious postings while minimizing false alarms, reducing platform reputation risks.  
- **Feature Insights:**  
  - Missing company logo, vague titles, or unspecified employment type correlated strongly with fraud.  
  - Certain educational requirements and remote job listings had higher fraud rates.  
- **Platform Benefits:** Framework can integrate into job boards to automatically assess posting legitimacy, improving user trust and retention.

## Code Highlights & How to Run  

**Highlights**  
- Modular preprocessing functions for both text and categorical columns.  
- Word2Vec embedding fine‑tuning for better semantic capture.  
- Integration of PCA for efficiency without major loss of information.  
- SMOTE oversampling seamlessly added to the ML workflow.  

**How to Run**  
1. Clone this repository:  
   ```bash
   git clone https://github.com/SnehaEkka/BA820-Job-Listing-Integrity-Investigation.git
   cd BA820-Job-Listing-Integrity-Investigation
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
4. Open and run `BA820 B1_04 - Text Classification.ipynb` cell‑by‑cell.  

## Future Improvements  
- Compare more embedding alternatives (e.g., domain‑trained GloVe or BERT models).  
- Deploy as an API for real‑time fraud detection on job platforms.  
- Integrate salary normalization and additional external data (industry trends, location risks).  
- Expand the dataset with cross‑platform postings to increase robustness.

## Alignment to Career Vision  
This project reflects my focus on **analytics engineering that bridges technical capability and business value**—in this case, operationalizing NLP models to directly improve an online platform’s trust and efficiency.

## Coursework & Contributors:
- Completed as part of BA820 - Unsupervised Machine Learning (Boston University MSBA), with emphasis on NLP, classification, and model evaluation.
- Contributors:
  - **Dian Jin:** EDA insights, initial data loading/cleaning, SVC model.
  - **Jenil Shah:** EDA on categorical variables, PCA, XGBoost, hyperparameter tuning.
  - **Mingze Wu:** Categorical data processing, tokenization/vectorization, Random Forest model.
  - **Sneha Ekka:** Text data cleaning, NLP preprocessing functions, feature integration, SMOTE handling, Stacking classifier, Word2Vec + feature merge explanation.
  - **Tanvi Sheth:** EDA on text columns, logistic regression model, result summaries, challenges/limitations.

## Additional Resources  
- 📄 [Final Project Report](BA820-B1_04-Project-Report-Final.pdf)  
- 📊 [Job Listing Fraud Detection - Presentation Deck](https://www.canva.com/design/DAF-f3ikgxk/TCh5JXHAoz-DvL6mkkLuJQ/view?utm_content=DAF-f3ikgxk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=heb121796c6)  
