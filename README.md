## **Twitter Sentiment Analysis on Covid19, Monkeypox & Health**

**Project Scope**

In this project TF-IDF, BOW (Count Vectoriser) and Word2vec techniques were used. 

The primary focus of this project is the use of Word2vec pretained model namely; glove-twitter-50 for model building for binary classification. Where 0 is netural and 1 is positive.

TF-IDF and BOW were used for predictions on multiclass where 0 is neutral, 1 is positive and -1 is negative. The table of values shows the f1-scores when using Random Forest with TF-IDF and BOW;
|  Model| f1-score- macro |f1-score- micro | f1-score- weighted
| ----------- | ----------- | ----------- | ----------- |
Random Forest with BOW |0.66|0.77|0.73|
Random Forest with TF-IDF |0.64|0.76|0.72|

The following can be observed from the test data using Word2Vec with the following models as shown in the table below;
|  Model| AUC Score |
| ----------- | ----------- |
Logistic Regression with PCA   | 0.74
KNN with PCA   | 0.72
KNN with without PCA   | 0.67
Random Forest with PCA  | 0.77
Random Forest with PCA and undersampler | 0.71
Random Forest without PCA  | 0.73
Random Forest without PCA and with undersampler | 0.70

## **Project Explanation**  

1. The project begins with fundamentals of Tweepy in tweet_sentiment.ipynb

2. Sentiment Analysis using VADER in tweet_sentiment_2.ipynb

3. Text classification in tweet_sentiment_3.ipynb
    - vectorizer.pkl and vectorizer_tfidf.pkl contain the vectorisers developed for Count Vectoriser and TF-IDF respectively.
    - cf_3.pkl and cf_3_tfidf.pkl contain the models developed using Count Vectoriser and TF-IDF respectively.

4. Predicting unseen data in tweet_sentiment_3._predict.ipynb
    - predict_set.csv contains the unseen data used for prediction
    - predicted.csv and predicted.xlsx are the outputs of the predictions made on predict_set.csv using the developed models and vectorisers.

5. Within the img folder contains the WordCloud imagery of the cleaned data, positive, neutral and negative tweets realting to Covid-19, Monkeypox and health.

### **What is word embeddings**

This is a technique which transforms individual words into a numerical representation of the word in within a vector space. By mapping each word to a vector, the vector is then learned in way that mimics that of a neural network. Below are popular word embeddings techniques;
- Bag of words
- TF-IDF
- Word2vec
- Glove embedding
- Fastext
- ELMO (Embeddings for Language models)

**What is Word2Vec?**

This is a word embedding technique which has an ability to group together vectors of similar words. If given a large enough datset, Word2Vec can make strong approximations about a word's meaning based on their occurrences in the text, which leads to better word assocaiations with other words in the corpus. Words like “King” and “Queen” for exampe, would be very similar to one another and when algebraic operations are conducted on word embeddings.
Word2vec uses two types of methods for learning these relationships which are; 
- Skip-gram
- CBOW (Continuous Bag of Words)

**Use of Word2Vec**

Files are located in the word2vex folder

1. Model Development of using Word2Vec in tweet_word2vex.ipynb
    - Sentiment analysis

    - Model training and experimentation

    - Exporting of trained models
    
    - w2v_model.pkl is the pretained model for logistic regression

    - w2v_model_knn.pkl is the pretained model for KNN Classifier without PCA

    - w2v_model_knn_no_pca.pkl is the pretained model for KNN Classifier without PCA
 
    - w2v_model_rf.pkl is the pretained model for Random Forest Classifier with PCA

    - w2v_model_rf_no_pca.pkl is the pretained model for Random Forest Classifier without PCA
 
    - w2v_model_rf_undersampler.pkl is the pretained model for Random Forest Classifier with undersampler but without PCA

    - w2v_model_rf_undersampler_pca.pkl is the pretained model for Random Forest Classifier with undersampler and PCA

2. Scraping of new data for predictions purposes and also EDA analysis on the newly sraped data in tweet_word2vex_2.ipynb
    - tweet_word2vex_for_prediction.csv contains the unseen data used for prediction

3. Use of newly scraped data for predictions in tweet_word2vex_3.ipynb
    - predicted_w2v.csv and predicted_w2v.xlsx are the outputs of the predictions made on tweet_word2vex_for_prediction.csv using the developed models and W2Vec model.

4. Within the img_tweet_word2vex folder contains the WordCloud imagery of the cleaned data, positive, neutral and negative tweets realting to Covid-19, Monkeypox and health.

## **Replicating this project**

Dependencies for this project can be found in env.yml

1. Download and install &nbsp;[Anaconda](https://www.anaconda.com/products/distribution#Downloads)

2. Navigate to your desktop and create a new folder using  **mkdir "folder_name1"** and paste the env.yml file into **folder_name1** 

3. Create another folder within folder_name1; **folder_name1/folder_name2**

4. Run **conda env create -f env.yml -p ../folder_name1/folder_name2**

5. Run **conda env list** to list all environments created using Anaconda

6. Run **conda activate ./folder_name2** to activate the environment

7. Run **conda list** to check if all dependencies have been installed

8. Have fun experimenting !