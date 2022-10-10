## Twitter Sentiment Analysis on Covid19, Monkeypox & Health
1. The project begins with fundamentals of Tweepy in tweet_sentiment.ipynb

2. Sentiment Analysis using VADER in tweet_sentiment_2.ipynb

3. Text classification in tweet_sentiment_3.ipynb
    - vectorizer.pkl and vectorizer_tfidf.pkl contain the vectorisers developed for Count Vectoriser and TF-IDF respectively.
    - cf_3.pkl and cf_3_tfidf.pkl contain the models developed using Count Vectoriser and TF-IDF respectively.

4. Predicting unseen data in tweet_sentiment_3._predict.ipynb
    - predict_set.csv contains the unseen data used for prediction
    - predicted.csv and predicted.xlsx are the outputs of the predictions made on predict_set.csv using the developed models and vectorisers.

5. Within the img folder contains the WordCloud imagery of the cleaned data, positive, neutral and negative tweets realting to Covid-19, Monkeypox and health.

## Use of Word2Vec

Files are located in tthe wor2vex folder

1. Model Development Using Word2Vec in tweet_word2vex.ipynb
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

2. Scraping of new data for predictions purposes  and also EDA analysis on the newly sraped data in tweet_word2vex_2.ipynb
    - tweet_word2vex_for_prediction.csv contains the unseen data used for prediction

3. Use of newly scraped data for predictions in tweet_word2vex_3.ipynb
    - predicted_w2v.csv and predicted_w2v.xlsx are the outputs of the predictions made on tweet_word2vex_for_prediction.csv using the developed models and W2Vec model.

4. Within the img_tweet_word2vex folder contains the WordCloud imagery of the cleaned data, positive, neutral and negative tweets realting to Covid-19, Monkeypox and health.

## Replicating this project

Dependencies for this project can be found in env.yml

1. Download and install &nbsp;[anaconda](https://www.anaconda.com/)

2. Run **conda env create -f env.yml**
