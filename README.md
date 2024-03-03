## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
  - [Source](#source)
  - [Data Analysis](#data-analysis)
  - [Preprocessing](#preprocessing)
- [Modeling](#modeling)
  - [Approach](#approach)
  - [Model Selection](#model-selection)
  - [Training and Evaluation](#training-and-evaluation)
- [Metrics](#metrics)


## Introduction
This project is to analyze and predict the NFL games based on ELO Ratings. Does it favor Home or Visiting teams? Does team rating has impact or quarterback rating? You can read more about ELO ratings [here](http://www.eloratings.net/about)

## Data
### Source
The dataset analyzed in this project is from [Project FiveThirtyEight](https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv). Data contains 33 features starting from 1920 until 2022. There are some features like importance that was introduced very late in 2021. It contains ELO ratings for Teams and Quarterbacks and has ratings for both pre-game and post-game.  

### Data Analysis
There 33 features in dataset that are split between both the teams, Home and Visiting team. There are around 17K rows and below are the columns in dataset,

```
['date', 'season', 'neutral', 'playoff', 'team1', 'team2', 'elo1_pre',
       'elo2_pre', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post',
       'qbelo1_pre', 'qbelo2_pre', 'qb1', 'qb2', 'qb1_value_pre',
       'qb2_value_pre', 'qb1_adj', 'qb2_adj', 'qbelo_prob1', 'qbelo_prob2',
       'qb1_game_value', 'qb2_game_value', 'qb1_value_post', 'qb2_value_post',
       'qbelo1_post', 'qbelo2_post', 'score1', 'score2', 'quality',
       'importance', 'total_rating']
```

We have different ELO ratings by team and quarter-backs like pre-game, probability and post-game ratings. We have dates when games were played and the season. We have ignored the dates and post ratings. And since the importance was not until 2021 that is also removed which removes total_rating too as it is derived from quality and importance.

Here is the data distribution of subset of features along with derived features (Feature Engineerin): 

![Features_distribution](https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/6ee33ea7-8b48-4d1e-a6ac-1e5192043513)


### Preprocessing

As mentioned above after removing certain features, and focusing on all pre-game and prob ratings, we got around 15K rows. There are no null's in this data which is good. The main data target is not part of the dataset, but considering the game prediction we have derived the Winner by comparing scores and assigning a binary score 0 for Home Team win and 1 for Visitors. 

Other feature engineering includes ELO difference between teams and quarter backs. 

Data analysis shows how the Home team has adavantage and also shows how the ELO ratings favor Home team, it is not very obvious but there is a slight +ve impact to  Home teams.  

The ELO ratings and how the winners are distributed:  
<img src="https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/cc67c1b2-7c74-4250-97d3-592b9c70e042" alt="ELO Rating vs Winner" width="600" height="400">  

The ELO ratings differences and how the winners are distributed:  
<img src="https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/d749a85e-baeb-455d-9c88-810f3183625c" alt="ELO Difference vs Winner" width="600" height="400">  

ELO Probability vs Scores and how winners are distributed:  
<img src="Visualizations and Modeled Predictions/Images/Elo_Probability_vs_Score.png" alt="ELO Prob vs Scores" width="600" height="400">  

Quarterback ELO ratings difference and how winners are distributed:  
<img src="Visualizations and Modeled Predictions/Images/qb_elo_diff_with_winner.png" alt="Quarterback ELO Difference vs Winner" width="600" height="400">  

Heatmap with all main features:  
<img src="https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/5eafa4b1-535a-41a1-bdbe-a77332359bf4" alt="Heatmap" width="600" height="400">  

## Modeling
### Approach

Taking all the main features (mainly ELO ratings) we have processed the data to evaluate different Classification models. Splitting the data with features and target and then training and testing data. Have scaled the training and test data. Below are the accuracy scores for all classification models trained on,  

<img src="https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/6c7bdbdf-62dd-4d0e-b298-9495bdce0f7c" alt="Accuracy Scores of all Models" width="600" height="400">  

### Model Selection

After reviewing all the classification models, Logistic Regression, AdaBooster and SVC seems to be closer. Performed more analysis using GridSearch with different hyperparameters for these 3 models. Logistic Regression and SVC came closer or similar to each other. Considering the cost of processing we went with Logistic Regression. 

### Training and Evaluation

Models were trained with different set of features, hyperparameters and training sets. We changed training sets with different sizes and random states. Checking the accuracy for each set, getting coefficients and determining what features impacts and evaluating the scores as we process.

Best scores and hyper parameters from GridSearch: 

Logistic Regression:  
```
Best Parameters: {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}
Best Score: 0.6559197355113419
Test Score: 0.6508407517309595
```

AdaBoosterClassifier:  
```
Best Parameters: {'learning_rate': 0.1, 'n_estimators': 200}
Best Score: 0.6574033876985577
Test Score: 0.648203099241675
```

SVC:  
```
Best Parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
Best Score: 0.6546003178623334
Test Score: 0.6571051763930102
```

## Metrics

We evaluated multiple metrics available from sklearn to determine the best model. Predictability Proba, Coefficients, Cross-validations scores, Balanced Accuracy Scores, Classification Reports and Accuracy scores. Here is one of the confusion matrix plot for Logistic Regression. 

![conf_matrix](https://github.com/itsakcode/nfl_elo_predictions/assets/93089647/d7532ea8-406e-4997-8e35-505e730f2d98)

Here is the classification reports:  

```
Logisitic Regression Classification Report
              precision    recall  f1-score   support

           0       0.60      0.50      0.55      1274
           1       0.68      0.76      0.72      1759

    accuracy                           0.65      3033
   macro avg       0.64      0.63      0.63      3033
weighted avg       0.65      0.65      0.64      3033
```

Cross-validation scores:  
```
Cross-validation scores: [0.67194197 0.66094987 0.6378628  0.6444591  0.65435356]
Mean accuracy: 0.6539134602921078
Standard deviation of accuracy: 0.01201449638315885
```



