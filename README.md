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
The dataset analyzed in this project is from [Project FiveThirtyEight](https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv). Data contains 33 features starting from 1920 until 2022. There are some features like importance was introduced very late in 2021. It contains ELO ratings for both Teams and Quarterbacks and has rating pre-game and post-game.  

### Data Analysis
There 33 features in dataset that are split between teams, Home team and Visiting team. There are around 17K rows and below the columns in dataset,

```
['date', 'season', 'neutral', 'playoff', 'team1', 'team2', 'elo1_pre',
       'elo2_pre', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post',
       'qbelo1_pre', 'qbelo2_pre', 'qb1', 'qb2', 'qb1_value_pre',
       'qb2_value_pre', 'qb1_adj', 'qb2_adj', 'qbelo_prob1', 'qbelo_prob2',
       'qb1_game_value', 'qb2_game_value', 'qb1_value_post', 'qb2_value_post',
       'qbelo1_post', 'qbelo2_post', 'score1', 'score2', 'quality',
       'importance', 'total_rating']
```

We have ELO ratings by team and quarter-backs like Pre-game, probability and post-game ratings. We have dates games were played and the season. We have ignored the dates and post ratings. And since the importance was not until 2021 that is also removed which removes total_rating too as it is derived from quality and importance.

### Preprocessing

As mentioned above after removing certain features, and focusing on all pre-game and prob ratings, we got around 15K rows. There are no null's in this data which is good. The main data target is not part of the dataset, but considering the game prediction we have derived the Winner by comparing scores and assigning a binary score 0 for Home Team win and 1 for Visitors. 

Other feature engineering includes ELO difference between teams and quarter backs. 

Data analysis shows how the Home team has adavantage and also shows how the ELO ratings favor Home team, it is not very obvious but there is a slight +ve impact to  Home teams.

## Modeling
### Approach
- Explanation of the chosen machine learning approach (e.g., supervised, unsupervised, or semi-supervised).
- Justification for the selected approach based on the problem and the nature of the data.
### Model Selection
- Description of the models considered for the task and the rationale behind their selection.
- Comparison of different algorithms, if applicable.
### Training and Evaluation
- Details of how the models were trained, including hyperparameter tuning and cross-validation.
- Evaluation metrics used to assess model performance, along with their interpretation.
- Discussion of the results, including any challenges encountered and insights gained from the modeling process.

## Metrics
- Explanation of the evaluation metrics used to assess the performance of the model(s).
- Justification for why these specific metrics were chosen and how they relate to the project's objectives.
- Interpretation of the metric values and their implications for the model's effectiveness.


