# College Football Play Prediction

This project investigates how predictable offensive play-calling decisions and play outcomes are in college football and whether machine learning can uncover patterns that inform offensive strategy. Using 2025 play-by-play data from the University of Georgia and Indiana University, the project will develop both classification and regression models using game situations, team tendencies, and recent play history.

The classification models will predict whether an offense will pass or rush, while the regression models will estimate expected play outcomes, such as yards gained or Expected Points Added (EPA). By comparing model performance and examining the features that contribute to predictions, this project aims to identify the factors associated with offensive play selection and estimate potential outcomes under different game conditions. The findings will support data-driven recommendations for coaching staffs to review play-calling tendencies and evaluate strategic opportunities.

## Research Questions

1. How accurately can machine learning predict offensive play selection in college football?
   - Establish how accurately machine learning models can classify plays as Pass or Rush using information available before the snap.
2. Which features of game situations contribute most to predictability?
   - Examine the predictive importance of down, distance, field position, score differential, time remaining, team tendencies, and recent play history.
3. How many yards can an offense expect to gain in a given situation?
   - Regression models target EPA, yards gained, or other success metrics

## Dataset
The project uses 2025 college football play-by-play data for University of Georgia and Indiana University. The dataset contains play-level information about game situations, teams, play types, and play outcomes.

## Planned Features

- Down
- Distance
- Field postion
- Score differential
- Quarter
- Time remaining
- Previous play
- Previous play outcome
- Recent play sequence
- Offensive team
- Defensive team

## Models

- Logistic/Linear Regression
- Regularized Logistic/Linear Regression
- Random Forest
- XGBoost
- Neural Network
- LTSM/GRU

## Evaluation

Play-call prediction will be evaluated using classification metrics such as:

- RMSE
- MAE
- $R^2$
- Expected Points Added (EPA)
