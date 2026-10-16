# College Football Play Prediction

This project investigates how predictable offensive play-calling decisions are in college football and whether machine learning can uncover patterns that inform offensive strategy. Using 2025 play-by-play data from the University of Georgia and Indiana University, the project will develop classification models to predict whether an offense will pass or rush based on game situations, team tendencies, and recent play history.

By comparing model performance and examining the features that contribute to predictions, this project aims to identify the situations in which offensive play calls are most predictable. The findings will support data-driven recommendations for coaching staffs to review play-calling tendencies and evaluate potential opportunities to become less predictable.

## Research Questions

1. How accurately can machine learning predict offensive play selection in college football?
   - Establish how accurately machine learning models can classify plays as Pass or Rush using information available before the snap.
3. Which features of game situations contribute most to predictability?
4. Does recent play history improve predictions beyond current game situations and team tendencies?
   - Compare models with and without information about previous plays to determine whether sequential patterns provide additional predictive value
6. How does the predictability of offensive play selection vary across different game situations?
   - Are certain game situations harder to predict plays?
   - Group test-set plays by situation and comparing prediction metrics or how often the model correctly predicts Pass vs. Rush
   - Ex- third and long, trailing late in the game, etc.

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

- Logistic Regression
- Regularized Logistic Regression
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
