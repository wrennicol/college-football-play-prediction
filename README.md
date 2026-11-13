# College Football Play Prediction

## Overview

This project investigates how predictable offensive play-calling decisions and play outcomes are in college football using machine learning. Using 2025 play-by-play data from the University of Georgia and Indiana University, we develop classification and regression models to analyze offensive tendencies and predict play outcomes.

The project focuses on two prediction tasks:

1. **Play Selection (Classification):** Predict whether an offense will Pass or Rush based on pre-snap game situations and historical play tendencies.
2. **Play Outcome (Regression):** Predict the points per average on an offensive play based on game situations and available pre-snap information.

By comparing model performance and examining the features that contribute most to predictions, this project aims to identify patterns in offensive decision-making and evaluate how effectively machine learning can predict play outcomes.

The ultimate goal is to develop data-driven insights that could help coaching staffs evaluate offensive tendencies and make more informed strategic decisions.

## Research Questions

### 1. Play Selection: Classification

How accurately can machine learning predict offensive play selection in college football?

- Predict whether an offense will Pass or Rush using information available before the snap.
- Evaluate how game situations and historical tendencies influence play-calling decisions.

### 2. Play Outcome: Regression

How many yards can an offense expect to gain in a given game situation?

- Predict yards gained on an offensive play.
- Examine how game situations and team tendencies relate to offensive performance.

### 3. Feature Importance

Which features contribute most to predicting offensive play selection and play outcomes?

- Evaluate the importance of down, distance, field position, score differential, and time remaining.
- Examine whether recent play history and team tendencies improve predictions.
- Compare the most influential features across the two prediction tasks.

## Datasets

The project uses 2025 college football play-by-play data from two teams:

- University of Georgia
- Indiana University

The datasets contain play-level information about game situations, offensive and defensive teams, play types, and play outcomes.

Using data from two different teams allows us to examine whether predictive patterns are consistent across different offensive tendencies and game situations.

The datasets will be cleaned, standardized, and combined into a unified dataset for exploratory analysis and modeling.

## Data Preprocessing

Preprocessing will include:

- Combining the Georgia and Indiana datasets.
- Removing irrelevant identifiers and redundant columns.
- Handling missing values.
- Converting categorical variables into appropriate numerical representations.
- Creating features from previous plays and recent play sequences.
- Removing features that introduce information unavailable before the snap.
- Preparing separate datasets for classification and regression.

## Exploratory Data Analysis

Exploratory data analysis will examine:

- Pass-to-rush distributions.
- Offensive play tendencies by down and distance.
- Play selection across different field positions and game situations.
- Distribution of yards gained.
- Relationships between game situation variables.
- Correlations between numerical features.
- Differences in offensive tendencies between Georgia and Indiana.

Visualizations will be used to identify patterns, relationships, and potential predictors before model development.

## Planned Features

Potential predictive features include:

| Feature | Description |
|---|---|
| Down | Current down (1st–4th) |
| Distance | Yards needed for a first down |
| Field Position | Offensive field position |
| Score Differential | Difference between offensive and defensive scores |
| Quarter | Current game quarter |
| Time Remaining | Time remaining in the game |
| Previous Play | Type of previous offensive play |
| Previous Play Outcome | Outcome of the previous play |
| Recent Play Sequence | Sequence of recent offensive plays |
| Offensive Team | Team currently possessing the ball |
| Defensive Team | Opposing team |

Only information available before the current play will be used as predictive features.

## Modeling Techniques

Two separate modeling pipelines will be developed.

### 1. Classification: Pass vs. Rush

**Target Variable:** Offensive play type (Pass or Rush)

Models:
- Null Model (majority class prediction)
- Logistic Regression
- Random Forest Classifier
- Neural Network Classifier

### 2. Regression: Yards Gained

**Target Variable:** Yards gained on the current play

Models:
- Null Model (mean prediction)
- Linear Regression
- Random Forest Regressor
- Neural Network Regressor

These models provide comparisons between baseline predictions, interpretable statistical models, nonlinear machine learning models, and more complex neural network approaches.

## Model Evaluation

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Regression Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Model performance will be compared against the corresponding null model to determine whether more sophisticated techniques provide meaningful improvements.

## Feature Selection and Importance

Feature selection and importance will be examined throughout model development.

- Evaluate correlations and remove redundant features.
- Examine feature importance from Random Forest models.
- Compare feature contributions across classification and regression.
- Identify which game situations and historical tendencies contribute most to predictions.

## Results and Interpretation

Results will be presented through tables and visualizations comparing the performance of each model.

The analysis will focus on:

- How accurately offensive play selection can be predicted.
- How accurately yards gained can be estimated.
- Which features contribute most to each prediction task.
- Whether predictive patterns differ between Georgia and Indiana.
- Whether complex models meaningfully outperform simpler approaches.

## Recommendations

The project will conclude with data-driven recommendations for evaluating offensive play-calling tendencies and identifying game situations associated with different offensive outcomes.

Potential applications include identifying predictable play-calling situations, evaluating offensive tendencies, and examining opportunities for more effective offensive decision-making.

