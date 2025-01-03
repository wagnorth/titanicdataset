# Titanic Survival Prediction and Model Evaluation
This project focuses on predicting Titanic passenger survival based on their features using multiple machine learning models. It also includes detailed evaluation of model performance using confusion matrices, accuracy scores, and decision tree rules parsing.

### Table of Contents
- Dataset
- Preprocessing
- Machine Learning Models
- Evaluation
- Results
- Setup
- Usage

### Dataset
The dataset SVMtrain.csv contains information about Titanic passengers:

URL: https://www.kaggle.com/datasets/shubhamgupta012/titanic-dataset/data

The dataset contains 9 attributes as follows:
- PassengerId: A unique identification number for each passenger.
- Survived: Survival status of the passenger (0 = Did not survive, 1 = Survived).
- Pclass: Passenger class (1 = First class, 2 = Second class, 3 = Third class).
- Sex: Passenger's gender. ( Male, Female )
- Age: Age of the passenger.
- SibSp: Number of siblings/spouses on board Titanic.
- Parch: Number of parents/children on board Titanic.
- Fare: The fare paid by the passenger.
- Embarked: Port of embarkation (1 = Cherbourg, 2 = Queenstown, 3 = Southampton).


### Preprocessing

1. Age Categorization:

- Baby (0-4): 1
- Child (5-12): 2
- Young (13-29): 3
- Middle-aged (30-59): 4
- Elderly (60+): 5

2. Sex Encoding:

- Female: 1
- Male: 2

3. Fare Adjustment:

- Missing or zero values in Fare replaced with the mean fare of the same Pclass (excluding min and max values).

4. Fare Normalization:

- Min-Max Scaling applied to normalize the Fare column.

5. Correlation Matrix:

- A correlation matrix is calculated for all features to identify relationships.

### Machine Learning Models
The following machine learning models are implemented:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

### Evaluation
1. Accuracy Scores
- Accuracy is calculated for each model using the entire dataset.

2. Confusion Matrices
Confusion matrices provide detailed performance evaluation for each model:

- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- True Positives (TP)

3. Decision Tree Rules Parsing
- Rules generated by a decision tree are parsed from a text file (decision_tree_rules.txt) into a structured table for better interpretability.


### Results
- Accuracy Scores: Models are ranked by their accuracy scores.

- Confusion Matrices: Provide insights into model-specific performance metrics such as sensitivity and specificity.

- Decision Tree Rules: Decision-making paths are represented in a human-readable format.

### License

Free

### Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to enhance this project.
