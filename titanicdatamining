import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('SVMtrain.csv')

# Drop the 'PassengerId' column
data = data.drop(columns=['PassengerId'])

# Age Categorization -> Baby: 0-4, Child: 5-12, Young: 13-29, Middle-aged: 30-59, Elderly: 60+
def categorize_age(age):
    if age <= 4:
        return 1  # Baby
    elif age <= 12:
        return 2  # Child
    elif age <= 29:
        return 3  # Young
    elif age <= 59:
        return 4  # Middle-aged
    else:
        return 5  # Elderly

data['Age'] = data['Age'].apply(categorize_age)

# Sex Categorization -> Female: 1, Male: 2
def categorize_sex(sex):
    if isinstance(sex, str):  # Ensure input is a string
        if sex.lower() == 'female':
            return 1  # Female
        elif sex.lower() == 'male':
            return 2  # Male
    return sex  # Return original value if not a string

# Apply the categorization function
data['Sex'] = data['Sex'].apply(categorize_sex)

# Compute Mean Fare by Pclass, excluding extremes (min and max)
mean_excluding_extremes = {}

for pclass in data['Pclass'].unique():
    # Filter data for the current Pclass
    class_data = data[data['Pclass'] == pclass]
    
    # Exclude max and min Fare values
    max_fare = class_data['Fare'].max()
    min_fare = class_data['Fare'].min()
    filtered_data = class_data[(class_data['Fare'] != max_fare) & (class_data['Fare'] != min_fare)]
    
    # Calculate mean Fare excluding extremes
    mean_excluding_extremes[pclass] = filtered_data['Fare'].mean()

# Replace Fare = 0 with mean Fare by Pclass
for pclass, mean_fare in mean_excluding_extremes.items():
    data.loc[(data['Fare'] == 0) & (data['Pclass'] == pclass), 'Fare'] = mean_fare

# Normalize the 'Fare' column using Min-Max Scaling
scaler = MinMaxScaler()
data['Fare'] = scaler.fit_transform(data[['Fare']])

# Reset index to start IDs from 1
data.index = data.index + 1
#data(useable)


# Calculate the correlation matrix for the dataset
correlation_matrix = data.corr()
#correlation_matrix(useable)

# Define a function to calculate confusion matrices for all models
def get_confusion_matrices(x, y):
    lr = LogisticRegression()
    dt = DecisionTreeClassifier(criterion="gini")
    kn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    g = GaussianNB()
    b = BernoulliNB()
    sv = SVC(kernel="rbf")
    gb = GradientBoostingClassifier()
    xg = XGBClassifier()

    models = [lr, dt, kn, rf, g, b, sv, gb, xg]
    model_names = ["Logistic_Regression", "Decision_Tree", "KNN", "Random_Forest", "Gaussian", "Bernoulli", "Support_Vector", "Gradient_Boost", "XGboost"]
    scale1 = StandardScaler()
    x_scaled = scale1.fit_transform(x)

    confusion_matrices = {}

    for model, name in zip(models, model_names):
        model.fit(x_scaled, y)
        y_pred = model.predict(x_scaled)
        cm = confusion_matrix(y, y_pred)
        confusion_matrices[name] = cm

    return confusion_matrices

# Prepare data for model training and confusion matrix calculation
X = data.drop(columns=['Survived'])
y = data['Survived']

# Define the classification function for the entire dataset
def classification_all_data(x, y):
    lr = LogisticRegression()
    dt = DecisionTreeClassifier(criterion="gini")
    kn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    g = GaussianNB()
    b = BernoulliNB()
    sv = SVC(kernel="rbf")
    gb = GradientBoostingClassifier()
    xg = XGBClassifier()

    models = [lr, dt, kn, rf, g, b, sv, gb, xg]
    model_names = ["Logistic_Regression", "Decision_Tree", "KNN", "Random_Forest", "Gaussian", "Bernoulli", "Support_Vector", "Gradient_Boost", "XGboost"]
    scale1 = StandardScaler()
    x_scaled = scale1.fit_transform(x)
    acc = []
    sonuc_df = pd.DataFrame(columns=["Accuracy Score"], index=model_names)

    for model in models:
        model.fit(x_scaled, y)
        y_pred = model.predict(x_scaled)
        acc.append(accuracy_score(y, y_pred) * 100)

    sonuc_df["Accuracy Score"] = acc
    return sonuc_df.sort_values("Accuracy Score", ascending=False)

# Run the classification function on the entire dataset
classification_all_results = classification_all_data(X, y)
#classification_all_results(usable)

# Calculate confusion matrices
confusion_matrices = get_confusion_matrices(X, y)

# Flatten the confusion matrices and create a DataFrame
flattened_matrices = []
for model, matrix in confusion_matrices.items():
    tn, fp, fn, tp = matrix.ravel()
    flattened_matrices.append([model, tn, fp, fn, tp])

confusion_matrices_df = pd.DataFrame(flattened_matrices, columns=['Model', 'TN', 'FP', 'FN', 'TP'])
#confusion_matrices_df (useable)

# Read the decision tree rules from the file 'decision_tree_rules.txt'
with open('decision_tree_rules.txt', 'r') as file:
    decision_tree_rules = file.read()

# Parse the decision tree rules into a structured table format
rules_lines = decision_tree_rules.split('\n')

# Extract relevant columns from the rules
parsed_rules = []
current_path = []

for line in rules_lines:
    depth = line.count('|')
    condition = line.split('|')[-1].strip()

    # Adjust the current path based on depth
    while len(current_path) > depth:
        current_path.pop()

    # Add the current condition to the path
    current_path.append(condition)

    # If the condition is a class prediction, save the full path
    if 'class:' in condition:
        parsed_rules.append({
            'Path': ' -> '.join(current_path[:-1]),
            'Prediction': condition.split(':')[-1].strip()
        })

# Convert the parsed rules into a DataFrame
rules_df = pd.DataFrame(parsed_rules)
#rules_df(useable)
