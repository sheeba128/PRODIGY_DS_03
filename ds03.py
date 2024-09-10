import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Sheeba C\Prodigy Infotech\Tasks\bank.csv")

# Convert categorical variables to numeric
df_encoded= pd.get_dummies(df, drop_first=True)
# Print column names after encoding
#print("After encoded", df_encoded.columns)

# Replace 'deposit' with the actual target column name after printing the column names
target_column = 'deposit_yes'  # Modify this if the target column is named differently
if target_column in df_encoded.columns:
    X = df_encoded.drop(target_column, axis=1)  # Features (remove the target column)
    y = df_encoded[target_column]  # Target variable
else:
    raise KeyError(f"Target column '{target_column}' not found in DataFrame")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.title('Decision Tree Classifier')
plt.show()


