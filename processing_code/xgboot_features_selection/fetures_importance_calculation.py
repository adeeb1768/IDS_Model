import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Load data
df=pd.read_csv(r'C:\Users\adeeb\Desktop\master thesis\8_files_balanced.csv')

# Split data into X and y
x=df.loc[:, df.columns != 'label']
y=df['label']

# Label encoding
label_encoder = LabelEncoder()
y=label_encoder.fit_transform(y)

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [4, 8, 16, 26],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=13),
                           param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the predictions (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with best model: %.2f%%" % (accuracy * 100.0))

# Print the best parameters found by GridSearchCV
print("Best parameters:", best_params)

# Plot feature importance (using the best model)
plot_importance(best_model)
plt.show()
