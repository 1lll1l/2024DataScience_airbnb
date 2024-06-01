import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('c:/work/Airbnb_Data.csv')

# Select the relevant columns
features = ['property_type', 'room_type', 'accommodates', 'bathrooms', 'bed_type','number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds']
target = 'log_price'

# Define X and y
X = data[features]
y = pd.cut(data[target], bins=[-float('inf'), 2, 6, float('inf')], labels=['low', 'medium', 'high'])

# Preprocess the features
# Define which columns are numeric and which are categorical
numeric_features = ['accommodates', 'bathrooms', 'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds']
categorical_features = ['property_type', 'room_type', 'bed_type']

# Create transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the KNN pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9]
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform GridSearchCV with K-Fold Cross Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameter
best_k = grid_search.best_params_['classifier__n_neighbors']
print(f"The best number of neighbors is: {best_k}")

# Evaluate the best model using K-Fold Cross Validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=3, scoring='accuracy')
print("Cross Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# You can also check classification report for each fold if needed
for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold} Accuracy:", score)

# Evaluate the best model using K-Fold Cross Validation and test set
for i, classifier in enumerate(grid_search.cv_results_['params'], start=1):
    final_model = grid_search.best_estimator_.set_params(**classifier)
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"Classifier {i} Test Set Accuracy:", test_accuracy)
