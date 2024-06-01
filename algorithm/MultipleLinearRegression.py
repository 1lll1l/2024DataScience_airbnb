import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Import Datasets
df_origin = pd.read_csv('c:/work/Airbnb_Data.csv', encoding='utf-8')

# Unnecessary attribute drop (many attributes with too many unique values unrelated to price have the same value)
columns_to_drop = ['id','description', 'thumbnail_url','first_review',
                   'host_since', 'bed_type','last_review','name',
                   'neighbourhood','zipcode','amenities', 'number_of_reviews',
                   'review_scores_rating', 'host_response_rate',
                   'host_has_profile_pic','host_identity_verified']

# Remove unnecessary columns
df = df_origin.drop(columns=columns_to_drop)

# DataFrame with rows with NaN values removed
df_cleaned = df.dropna()

def fill_nan_with_mean(df, log_price_column, null_column, window):
    
    # DataFrame with rows with NaN values removed
    df_cleaned = df.dropna()
    
    # Get log_price of rows with NaN values
    price_of_null_values = df[df[null_column].isnull()][log_price_column]

    for h in price_of_null_values:
        lower_bound = h - window
        upper_bound = h + window

        # Calculate the average value of rows in a range
        mean_value = df_cleaned[(df_cleaned[log_price_column] >= lower_bound)
         & (df_cleaned[log_price_column] <= upper_bound)][null_column].mean()
        mean_value = round(mean_value)

        # Update the value of the row with NaN with the calculated mean value
        df.loc[df[log_price_column] == h, null_column] = mean_value

# Dealing with missing values
fill_nan_with_mean(df, 'log_price', 'bedrooms', 0.4)
fill_nan_with_mean(df, 'log_price', 'bathrooms', 0.4)
fill_nan_with_mean(df, 'log_price', 'beds', 0.4)



# Select Numerical Data
outlier_numeric = ['log_price', 'accommodates']
numeric_data = df[outlier_numeric]

# outlier visualization
plt.figure(figsize=(12, 6))
for i, column in enumerate(numeric_data.columns, 1):
    plt.subplot(1, 2, i)
    plt.boxplot(numeric_data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

def remove_outliers_iqr(dataframe, columns, threshold=1.5):
    new_dataframe = dataframe.copy()
    for column in columns:
        Q1 = dataframe[column].quantile(0.3)
        Q3 = dataframe[column].quantile(0.7)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        new_dataframe = new_dataframe[(new_dataframe[column] >= lower_bound)
                                      & (new_dataframe[column] <= upper_bound)]
    return new_dataframe

# Data after outlier removal
df_no_outliers = remove_outliers_iqr(df, outlier_numeric)

# Visualize after outlier removal
plt.figure(figsize=(12, 6))
for i, column in enumerate(numeric_data.columns, 1):
    plt.subplot(1, 2, i)
    plt.boxplot(df_no_outliers[column])
    plt.title(column)
plt.tight_layout()
plt.show()



# Change dtype (shape unification with cleaning_fee)
pd.set_option('future.no_silent_downcasting', True)
df['instant_bookable'] = df['instant_bookable'].replace({'t': True, 'f': False})

category_columns = ['property_type', 'room_type', 'cancellation_policy',
                    'city', 'cleaning_fee', 'instant_bookable']
# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=category_columns)

# Convert Boolean column to int
boolean_columns = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)




# Numerical Data Scaling

numerical_cols = ['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'latitude', 'longitude']

# Standard Scaling
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print("StandardScaling", end="\n\n")
print(df_encoded[numerical_cols].head())


X = df_encoded.drop(columns=['log_price'])
y = df_encoded['log_price']




model = RandomForestRegressor()
model.fit(X, y)

# Set all columns to display
pd.set_option('display.max_columns', None)

# Attribute Importance Output
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feature_importances)


selected_features = ['room_type_Entire home/apt', 'bedrooms', 'latitude','longitude',
                     'beds', 'cancellation_policy_flexible',
                     'cleaning_fee_False', 'cancellation_policy_strict', 'cleaning_fee_True',
                     'property_type_Apartment', 'property_type_House',
                     'bathrooms', 'accommodates', 'room_type_Private room', 'log_price']
X= df_encoded[selected_features]


df_selected = df_encoded[selected_features]

# Calculation of correlation matrix for data frames
correlation_matrix = df_selected.corr()
# Draw Correlated Heat Maps
plt.figure(figsize=(10, 8))  
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Reduce the font size of the axis.
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)

plt.title('Correlation Heatmap')
plt.tight_layout()  # Layout adjustments prevent elements from being cut.
plt.show()


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up the range of k values
k_values = range(1, X_train.shape[1] + 1)
mean_mse_scores = []

# Performing cross-validation for each k value
for k in k_values:
    # Creating a pipeline: SelectKBest + LinearRegression
    pipeline = Pipeline([
        ('select', SelectKBest(score_func=f_regression, k=k)),
        ('regression', LinearRegression())
    ])
    
    # Performing cross-validation
    mse_scores = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse_scores.append(np.mean(mse_scores))

# Finding the optimal k value
optimal_k = k_values[np.argmin(mean_mse_scores)]
print(f"Optimal k: {optimal_k}")

# Plotting the k MSE graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_mse_scores, marker='o')
plt.xlabel('Number of Features (k)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Number of Features')
plt.show()

# Using the optimal k value for feature selection
selector = SelectKBest(score_func=f_regression, k=optimal_k)
X_selected = selector.fit_transform(X, y)

# Get the indices of selected features
selected_indices = selector.get_support(indices=True)

# Get the names of selected features
selected_features = X.columns[selected_indices]

print("Selected Features:")
print(selected_features)



# Training the multiple linear regression model
model = LinearRegression()
model.fit(X_selected, y)
y_pred = model.predict(X_selected)

# Plotting the comparison graph between actual and predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Initializing the list
scores=[]

# Performing model execution and evaluation using K-Fold
for train_index, test_index in kf.split(X_selected):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Model training
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)

    score=model.score(X_test, y_test)
    scores.append(score)
    

# Printing k-fold cross validation result
print("Cross validation Accuracy: {}".format(scores))
print("Average Accuracy score: {}" .format(np.mean(scores)))
