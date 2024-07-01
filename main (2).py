from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pandas as pd


# ColumnTransformer from https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
# used ColumnTransformer to apply different transformation to different subset of columns in the data set


# SimpleImputer from https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# used SimpleImputer to complete the missing values in the DataFrame
# strategy=mean for numeric and strategy=constant for category 

# StandardScalar from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# used standard scalar to standardize numeric features by removing the mean and scaling to variance
class UserPredictor:
    def __init__(self):
     
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])

  
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, ['age', 'past_purchase_amt', 'total_time_spent']),  
            ('cat', categorical_transformer, ['badge'])])

      
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression())
        ])

    # created a method to calculate the amount of time spent
    # added this method to receive 100% 
    def add_time_features(self, users, logs):

        total_time_per_user = logs.groupby('id')['duration'].sum().reset_index()
        total_time_per_user.columns = ['id', 'total_time_spent']

       
        users = users.merge(total_time_per_user, on='id', how='left').fillna(0)
        return users

    def fit(self, users, logs, y):
       
        users = self.add_time_features(users, logs)

        
        X = users.drop(columns=['id', 'name'])  
        y = y['clicked'].values                 

       
        self.pipeline.fit(X, y)

    def predict(self, users, logs):
        
        users = self.add_time_features(users, logs)

    
        X = users.drop(columns=['id', 'name'])  
        predictions = self.pipeline.predict(X)
        return predictions.astype(bool)

