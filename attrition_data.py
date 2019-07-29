import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def get_labels(df):
    labels = df['Attrition']
    labels = [1 if i == 'Yes' else 0 for i in labels]
    
    return labels
    
def drop_from_df(df, to_drop):
    df = df.drop(to_drop, axis = 1)
    
    return df
    
def get_dummies(df, to_dummy):
    dummies = pd.get_dummies(df[to_dummy])
    
    #print(dummies)
    df = df.drop(to_dummy, axis = 1)
    
    frames = [df, dummies]
    df = pd.concat(frames, axis = 1)
    
    return df
    
def get_scaled(df, continous):
    scaler = MinMaxScaler((0.05, 0.95))
    df[continous] = scaler.fit_transform(df[continous])
    
    return df
    
def main():
    wages = ['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate']

    to_dummy = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']##7

    categorical = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
               'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'] ##9

    to_continous = ['Age', 'DistanceFromHome', 'HourlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
             'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
             'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Education', 'EnvironmentSatisfaction',
                'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
               'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'] ##11

    to_drop = ['DailyRate', 'MonthlyIncome', 'MonthlyRate', 'Attrition',
           'EmployeeCount','Over18', 'StandardHours', 'EmployeeNumber']
    
    df = pd.read_csv('hr_attrition.csv')
    labels = get_labels(df)
    df = drop_from_df(df, to_drop)
    df_with_dummies = get_dummies(df, to_dummy)
    dummy_scaled_df = get_scaled(df_with_dummies, to_continous)
    labels = pd.DataFrame(labels)
    features = dummy_scaled_df.copy()
    
    assert len(labels) == len(features)
    labels.to_csv('hr_labels.csv', index = False)
    features.to_csv('hr_features.csv', index = False)
    
if __name__ == '__main__':
    main()
