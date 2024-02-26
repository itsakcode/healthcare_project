import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

def get_data_types(df):
    '''
    
    '''
    col_data_types = {
        'category' : [],
        'numerical' : [],
        'other' : []
    }

    for col in df.columns:
        if (df[col]).dtype == object:
            col_data_types['category'].append(col)
        elif (df[col]).dtype in (int, float):
            col_data_types['numerical'].append(col)
        else:
            col_data_types['other'].append({col : (df[col]).dtype})
    
    return col_data_types

def get_unique_values(df, category_columns):
        '''
            Get unique values and their count of all category columns in a
            dataframe.
        '''
        col_unique_values = {}
        try:
            
            for col in category_columns:
                col_unique_values[col] = df[col].value_counts().to_dict()

        except Exception as ex:
            print(f"get_unique_values:Error {ex}")

        return col_unique_values

def check_unique_values(df, cat_columns):
    unique_values = {
        "col_name" : cat_columns,
        "num_unique_vals" : []
    }

    try: 
        for column in cat_columns:
            unique_values['num_unique_vals'].append(len(df[column].unique()))

        return pd.DataFrame(unique_values, columns=['col_name', 'num_unique_vals'])
    except Exception as ex:
        print(f"check_unique_values:Error: {ex}")

    return None

def get_ordinal_encoders(X_train, feature_details):
    '''
    
    '''
    encoders = {}

    try:
        # encode each column using Ordinal Encoder
        for f_name, f_unique_dict in feature_details.items():
            # get unique value names
            category_values = list(f_unique_dict.keys())
            ordinal_encoder = OrdinalEncoder(categories=[category_values], handle_unknown='use_encoded_value', unknown_value=-1)
            encoders[f_name] = ordinal_encoder.fit(X_train[f_name].values.reshape(-1,1))

    except Exception as ex:
        print(f"get_ordinal_encoders:Error {ex}")

    return encoders

def encode_features(data, feature_encoders):
    '''
    
    '''
    data_encoded = data.copy()

    try:
        # encode each column using Ordinal Encoder
        for f_name, f_encoder in feature_encoders.items():
            data_encoded[f_name] = f_encoder.transform(data[f_name].values.reshape(-1, 1))
        
        return data_encoded
    except Exception as ex:
        print(f"encode_features:Error {ex}")

def get_features_scaler(X_train, scaler=None):
    '''
    
    '''
    # use Standard Scaler by default
    if scaler == None:
        scaler = StandardScaler()

    try: 
        fitted_scaler = scaler.fit(X_train)
        return fitted_scaler
    
    except Exception as ex:
        print(f"scale_X_data:Error {ex}")

def scale_features_data(X_data, scaler):
    '''
    
    '''
    try: 
        X_data_scaled = scaler.transform(X_data)

        return X_data_scaled
    except Exception as ex:
        print(f"scale_X_data:Error {ex}")

def get_target_encoder(y_train):
    '''
    
    '''
    # encode target column using Label Encoder
    label_encoder = LabelEncoder()

    y_encoder = label_encoder.fit(y_train)

    return y_encoder