import pandas as pd
import numpy as np

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
    '''
    
    '''
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

