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

def get_unique_values(df, cat_columns):
    '''
    
    '''
    unique_values_list = []

    try:
        for column in cat_columns:
            unique_values = df[column].unique()
            unique_values_list.append(unique_values)

        # Find the maximum value among all columns
        max_length = max(len(values) for values in unique_values_list)

        unique_df = pd.DataFrame(columns=cat_columns)

        # Iterate over each column and fill with unique values, 
        # padding with NaN if necessary
        for i, column in enumerate(cat_columns):
            unique_values = unique_values_list[i]
            padded_values = np.append(unique_values, [np.nan] * (max_length - len(unique_values)))
            unique_df[column] = padded_values

        return unique_df
    except Exception as ex:
            print(f"get_unique_values:Error: {ex}")

    return None

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

