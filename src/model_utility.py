from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def prepare_train_test_data(X, y, scaler=None):
    '''
        Split data into trainig and testing data. Scale if scaler is passed. 

        Arguments: X, y, and an optional Scaler object 

        Errors: TypeError is invalid scaler is passed
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    try:            
        if scaler and (type(scaler) in (StandardScaler, MinMaxScaler)):
            #print(f"Scaling data using : {type(scaler)}")
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        return X_train, X_test, y_train, y_test
    except TypeError as te:
        print(f"train_test_data Error: {te}")
    except Exception as ex:
        print(f"train_test_data Error: {ex}")

def process_models(models, X, y, scaler=None):
    '''
        Process the models by preparing training and testing data, 
        calculate accuracy score, confusion matrix and classification report

        Arguments: models - is a dictionary with model name as key and model
                    instance as value.
                    X , y - data
                    scaler - optional - if not passed MinMaxScaler will be used
        
        Returns: dictionary with model name as key and dictionary of 
                all the metrics as value

        Error: Errors will be caught and printed
    '''
    if models == None:
        models = { 
            'Logistic Regression' : LogisticRegression(max_iter=250, random_state=42),
            'RandomForest Classifier' : RandomForestClassifier(n_estimators=500, random_state=42),
            'SVC' : SVC(),
            'XGBoost Classifier' : XGBClassifier()
        }

    model_results = {}

    X_train, X_test, y_train, y_test = prepare_train_test_data(X, y, StandardScaler())

    for m_name, model in models.items():
        try:
            model.fit(X_train, y_train)

            t_score = model.score(X_train, y_train)

            y_predict = model.predict(X_test)
            
            a_score = accuracy_score(y_test, y_predict)
        
            conf_matrix = confusion_matrix(y_test, y_predict)

            class_report = classification_report(y_test, y_predict)

            model_results[m_name] = {
                "model" : model,
                "train_score" : t_score,
                "accuracy_score" : a_score,
                "confusion_matrix" : conf_matrix,
                "classification_report" : class_report
            }
        except Exception as ex: 
            print(f"process_models:Error: {ex}")
    
    return model_results
    
