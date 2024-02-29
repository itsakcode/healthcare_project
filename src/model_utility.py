from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR

regression_models = {"Linear Regression" : LinearRegression(), 
          "KNeighbors Regressor" : KNeighborsRegressor(n_neighbors=9), 
          "RandomForest Regressor" : RandomForestRegressor(n_estimators=128, random_state=1), 
          "ExtraTrees Regressor" : ExtraTreesRegressor(n_estimators=128, random_state=1), 
          "AdaBoost Regressor" : AdaBoostRegressor(n_estimators=128, random_state=1), 
          "SVR" : SVR(C=1.0, epsilon=0.2)
}

classification_models = {
          "Logistic Regression" : LogisticRegression(random_state=42), 
          "SVC" : SVC(kernel='poly', probability=True), 
          "KNeighbors Classifier" : KNeighborsClassifier(n_neighbors=9), 
          "DecisionTree Classifier" : DecisionTreeClassifier(), 
          "RandomForest Classifier" : RandomForestClassifier(n_estimators=256, random_state=42)
}

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

def process_models_Xy(models, X, y, scaler=None):
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
    model_results = {}

    X_train, X_test, y_train, y_test = prepare_train_test_data(X, y, MinMaxScaler())

    for m_name, model in models.items():
        try:
            model.fit(X_train, y_train)

            t_score = model.score(X_train, y_train)

            y_predict = model.predict(X_test)

            model_results[m_name] = {
                "model" : model,
                "train_score" : t_score,
                "accuracy_score" : accuracy_score(y_test, y_predict),
                "balanced_accuracy_score" : balanced_accuracy_score(y_test, y_predict),
                "confusion_matrix" : confusion_matrix(y_test, y_predict),
                "classification_report" : classification_report(y_test, y_predict)
            }
        except Exception as ex: 
            print(f"process_models:{m_name}:Error: {ex}")
    
    return model_results


def process_models(models, X_train, X_test, y_train, y_test):
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
    model_results = {}

    for m_name, model in models.items():
        try:
            model.fit(X_train, y_train)

            t_score = model.score(X_train, y_train)

            y_predict = model.predict(X_test)

            model_results[m_name] = {
                "model" : model,
                "train_score" : t_score,
                "accuracy_score" : accuracy_score(y_test, y_predict),
                "balanced_accuracy_score" : balanced_accuracy_score(y_test, y_predict),
                "confusion_matrix" : confusion_matrix(y_test, y_predict),
                "classification_report" : classification_report(y_test, y_predict)
            }
        except Exception as ex: 
            print(f"process_models:{m_name}:Error: {ex}")
    
    return model_results
    
