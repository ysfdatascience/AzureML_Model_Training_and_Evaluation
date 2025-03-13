import os
import mlflow
import mlflow.sklearn
import argparse
import pandas as pd

def read_data(input_data: str):
    """reading training data

    Args:
        input_data (str): path to input data

    Returns:
        pd.Dataframe: training data

    """
    df = pd.read_csv(input_data)
    return df



def clean_and_preprocess_data(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): training data

    Returns:
        pd.DataFrame: cleaned and processed data
    """    
    
    from sklearn.preprocessing import OrdinalEncoder

    ordinal_encoder = OrdinalEncoder(categories = [['High School', 'Bachelor', 'Master','Associate', 'Doctorate']])
    df.person_education = ordinal_encoder.fit_transform(df.person_education.values.reshape(-1, 1))
    df.person_gender = df.person_gender.map({"female": 0, "male": 1})
    df.previous_loan_defaults_on_file = df.previous_loan_defaults_on_file.map({"No": 0, "Yes":1})
    df = pd.get_dummies(data = df, columns = ["person_home_ownership", "loan_intent"], dtype = int, drop_first = True)

    return df


def split_data(df: pd.DataFrame, test_ratio: float, random_state: int):
    """_summary_

    Args:
        df (pd.DataFrame): training data for splitting
        test_ratio (float): train-test split ratio
        random_state (int): random seed for reproducibility

    Returns:
        pd.DataFrame: x_train, x_test, y_train, y_test
    """    

    from sklearn.model_selection import train_test_split
    x = df.drop("loan_status", axis = 1)
    y = df["loan_status"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=random_state)
    mlflow.log_metric("x train samples", x_train.shape[0])
    mlflow.log_metric("x test samples", x_test.shape[0])
    mlflow.log_metric("y train samples", y_train.shape[0])
    mlflow.log_metric("y test samples", y_test.shape[0])

    return x_train, x_test, y_train, y_test



def train_model(x_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, learning_rate: float, min_samples_split: int):
    """_summary_

    Args:
        x_train (pd.DataFrame): features
        y_train (pd.Series): labels
        n_estimators (int): number of trees
        learning_rate (float): learning rate
        min_samples_split (int): min sample split

    Returns:
        fitted model
    """    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_samples_split = min_samples_split
    ) 

    fitted_model = model.fit(x_train, y_train)
    registred_model_name = "loan_model"
    mlflow.sklearn.save_model(
        sk_model = model,
        path = os.path.join(registred_model_name, "trained_model")
    )
    return fitted_model


def eval_model(model: "GradientBoostingClassifier", x_test: pd.DataFrame, y_test: pd.Series):
    """_summary_

    Args:
        model (GradientBoostingClassifier): fitted scikit-learn model
        x_test (pd.DataFrame): features
        y_test (pd.Series): labels
    """    
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(x_test)

    
    cr = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(cr, "data.json")
    acc_score = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy_score", acc_score)
    


def arguments():
    """argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type = str, help = "path to input data")
    parser.add_argument("--learning_rate", type = float)
    parser.add_argument("--n_estimators", type = int)
    parser.add_argument("--min_sample_split", type = int)
    parser.add_argument("--test_ratio", type = float)
    parser.add_argument("--random_state", type = int)

    args = parser.parse_args()
    
    return args



def main(args):
    """main function"""
    mlflow.sklearn.autolog()
    df = read_data(input_data=args.input_data)

    cleaned_data = clean_and_preprocess_data(df=df)
    
    x_train, x_test, y_train, y_test = split_data(cleaned_data, test_ratio=args.test_ratio, random_state=args.random_state)

    model = train_model(x_train=x_train, y_train=y_train, n_estimators=args.n_estimators, learning_rate=args.learning_rate, min_samples_split=args.min_sample_split)

    eval_model(model=model, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    args = arguments()
    main(args)    
        
    
