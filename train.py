import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir, index_col=["PassengerId"])
    test = pd.read_csv(test_dir, index_col=["PassengerId"])

    return train, test


def pre_processing(train, test):
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1

    feature_names = ["Pclass", "Sex", "Fare", "SibSp", "Parch"]

    train.drop(columns = 'Parch', inplace = True)
    test.drop(columns = 'Parch', inplace = True)

    train_x, train_y = train[feature_names], train["Survived"]
    test_x, test_y = train[feature_names], train["Survived"]

    return train_x, train_y, test_x, test_y


def build_model(train_x, train_y):
    model = tree.BaseDecisionTree()
    # model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)

    return model


def evaluate(model, test_x, test_y):
    pred_y = model.predict(test_x)
    score = f1_score(test_y, pred_y)

    return score


if __name__ == "__main__":
    # Directory
    train_dir = "train.csv"
    test_dir = "test.csv"

    # Flow
    train, test = load_data(train_dir, test_dir)
    train_x, train_y, test_x, test_y = pre_processing(train, test)
    model = build_model(train_x, train_y)
    score = evaluate(model, test_x, test_y)
    print(score)

    pred_x = model.predict(test_x)
