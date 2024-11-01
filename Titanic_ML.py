import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error


def fill_missed_values(df):
    df = df.drop(columns=["Ticket", "PassengerId", "Cabin", "Name"])
    cols = ["SibSp", "Fare", "Parch", "Age"]
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    df["Embarked"] = df["Embarked"].fillna("U")
    return df


if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    test_id = test["PassengerId"]

    train = fill_missed_values(train)
    test = fill_missed_values(test)

    # encode string data
    le = preprocessing.LabelEncoder()
    le_cols = ["Sex", "Embarked"]
    for col in le_cols:
        train[col] = le.fit_transform(train[col])
        test[col] = le.fit_transform(test[col])

    # data split
    y = train["Survived"]
    x = train.drop(columns="Survived")
    X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=45)

    # logistic regression model
    lg_model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    lg_prediction = lg_model.predict(X_val)
    lg_val_mae = mean_absolute_error(y_val, lg_prediction)
    print("Validation MAE: {:,.2f}".format(lg_val_mae))
    print(classification_report(y_val, lg_prediction))


    # prediction submission
    sub_surv_pred = lg_model.predict(test)
    df = pd.DataFrame({"PassengerId": test_id.values,
                      "Survived": sub_surv_pred,
                       })
    df.to_csv("sub.csv", index=False)




