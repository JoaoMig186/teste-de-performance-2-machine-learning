import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("datasettitanic.csv")

df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

imputer_num = SimpleImputer(strategy="median")
imputer_cat = SimpleImputer(strategy="most_frequent")

df["Age"] = imputer_num.fit_transform(df[["Age"]])
df["Embarked"] = imputer_cat.fit_transform(df[["Embarked"]]).ravel()

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.reset_index(drop=True, inplace=True)

print(X_train.head())
