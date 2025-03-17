import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasettitanic.csv")

df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

num_features = ["Age", "Fare"]
cat_features = ["Pclass", "Sex_male", "Embarked", "SibSp", "Parch"]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=19))
])


knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f" Acurácia do modelo KNN: {accuracy:.4f}")
print("\n Relatório de Classificação:\n", report)

sns.set(style="whitegrid")

num_features = ["Age", "Fare"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(x=df["Age"], ax=axes[0])
axes[0].set_title("Boxplot da Idade")

sns.boxplot(x=df["Fare"], ax=axes[1])
axes[1].set_title("Boxplot da Tarifa")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df["Age"], kde=True, ax=axes[0])
axes[0].set_title("Distribuição da Idade")

sns.histplot(df["Fare"], kde=True, ax=axes[1])
axes[1].set_title("Distribuição da Tarifa")

plt.tight_layout()
plt.show()

