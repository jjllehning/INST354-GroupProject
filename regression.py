import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

comp_df = pd.read_csv("complaint.csv")


#Plot for GPA
#Cut continuous, bound data into parts
bins = [0, 2.5, 3.5, 4.5]
labels = [1, 2, 3]
comp_df['Gpa'] = pd.cut(comp_df['Gpa'], bins=bins, labels=labels, include_lowest=True)

#Drop missing rows
comp_df.dropna(subset=['Gpa'], inplace=True)

X = comp_df[['Gpa']]
y = comp_df['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

LogReg = LogisticRegression()

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))

print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
genres = comp_df['Genre'].unique()
print("Confusion Matrix:\n", conf_matrix)

fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g', xticklabels=genres, yticklabels=genres)
plt.title("Confusion Matrix- GPA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.show()


#Plot for Age
X2 = comp_df[['Age']]
y2 = comp_df['Genre']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

LogReg2 = LogisticRegression()

scaler2 = preprocessing.StandardScaler()
X_train2 = scaler2.fit_transform(X2_train)
X_test2 = scaler2.transform(X2_test)

LogReg2.fit(X2_train, y2_train)

y2_pred = LogReg2.predict(X2_test)

print("Accuracy:", accuracy_score(y2_test, y2_pred))
print("Precision:", precision_score(y2_test, y2_pred, average='weighted'))
print("Recall:", recall_score(y2_test, y2_pred, average='weighted'))

print("Classification Report:\n", classification_report(y2_test, y2_pred))

conf_matrix2 = confusion_matrix(y2_test, y2_pred)
genres2 = comp_df['Genre'].unique()
print("Confusion Matrix:\n", conf_matrix2)

fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(conf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g', xticklabels=genres, yticklabels=genres)
plt.title("Confusion Matrix- Age")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.show()

