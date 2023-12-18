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

plt.figure(figsize=(25, 25))
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g', xticklabels=genres, yticklabels=genres)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.show()
