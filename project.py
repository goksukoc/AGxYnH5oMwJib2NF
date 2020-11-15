import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
plt.style.use('ggplot')

%matplotlib inline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data = pd.read_csv("C:/Users/Goksu/Downloads/term-deposit-marketing-2020.csv")

dt_cpy = data.copy()
dt_cpy["y"].value_counts().plot(kind='bar')

data["job"] = LabelEncoder().fit_transform(data["job"])
data["marital"] = LabelEncoder().fit_transform(data["marital"])
data["education"] = LabelEncoder().fit_transform(data["education"])
data["housing"] = LabelEncoder().fit_transform(data["housing"])
data["loan"] = LabelEncoder().fit_transform(data["loan"])
data["contact"] = LabelEncoder().fit_transform(data["contact"])
data["month"] = LabelEncoder().fit_transform(data["month"])
data["y"] = LabelEncoder().fit_transform(data["y"])
data["default"] = LabelEncoder().fit_transform(data["default"])


sns.pairplot(dt_cpy, hue='y');

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 0:13], data.iloc[:,13],
                                                    test_size = 0.2, random_state=42)
from imblearn.under_sampling import RandomUnderSampler
X_under, y_under = RandomUnderSampler(random_state=42).fit_sample(X_test,y_test)

from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_under, y_under)

#Predict the response for test dataset
y_pred = gb.predict(X_test)

scores = cross_val_score(gb, X_under, y_under, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

mtxr_undr =confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mtxr_undr)
plt.show()

# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))