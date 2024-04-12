import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
pd.options.mode.copy_on_write = True

DATA_FOLDER = "../data"

metrics_df = pd.read_csv(f"{DATA_FOLDER}/metrics.csv")
analysis_df = pd.read_csv(f"{DATA_FOLDER}/analysis.csv")

# Select columns with statistical significance
stat_rel = (analysis_df[analysis_df["p_value"] < 0.01]["metric"]).tolist()
sel_columns = stat_rel + ['healthy']
dataset = metrics_df[sel_columns]

# Normalize data
for metric in stat_rel:
    data = dataset[metric]
    dataset[metric] = (data - data.mean()) / data.std()

# Prepare data for classifier
X = dataset.drop(columns=['healthy'])
y = dataset["healthy"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression Classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)


# Linear SVC Classifier
lc = LinearSVC(dual="auto")
lc.fit(X_train, y_train)
lc_y_pred = lc.predict(X_test)
lc_accuracy = accuracy_score(y_test, lc_y_pred)

# K-Means Clustering
km = KMeans(n_clusters=2)
km.fit(X_train)
km_y_pred = km.predict(X_test)
km_accuracy = accuracy_score(y_test, km_y_pred)

print("Logistic Regression Accuracy: ", lr_accuracy)
print("Linear Support Vector Accuracy: ", lc_accuracy)
print("K-Means Accuracy: ", km_accuracy)