import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
pd.options.mode.copy_on_write = True

DATA_FOLDER = "../data"
P_VALUE_THRESHOLD = 0.05
N_TRAINING_SETS = 20
TEST_SIZE = 0.3

metrics_df = pd.read_csv(f"{DATA_FOLDER}/metrics.csv")
analysis_df = pd.read_csv(f"{DATA_FOLDER}/analysis.csv")

# Select columns with statistical significance
stat_rel = (analysis_df[analysis_df["p_value"] < P_VALUE_THRESHOLD]["metric"]).tolist()
sel_columns = stat_rel + ['healthy']
dataset = metrics_df[sel_columns]

# Normalize data
for metric in stat_rel:
    data = dataset[metric]
    dataset[metric] = (data - data.mean()) / data.std()

# Prepare data for classifier
X = dataset.drop(columns=['healthy'])
y = dataset["healthy"].astype(int)
lr = LogisticRegression()
lc = LinearSVC(dual="auto")
km = KMeans(n_clusters=2)

lr_acc_avg = 0
lc_acc_avg = 0
km_acc_avg = 0

# Average accuracy in 20 random training sets
for i in range(N_TRAINING_SETS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=i)

    # Linear regression Classifier
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_y_pred)
    
    # Linear SVC Classifier
    lc.fit(X_train, y_train)
    lc_y_pred = lc.predict(X_test)
    lc_acc = accuracy_score(y_test, lc_y_pred)

    # K-Means Clustering
    km.fit(X_train)
    km_y_pred = km.predict(X_test)
    km_acc = accuracy_score(y_test, km_y_pred)

    print(f"{i}) Logistic Regression Accuracy: {lr_acc:.3f}")
    print(f"{i}) Linear Support Vector Accuracy: {lc_acc:.3f}")
    print(f"{i}) K-Means Accuracy: {km_acc:.3f}")

    lr_acc_avg += lr_acc
    lc_acc_avg += lc_acc
    km_acc_avg += km_acc


lr_acc_avg /= N_TRAINING_SETS
lc_acc_avg /= N_TRAINING_SETS
km_acc_avg /= N_TRAINING_SETS

print(f"Logistic Regression Accuracy Average: {lr_acc_avg:.3f}")
print(f"Linear Support Vector Accuracy Average: {lc_acc_avg:.3f}")
print(f"K-Means Accuracy Average: {km_acc_avg:.3f}")
