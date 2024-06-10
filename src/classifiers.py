import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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
knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier(criterion='gini', max_depth=5)


lr_acc_avg = 0
lc_acc_avg = 0
knn_acc_avg = 0
nb_acc_avg = 0
dt_acc_avg = 0

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

    # K-nearest neighboors
    knn.fit(X_train, y_train)
    knn_y_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_y_pred)

    # Naive Bayes
    nb.fit(X_train, y_train)
    nb_y_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_y_pred)

    # Decision Tree
    dt.fit(X_train, y_train)
    dt_y_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_y_pred)

    lr_acc_avg += lr_acc
    lc_acc_avg += lc_acc
    knn_acc_avg += knn_acc
    nb_acc_avg += nb_acc
    dt_acc_avg += dt_acc


lr_acc_avg /= N_TRAINING_SETS
lc_acc_avg /= N_TRAINING_SETS
knn_acc_avg /= N_TRAINING_SETS
nb_acc_avg /= N_TRAINING_SETS
dt_acc_avg /= N_TRAINING_SETS

print(f"LR Average: {lr_acc_avg:.3f}")
print(f"SVM Average: {lc_acc_avg:.3f}")
print(f"KNN Average: {knn_acc_avg:.3f}")
print(f"NB Average: {nb_acc_avg:.3f}")
print(f"DT Average: {dt_acc_avg:.3f}")
