# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


data = load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


log_reg = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)


y_pred_log_reg = log_reg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_rf = rf.predict(X_test)


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return accuracy, f1, precision, recall

# Log Reg
acc_log, f1_log, prec_log, rec_log = evaluate_model(y_test, y_pred_log_reg)
print("Logistic Regression - Accuracy: {:.2f}, F1-Score: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(acc_log, f1_log, prec_log, rec_log))

# K-Nearest Neighbors 
acc_knn, f1_knn, prec_knn, rec_knn = evaluate_model(y_test, y_pred_knn)
print("KNN - Accuracy: {:.2f}, F1-Score: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(acc_knn, f1_knn, prec_knn, rec_knn))

# Random Forest 
acc_rf, f1_rf, prec_rf, rec_rf = evaluate_model(y_test, y_pred_rf)
print("Random Forest - Accuracy: {:.2f}, F1-Score: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(acc_rf, f1_rf, prec_rf, rec_rf))

# Comparison
best_model = max([('Logistic Regression', acc_log), ('KNN', acc_knn), ('Random Forest', acc_rf)], key=lambda x: x[1])
print(f"The best performing model is: {best_model[0]} with an accuracy of {best_model[1]:.2f}")
