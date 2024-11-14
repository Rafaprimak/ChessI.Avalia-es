import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


file_path = "chesscom.csv"
data = pd.read_csv(file_path)


target_variable = "winner"
class_labels = {"white": 1, "black": 0}


data[target_variable] = data[target_variable].map(class_labels)
data = data.dropna(subset=[target_variable])

X = data.drop(columns=[target_variable])
y = data[target_variable]


non_numeric_columns = X.select_dtypes(include=['object']).columns


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for column in non_numeric_columns:
    X[column] = label_encoder.fit_transform(X[column])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}


results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    especificidade = tn / (tn + fp)


    results.append({
        "Model": name,
        "Accuracy": acuracia,
        "F1 Score": f1,

        "Precision": precisao,
        "Recall": recall,
        "Specificity": especificidade
    })


results_df = pd.DataFrame(results)
print(results_df)