import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load the dataset
file_path = 'Phishing_Email.csv'
phishing_data = pd.read_csv(file_path)

# Clean and preprocess the dataset
phishing_data = phishing_data[['Email Text', 'Email Type']].dropna()  # Keep relevant columns and remove missing values
phishing_data['Email Type'] = phishing_data['Email Type'].map(
    {'Phishing Email': 1, 'Safe Email': 0})  # Map labels to 1 and 0

# Split the data into features and labels
X = phishing_data['Email Text']
y = phishing_data['Email Type']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Evaluate each model
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred, output_dict=False)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Store results
    results[model_name] = {
        "Model": model,
        "Accuracy": accuracy,
        "Classification Report": classification,
        "Confusion Matrix": conf_matrix
    }

    # Print results
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification)
    print("\n")

# Visualize confusion matrix for each model
for model_name, result in results.items():
    conf_matrix = result["Confusion Matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe Email', 'Phishing Email'],
                yticklabels=['Safe Email', 'Phishing Email'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
