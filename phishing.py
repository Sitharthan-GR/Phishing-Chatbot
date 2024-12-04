import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import glob


# Step 1: Load Enron Dataset
def load_enron_dataset(data_dir):
    print("Loading Enron dataset...")
    emails = []
    for filename in glob.glob(os.path.join(data_dir, '*/*')):
        with open(filename, 'r', encoding='latin1') as file:
            try:
                content = file.read()
                emails.append(content)
            except:
                continue

    print(f"Loaded {len(emails)} emails.")
    return pd.DataFrame(emails, columns=['email_text'])


# Step 2: Create Synthetic Labels
def generate_labels(data, phishing_keywords):
    print("Generating synthetic labels...")
    data['label'] = data['email_text'].apply(
        lambda x: 1 if any(keyword in x.lower() for keyword in phishing_keywords) else 0
    )
    return data


# Step 3: Preprocess and Vectorize
def preprocess_and_vectorize(data):
    print("Cleaning data...")

    # Remove empty emails
    data = data[data['email_text'].str.strip() != '']

    # Remove emails with no meaningful content
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    def is_meaningful(text):
        words = text.split()
        meaningful_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
        return len(meaningful_words) > 0

    data = data[data['email_text'].apply(is_meaningful)]

    # Vectorize the cleaned data
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2)
    X = vectorizer.fit_transform(data['email_text'])
    y = data['label']
    return X, y,



# Step 4: Train and Evaluate the Model
def train_and_evaluate(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model


# Step 5: Interactive Email Classification
def classify_email(model, vectorizer, email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    return "Phishing Email" if prediction == 1 else "Legitimate Email"


# Main Execution
if __name__ == '__main__':
    # Path to Enron dataset (directory with subfolders)
    ENRON_DATASET_PATH = 'emails.csv'  # Update this path

    # Load dataset
    data = load_enron_dataset(ENRON_DATASET_PATH)

    # Define phishing keywords for synthetic labeling
    phishing_keywords = ['verify', 'update', 'account', 'urgent', 'password', 'click', 'login', 'reset']
    data = generate_labels(data, phishing_keywords)

    # Preprocess and vectorize
    X, y, vectorizer = preprocess_and_vectorize(data)

    # Train and evaluate
    model = train_and_evaluate(X, y)

    # Test a sample email
    test_email = "Urgent! Please verify your account by clicking the link below."
    print("Sample Classification:", classify_email(model, vectorizer, test_email))
