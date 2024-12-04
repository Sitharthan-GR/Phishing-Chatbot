import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# Load the dataset
file_path = 'Phishing_Email.csv'
phishing_data = pd.read_csv(file_path)

# Clean and preprocess the dataset
phishing_data = phishing_data[['Email Text', 'Email Type']].dropna()  # Keep relevant columns and remove missing values
phishing_data['Email Type'] = phishing_data['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})  # Map labels to 1 and 0

# Split the data into features and labels
X = phishing_data['Email Text']
y = phishing_data['Email Type']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to classify emails
def classify_email(email_text):
    email_vectorized = vectorizer.transform([email_text])  # Convert text to TF-IDF features
    prediction = model.predict(email_vectorized)[0]  # Predict using the trained model
    if prediction == 1:
        return "Phishing Email"
    else:
        return "Safe Email"

# Gradio interface
description = """
This chatbot detects whether an email is a phishing attempt or a legitimate email. Enter the email text below to classify.
"""
interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, placeholder="Enter email content here..."),
    outputs="text",
    title="Phishing Email Detector",
    description=description
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
