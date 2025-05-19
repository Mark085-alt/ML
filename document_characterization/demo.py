import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text
from src.vectorizer import get_vectorizer
from src.model import train_model, evaluate_model

# Load dataset
df = pd.read_csv(r"D:\College Project\Combined\document_characterization\data\documents.csv")
df['text'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = get_vectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)
