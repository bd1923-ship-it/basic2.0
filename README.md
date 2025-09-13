💳 Credit Card Fraud Detection (Classification)
🚀 Project Overview
The goal is to build a machine learning model that predicts whether a credit card transaction is fraudulent or genuine.

📊 Problem Statement
Goal: Classify transactions as fraudulent or genuine.
Dataset: Kaggle Credit Card Fraud Dataset
Evaluation Metrics:
Confusion Matrix
Precision
Recall
F1-Score

🛠️ Tech Stack & Dependencies
Python
NumPy
Pandas
Matplotlib, Seaborn
Scikit-learn

Install dependencies with:
pip install -r requirements.txt

📂 Project Structure
📦 Credit-Card-Fraud-Detection
 ┣ 📜 fraud_detection.ipynb   # Google Colab/Jupyter Notebook
 ┣ 📜 requirements.txt        # List of dependencies
 ┣ 📜 README.md               # Project documentation
 ┗ 📂 data/                   # Dataset (optional, if included)

⚙️ Preprocessing Explanation
Before training any model, the dataset must be prepared properly:
1.Feature & Target Split
The dataset has multiple features (like V1, V2, ..., V28, Amount, Time) and one target column Class.
Class = 0 → Genuine transaction
Class = 1 → Fraudulent transaction
We separate them into:
X = data.drop("Class", axis=1)   # Features
y = data["Class"]                # Target


2.Scaling Features
Features such as Time and Amount are on a different scale compared to PCA-transformed features (V1–V28).
To ensure fair training, we use StandardScaler to normalize the values:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


3.Train-Test Split
The dataset is split into 80% training and 20% testing.
Since the dataset is highly imbalanced, we use stratify=y to ensure both training and testing sets have the same proportion of fraud vs genuine cases:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


✅ This preprocessing ensures:
->All features are on the same scale.
->Models are trained fairly.
->The dataset imbalance is handled better during splitting.

📈 Results
->Models trained:
i.Logistic Regression
ii.Decision Tree
iii.Random Forest
->Best performing model: Random Forest (balanced precision & recall)
i.Evaluation Metrics:
ii.Confusion Matrix
iii.Precision, Recall, F1-Score

✅ Conclusion:
->Random Forest performed best on fraud detection.
->Proper preprocessing (scaling + stratified split) was crucial for fair evaluation.
->Future improvements could include SMOTE oversampling, hyperparameter tuning, or deep learning approaches.
  eprocessing was done).
