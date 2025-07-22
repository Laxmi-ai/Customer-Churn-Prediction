# 📉 Telco Churn Prediction App

This is a **Streamlit-based web app** that predicts whether a customer will churn (leave) or not for a telecom company. The app uses a machine learning model trained on the **Telco Customer Churn dataset** from Kaggle.

---

## 🔍 Features

✅ Predict customer churn using demographic and account data  
✅ Interactive dashboard with charts using Plotly  
✅ Model explainability using feature importance  
✅ Secure interface with optional login  
✅ Built with 💻 Python, scikit-learn, XGBoost, and Streamlit

---

## 📊 Demo

👉 **Live App**: [Coming Soon]  
👉 **Demo Video**:  
👉 **Dataset**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📦 Folder Structure

```
telco_churn_app/
│
├── app.py                    # Streamlit frontend app
├── telco_churn_pipeline.py  # Model training pipeline
├── customer_churn_model.pkl # Trained model file
├── encoders.pkl             # Encoders used for preprocessing
├── requirements.txt         # Python dependencies
└── README.md                # You're reading this!
```

---

## 🛠 Installation & Running Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/Laxmi-ai/telco_churn_app.git
cd telco_churn_app
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📈 Model Info

- Algorithm: `RandomForestClassifier` / `XGBoostClassifier`
- Metrics: Accuracy, Confusion Matrix, Classification Report
- Balanced with: SMOTE oversampling
- Encoded Categorical Features
- Feature Importance Visualized

---

## 📌 Deployment

This app is deployed using **[Streamlit Cloud](https://streamlit.io/cloud)**

To deploy:

1. Push this folder to a public GitHub repo
2. Go to Streamlit Cloud → “New App”
3. Select the repo, branch, and `app.py`
4. Click “Deploy”

---

## 🙋‍♀️ Author

**Laxmi Kumari Singh**  
📧 [LinkedIn](https://www.linkedin.com/in/laxmisingh79)  
💻 [GitHub](https://github.com/Laxmi-ai)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
