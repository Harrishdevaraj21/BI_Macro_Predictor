# 🏦 BI-MACRO Predictor  
### Explainable AI System for Bank Lending Rate Forecasting (India)

🔗 **Live Application:** https://bi-macropredictor.streamlit.app/  
📦 **Source Code Repository:** https://github.com/Harrishdevaraj21/BI_Macro_Predictor  

---

## 📌 Project Overview

**BI-MACRO Predictor** is an **AI-powered decision-support system** designed to forecast **Indian bank lending rates** using a combination of **macroeconomic indicators**, **time-series modeling**, **machine learning**, and **financial sentiment analysis**.

The system goes beyond numeric prediction by providing:
- Explainable forecasts
- Market-condition validation
- Confidence and reliability assessment
- Natural-language economic interpretation

This project is developed as a **final-year academic project** and demonstrates **industry-standard ML deployment practices**.

---

## 🎯 Objectives

- Forecast current bank lending rates based on historical macroeconomic data  
- Combine statistical and machine-learning models for robustness  
- Integrate financial sentiment (FinBERT-inspired logic) for interpretability  
- Provide a **non-technical, user-friendly interface**  
- Deploy a scalable ML application on **Streamlit Cloud**  

---

## 🧠 System Architecture

### 🔹 Models Used
- **VAR (Vector Autoregression)** – Captures temporal dependencies  
- **XGBoost Regressor** – Learns nonlinear macroeconomic relationships  
- **Ensemble Strategy** – Weighted average of VAR and XGBoost outputs  

### 🔹 Explainability Layer
- User-answered economic validation questions  
- FinBERT-inspired sentiment analysis  
- Reliability & consistency scoring  
- Natural-language explanations  

---

## 📊 Dataset Description

- **Time Period:** 1990 – 2025 (monthly)  
- **Total Records:** 429  
- **Key Variables:** Repo Rate, Lending Rate, Inflation, GDP, CPI, Crude Oil Price, USD-INR  

Dataset file:
```
macro_data_monthly_1990_2025_cleaned.csv
```

---

## 🖥️ Application Features

- **Home:** Overview, metrics, historical trends
<img width="1912" height="1024" alt="Screenshot 2026-01-31 121859" src="https://github.com/user-attachments/assets/6697a744-f4e4-4408-a042-d25f82c47325" />
- **Data:** Dataset exploration, filtering, charts  
<img width="1919" height="961" alt="Screenshot 2026-01-31 121929" src="https://github.com/user-attachments/assets/ac7f9d0c-6146-48b7-9029-ec224f227e14" />
- **Model:** Architecture & performance
<img width="1919" height="966" alt="Screenshot 2026-01-31 121951" src="https://github.com/user-attachments/assets/15172ba6-8b99-405c-aa66-b520c4468a84" />
- **Predictions:** Rate forecasting, validation & confidence  
<img width="1917" height="970" alt="Screenshot 2026-01-31 122159" src="https://github.com/user-attachments/assets/6f0a03a9-16f3-4005-a8df-3ce7bb80dd45" />
- **Insights:** Q&A system & market insights  
<img width="1919" height="971" alt="Screenshot 2026-01-31 122236" src="https://github.com/user-attachments/assets/fb43a1df-8efd-4449-95e0-9616e43edbff" />

---

## ⚙️ Deployment Details

- **Platform:** Streamlit Community Cloud  
- **Model Handling:**  
  Large pretrained models are downloaded dynamically at runtime due to Git LFS limitations on Streamlit Cloud.

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- XGBoost, statsmodels (VAR)  
- Pandas, NumPy  
- Git & GitHub  

---

## 📦 Installation & Local Setup

```bash
git clone https://github.com/Harrishdevaraj21/BI_Macro_Predictor.git
cd BI_Macro_Predictor
pip install -r requirements.txt
streamlit run bi_macro_enhanced.py
```

---

## 📈 Model Performance

- **R²:** ~0.94  
- **MAE:** ~0.12  
- **RMSE:** ~0.18  

---

## 🔮 Future Enhancements

- Live news sentiment integration  
- RBI policy document analysis  
- FastAPI backend  
- Model compression  

---

## 👤 Author

**Harrish Devaraj**  
Final-Year Student – AI / DS / ML / Data Analytics  
GitHub: https://github.com/Harrishdevaraj21  

---

## 📜 License

MIT License – Academic & Educational Use
