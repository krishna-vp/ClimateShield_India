# 🌦️ ClimateShield India

### A Climate-Driven Disease Risk Prediction and Early Warning System

ClimateShield India is a Machine Learning project that predicts **Low, Medium, and High disease risk** for Indian districts using climate and geographical factors such as rainfall, temperature, LAI, location, and disease information.

---

## 📌 Project Objective

To develop an AI-based early warning system that predicts climate-driven disease risk across Indian districts using supervised machine learning techniques.

---

## 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Flask
- Joblib

---

## 📂 Dataset

**Source:** Climate_data.csv

The dataset contains weekly disease outbreak records from Indian districts.

### Features

| Feature | Description |
|----------|-------------|
| week_of_outbreak | Week of disease outbreak |
| state_ut | State / Union Territory |
| district | District name |
| Disease | Disease type |
| day | Day |
| mon | Month |
| year | Year |
| Latitude | Latitude |
| Longitude | Longitude |
| preci | Rainfall |
| LAI | Location Identity Index |
| Temp | Temperature |

### Target Variable

- Low Risk
- Medium Risk
- High Risk

---

## 📊 Exploratory Data Analysis

The project includes:

- Histogram
- Box Plot
- KDE Plot
- Heatmap
- Count Plot
- Pie Chart
- Bar Plot
- Line Plot
- Correlation Analysis

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Best Model

**Random Forest Classifier**

| Metric | Score |
|---------|------|
| Accuracy | **62.71%** |
| Precision | **59.65%** |
| Recall | **62.71%** |
| F1 Score | **56.54%** |

---

## 🌱 Feature Importance

The most influential features identified by Random Forest:

- Rainfall (preci)
- Temperature
- LAI
- Latitude
- Longitude

These climate variables contributed most significantly to disease risk prediction.

---

## 🌐 Flask Deployment

The project includes a Flask web application where users can:

- Select State
- Select District
- Select Disease
- Enter Climate Data
- Predict Disease Risk

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ClimateShield_India.git
cd ClimateShield_India
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask App

```bash
python app.py
```

### 4. Open Browser

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```text
ClimateShield_India/
│
├── ClimateShield_India.ipynb
├── Climate_data.csv
├── app.py
├── ClimateShield_RF_Model.pkl
├── ClimateShield_Scaler.pkl
├── ClimateShield_Feature_Columns.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

