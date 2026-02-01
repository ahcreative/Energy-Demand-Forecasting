# ⚡ Energy Demand Forecasting System

AI-Powered Electricity Demand Prediction & Analysis Platform

![Next.js 14](https://img.shields.io/badge/Next.js-14-blue?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-lightgrey?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-red?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-green?style=flat-square)

---

## 📖 Overview

A comprehensive full-stack application for forecasting electricity demand using advanced machine learning techniques.  
This system combines data preprocessing, clustering analysis, and multiple forecasting models to provide accurate predictions for energy consumption patterns across multiple cities.

The platform integrates weather data, temporal patterns, and historical demand to deliver actionable insights for energy management and grid optimization.

---

## ✨ Key Features

- **🤖 Multiple ML Models:** Linear Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, and SARIMA models for comprehensive forecasting.
- **🔍 Advanced Clustering:** K-Means, DBSCAN, and Hierarchical clustering for demand pattern analysis.
- **📊 Data Visualization:** Interactive charts, PCA/t-SNE visualizations, and comprehensive dashboards.
- **🌦️ Weather Integration:** Real-time weather data processing for accurate demand correlation.
- **🎯 Ensemble Methods:** Averaging and stacking ensemble techniques for improved accuracy.
- **⚡ Real-time Predictions:** Fast API responses with optimized model serving.

---

## 🏗️ Project Structure

### 📱 Nextjs Frontend
- Modern React-based UI
- Interactive data visualizations
- Real-time forecast displays
- Responsive design
- Chart.js integrations

### 🔧 Backend
- Data preprocessing pipeline
- ML model training scripts
- Clustering algorithms
- Feature engineering
- Model evaluation tools

### 🔌 Flask API
- RESTful API endpoints
- Model serving layer
- Request handling
- CORS configuration
- Error handling

---

## 🛠️ Technology Stack

Next.js 14 • React • Python 3.10+ • Flask • TensorFlow • XGBoost • scikit-learn • Pandas • NumPy • Matplotlib • Seaborn

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/energy-demand-forecasting.git
cd energy-demand-forecasting
````

### 2. Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run preprocessing
python preprocessing_and_merging.py

# Run clustering analysis
python clustering.py

# Train forecasting models
python electricity_forecaster.py
```

### 3. Setup Flask API

```bash
cd ../Flaskforconnection
pip install flask flask-cors joblib pandas numpy scikit-learn

# Start Flask server
python app.py
# Server runs on http://localhost:5000
```

### 4. Setup Frontend

```bash
cd ../Nextjs\ Frontend
npm install
# or
yarn install

# Start development server
npm run dev
# or
yarn dev

# Open http://localhost:3000
```

---

## 📡 API Endpoints

| Endpoint        | Method | Description                                  |
| --------------- | ------ | -------------------------------------------- |
| `/api/forecast` | POST   | Get demand forecast for specified parameters |
| `/api/clusters` | GET    | Retrieve clustering analysis results         |
| `/api/models`   | GET    | List available forecasting models            |
| `/api/metrics`  | GET    | Get model performance metrics                |
| `/api/cities`   | GET    | List available cities for forecasting        |

---

## 📊 Data Pipeline

1. **Data Preprocessing**
   `preprocessing_and_merging.py` handles data loading, cleaning, merging multiple sources, feature engineering, normalization, and anomaly detection.

2. **Clustering Analysis**
   `clustering.py` performs dimensionality reduction (PCA, t-SNE), applies K-Means, DBSCAN, and hierarchical clustering, and generates comprehensive visualizations.

3. **Forecasting Models**
   `electricity_forecaster.py` trains multiple models including Linear Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, SARIMA, and ensemble methods.

---

## 📈 Model Performance

The system evaluates models using multiple metrics:

* **MAE (Mean Absolute Error):** Average prediction error
* **RMSE (Root Mean Square Error):** Penalizes larger errors
* **MAPE (Mean Absolute Percentage Error):** Percentage-based accuracy
* **Silhouette Score:** Cluster quality measurement

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```text
1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

Your Name - [@ahcreative](https://github.com/ahcreative)

---

## 🙏 Acknowledgments

* Weather data providers
* Open-source ML libraries
* Energy grid datasets
* Research papers on time series forecasting

---
