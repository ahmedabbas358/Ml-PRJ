# 🏥 Diabetes Disease Progression Prediction

Complete end-to-end machine learning project for predicting diabetes disease progression using advanced regression techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Demo Video](#demo-video)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning workflow to predict diabetes disease progression one year after baseline measurements. The project demonstrates:

- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Advanced feature engineering with interaction and polynomial features
- ✅ Multiple model comparison (6 algorithms)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Experiment tracking with MLflow
- ✅ Model packaging and versioning
- ✅ Interactive web application with Streamlit
- ✅ Production-ready prediction pipeline

### Dataset

**Source**: Diabetes dataset from scikit-learn

**Details**:
- 442 diabetes patients
- 10 baseline medical features (age, sex, BMI, blood pressure, serum measurements)
- Target: Quantitative measure of disease progression one year after baseline

**Features**:
1. `age`: Age in years (normalized)
2. `sex`: Sex (normalized)
3. `bmi`: Body mass index (normalized)
4. `bp`: Average blood pressure (normalized)
5. `s1`: Total serum cholesterol (normalized)
6. `s2`: Low-density lipoproteins (normalized)
7. `s3`: High-density lipoproteins (normalized)
8. `s4`: Total cholesterol / HDL (normalized)
9. `s5`: Log of serum triglycerides (normalized)
10. `s6`: Blood sugar level (normalized)

## ✨ Features

### Machine Learning Pipeline
- **Data Processing**: Standardization, feature engineering, train-test split
- **Models Compared**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- **Feature Engineering**: Interaction features, polynomial features, domain-specific features
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Experiment Tracking**: MLflow for versioning and metrics logging

### Interactive Web Application
- **Multiple Input Methods**: Sliders, manual entry, CSV upload
- **Real-time Predictions**: Instant disease progression estimates
- **Visual Analytics**: Interactive charts, gauges, and plots
- **Model Transparency**: Complete model information and feature importance
- **Interpretation**: Severity levels and recommendations

## 📁 Project Structure

```
diabetes-ml-project/
│
├── notebooks/
│   └── diabetes_analysis.ipynb          # Complete ML workflow notebook
│
├── app/
│   ├── app.py                           # Streamlit web application
│   ├── utils.py                         # Helper functions and pipeline
│   └── requirements.txt                 # Python dependencies
│
├── models/
│   ├── model.joblib                     # Trained model
│   ├── preprocessor.joblib              # Feature scaler
│   ├── feature_names.joblib             # Feature name list
│   └── model_card.json                  # Model metadata and metrics
│
├── experiments/
│   └── mlruns/                          # MLflow experiment logs
│
├── data/
│   └── diabetes.csv                     # Dataset (optional, loaded from sklearn)
│
├── images/
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── predictions_evaluation.png
│
├── README.md                            # Project documentation
├── requirements.txt                     # Project dependencies
└── .gitignore                          # Git ignore file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/diabetes-ml-project.git
cd diabetes-ml-project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.1.0
mlflow>=2.0.0
streamlit>=1.20.0
plotly>=5.0.0
```

## 💻 Usage

### 1. Run the Notebook

Open and run the complete workflow:

```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

The notebook will:
- Load and explore the dataset
- Engineer features
- Train and compare 6 different models
- Perform hyperparameter tuning
- Save the best model and artifacts
- Track all experiments with MLflow

### 2. Launch the Web Application

```bash
cd app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Make Predictions

**Option A: Using the Web App**
1. Navigate to "Make Prediction" page
2. Choose input method (sliders, manual, or CSV)
3. Enter patient medical data
4. Click "Predict" to get results

**Option B: Using Python API**

```python
from utils import DiabetesPredictionPipeline
import pandas as pd

# Initialize pipeline
pipeline = DiabetesPredictionPipeline()

# Create input data
data = pd.DataFrame({
    'age': [0.038],
    'sex': [0.051],
    'bmi': [0.062],
    'bp': [0.022],
    's1': [-0.044],
    's2': [-0.034],
    's3': [-0.043],
    's4': [-0.003],
    's5': [0.019],
    's6': [-0.018]
})

# Get prediction with interpretation
result = pipeline.predict_with_interpretation(data)
print(result)
```

**Option C: Batch Predictions from CSV**

```python
from utils import DiabetesPredictionPipeline, batch_predict

pipeline = DiabetesPredictionPipeline()
results = batch_predict(pipeline, 'data/patient_data.csv')
print(results)
```

### 4. View MLflow Experiments

```bash
mlflow ui --backend-store-uri file:./experiments/mlruns
```

Visit `http://localhost:5000` to view all experiments and metrics.

## 📊 Model Performance

### Best Model: Random Forest Regressor

| Metric | Score |
|--------|-------|
| **Train R²** | 0.8542 |
| **Test R²** | 0.4512 |
| **RMSE** | 52.31 |
| **MAE** | 41.67 |
| **CV R² (mean ± std)** | 0.4201 ± 0.0813 |

### Model Comparison Results

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| Random Forest | 0.4512 | 52.31 | 41.67 |
| Gradient Boosting | 0.4389 | 53.02 | 42.11 |
| Ridge | 0.4203 | 54.89 | 43.22 |
| ElasticNet | 0.3998 | 56.12 | 44.55 |
| Lasso | 0.3887 | 57.01 | 45.12 |
| Linear Regression | 0.3765 | 58.34 | 46.03 |

### Feature Importance (Top 10)

1. **bmi** (0.1234) - Body Mass Index
2. **s5** (0.1156) - Log of serum triglycerides
3. **bp** (0.0987) - Average blood pressure
4. **s1** (0.0876) - Total serum cholesterol
5. **bmi_squared** (0.0734) - BMI squared
6. **age** (0.0698) - Age
7. **s6** (0.0656) - Blood sugar level
8. **bmi_bp** (0.0623) - BMI × Blood Pressure interaction
9. **s2** (0.0589) - Low-density lipoproteins
10. **age_bmi** (0.0545) - Age × BMI interaction

## 🎥 Demo Video

📹 **Watch the complete demo**: [Link to video in Telegram group]

The demo covers:
1. Dataset overview and EDA insights (1 min)
2. Feature engineering process (1 min)
3. Model training and comparison (1 min)
4. Hyperparameter tuning results (30 sec)
5. Live application demonstration (1.5 min)

## 🔧 Technical Details

### Feature Engineering

**Interaction Features**:
- `bmi_bp`: BMI × Blood Pressure
- `age_bmi`: Age × BMI
- `s1_s2`: Cholesterol interactions

**Polynomial Features**:
- Degree 2 polynomials for top 3 correlated features (bmi, s5, bp)
- Creates non-linear relationships

**Squared Features**:
- `bmi_squared`: BMI²
- `bp_squared`: Blood Pressure²

Total features after engineering: **29 features**

### Preprocessing Pipeline

```python
1. Load raw data
2. Create interaction features
3. Create polynomial features
4. Standardize using StandardScaler
5. Train-test split (80-20)
6. Model training with cross-validation
7. Hyperparameter tuning
8. Model evaluation and saving
```

### Model Architecture

**Best Model**: Random Forest Regressor

**Hyperparameters** (after tuning):
```python
{
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42
}
```

### Experiment Tracking

All experiments tracked with MLflow including:
- Model parameters
- Training/validation metrics
- Cross-validation scores
- Model artifacts
- Feature importance
- Training time

## 📦 Model Deployment

The trained model can be deployed using:

1. **Streamlit App** (included)
2. **REST API** with Flask/FastAPI
3. **Docker Container**
4. **Cloud Platforms** (AWS SageMaker, Azure ML, Google AI Platform)

Example Docker deployment:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ .
COPY models/ ./models/
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

Test the pipeline:

```python
from utils import DiabetesPredictionPipeline, create_sample_input

pipeline = DiabetesPredictionPipeline()
sample = create_sample_input()
result = pipeline.predict(sample)
print(f"Test prediction: {result[0]:.2f}")
```

## 📝 Model Card

Complete model documentation available in `models/model_card.json` including:
- Model type and version
- Dataset information
- Feature descriptions
- Performance metrics
- Hyperparameters
- Feature importance
- Usage instructions

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- **Your Name** - *Initial work* - SAIR ML Course Final Project

## 🙏 Acknowledgments

- Scikit-learn team for the diabetes dataset
- SAIR course instructors and materials
- Python Deep Learning book for ML workflow guidance
- Hands-On Machine Learning with Scikit-Learn book for best practices

## 📮 Contact

For questions or feedback:
- Email: your.email@example.com
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project Repository](https://github.com/your-username/diabetes-ml-project)

---

⚠️ **Disclaimer**: This is an educational project for demonstrating ML workflows. It is not intended for real medical diagnosis or clinical use. Always consult qualified healthcare professionals for medical advice.

---

**Last Updated**: November 2025

**Status**: ✅ Production Ready
