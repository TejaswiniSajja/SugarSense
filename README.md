# 🩺 SugarSense: Early Diabetes Risk Screening Using Supervised Machine Learning

## 📌 Project Overview

SugarSense is a machine learning mini-project that estimates the likelihood of diabetes using patient health measurements. It applies supervised machine learning algorithms to identify patterns in health data and compares their performance using multiple evaluation metrics.

The project aims to demonstrate how machine learning can support preliminary risk screening and data-driven analysis. It is intended for educational purposes and is **not a substitute for professional medical advice or diagnosis**.

## 🎯 Objectives

- Understand and preprocess a diabetes-related dataset.
- Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
- Engineer additional features to improve model representation.
- Train and compare multiple supervised machine learning algorithms.
- Tune hyperparameters using cross-validation.
- Evaluate model performance using classification metrics and visualizations.
- Explore decision-threshold tuning to balance recall and precision.
- Build an interactive application for entering patient measurements and viewing model-estimated probabilities.

## 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**, which contains 768 records and 8 original predictor features, along with a binary target variable named `Outcome`.

### Features

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skinfold thickness   |
| Insulin                  | Two-hour serum insulin       |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Age in years                 |
| Outcome                  | Target label: 0 or 1         |

Some zero values in medical measurement columns may represent missing or unrecorded measurements rather than genuine values. These are examined and handled during preprocessing.

## 🛠️ Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook / Google Colab
- **Version Control:** Git and GitHub

## 🔍 Project Workflow

### Phase 1: Data Understanding and Cleaning

- Inspect dataset dimensions, data types, descriptive statistics, and duplicate records.
- Examine the distribution of the target variable.
- Identify suspicious zero values in relevant measurement columns.
- Prepare the data for further analysis.

### Phase 2: Exploratory Data Analysis

- Visualize feature distributions.
- Compare measurements across outcome classes.
- Examine correlations between features.
- Identify potentially useful patterns and relationships.

### Phase 3: Feature Engineering

Create additional features from the original measurements, including:

- `Obesity_Flag`
- `High_Glucose_Flag`
- `Age_BMI_Interaction`

### Phase 4: Data Splitting and Preprocessing

- Separate predictor variables from the target.
- Split the data into training and testing sets using an 80:20 ratio.
- Use stratified splitting to preserve class proportions.
- Apply preprocessing within machine learning pipelines to reduce data leakage.

### Phase 5: Model Training and Comparison

Train and compare the following supervised machine learning algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. XGBoost
6. Support Vector Machine (SVM)

Use stratified five-fold cross-validation to compare model performance.

### Phase 6: Hyperparameter Tuning

Use `RandomizedSearchCV` to explore hyperparameters for selected models and compare their cross-validation ROC-AUC scores.

### Phase 7: Model Evaluation

Evaluate the selected tuned models on the held-out test set using appropriate classification metrics and visualizations, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve and Precision-Recall curve

### Phase 8: Decision-Threshold Analysis

Explore alternative classification thresholds using out-of-fold training predictions. Analyze the trade-off between recall and precision and compare the selected threshold with the default threshold.

### Phase 9: Interpretation and Limitations

Review model performance, discuss important features where supported by the analysis, and document limitations and responsible-use considerations.

## 📈 Evaluation Metrics

| Metric           | Purpose                                                                         |
| ---------------- | ------------------------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correct predictions                                  |
| Precision        | Measures the proportion of positive predictions that are correct                |
| Recall           | Measures the proportion of actual positive cases identified                     |
| F1-score         | Balances precision and recall                                                   |
| ROC-AUC          | Measures how well the model ranks positive cases above negative cases           |
| Confusion Matrix | Summarizes true positives, true negatives, false positives, and false negatives |

Recall is particularly relevant to screening because missed positive cases can be important. However, model selection should consider multiple metrics and the consequences of false positives and false negatives.

## 💻 Streamlit Application

SugarSense includes an interactive Streamlit interface where users can enter the following measurements:

- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI
- Diabetes pedigree function
- Age

The application processes the inputs, derives the additional features required by the model, and displays the model-estimated probability along with an illustrative Low, Medium, or High category.

**Important:** The displayed categories are illustrative software thresholds, not clinically validated medical risk categories. The estimated probability is not a diagnosis and should not be interpreted as an individual's definitive medical risk.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

```

Activate it on Windows:

```bash
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib jupyter

```

### 4. Run the Notebook

Open Jupyter Notebook:

```bash
jupyter notebook

```

Open the project notebook and run the cells in order to explore the data, train the models, and evaluate their performance.

### 5. Run the Streamlit Application

Ensure `app.py` and the compatible saved model file `sugarsense_model.joblib` are in the same directory.

Run:

```bash
python -m streamlit run app.py

```

Streamlit will provide a local URL where you can open the application in your browser.


## 👩‍💻 Author

**Tejaswini Surya Sindhu Sajja**

- GitHub: [TejaswiniSajja](https://github.com/TejaswiniSajja)
- LinkedIn: [Tejaswini Sajja](https://linkedin.com/in/sajja-tejaswini)

