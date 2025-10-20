# Loan Approval Prediction

## Project Overview
This project is focused on building a **Machine Learning model** to predict whether a loan application will be **approved or rejected** based on applicant data. The goal is to assist financial institutions in making informed and accurate loan decisions by leveraging historical loan application data.

The project handles **missing values**, encodes categorical features, addresses **class imbalance**, and compares multiple models, including **Logistic Regression**, **Decision Tree**, and **XGBoost**, evaluated using **precision, recall, and F1-score** metrics.

---

## Dataset
- **Source:** [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)  
- **File used:** `loan.csv`  
- **Number of rows:** 614  
- **Number of features:** 13  
- **Target variable:** `Loan_Status` (1 = Approved, 0 = Rejected)  

**Key features:**
- Gender, Married, Dependents, Education, Self_Employed  
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term  
- Credit_History, Property_Area

---

## Project Structure
loan-approval-prediction/
├── data/
│ └── loan.csv # Raw dataset
├── notebooks/
│ └── loan_approval.ipynb # Jupyter notebook with all steps
├── README.md # Project description
├── requirements.txt # Project dependencies
└── .gitignore # Files/folders to ignore in Git

---

## Steps Performed

1. **Data Loading and Exploration**
   - Loaded dataset into Pandas DataFrame  
   - Checked shape, missing values, duplicates, and basic statistics  

2. **Data Preprocessing**
   - Handled missing values using **median** for numerical and **mode** for categorical columns  
   - Dropped irrelevant columns (`Loan_ID`)  
   - Encoded categorical variables using `LabelEncoder`  
   - Scaled numerical features using `MinMaxScaler`  
   - Checked for class imbalance  

3. **Handling Imbalanced Data**
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset  
   - Ensured equal representation of approved and rejected loans for training  

4. **Model Training**
   - Trained multiple models:
     - **Logistic Regression**
     - **Decision Tree Classifier**
     - **XGBoost Classifier**  
   - Hyperparameters tuned for performance  

5. **Model Evaluation**
   - Metrics used:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**  
   - Visualized predictions using **confusion matrix**

---

## Tools & Libraries
- **Python 3.x**  
- **Jupyter Notebook**  
- **Pandas & NumPy** for data manipulation  
- **Scikit-learn** for preprocessing, model building, and evaluation  
- **XGBoost** for gradient boosting classifier  
- **Imbalanced-learn** for SMOTE  
- **Seaborn & Matplotlib** for visualization  

---

## Installation & Setup

1. Clone this repository
```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
2. Create a new conda environment and activate it
conda create -n loan-env python=3.10 -y
conda activate loan-env
3. Install required dependencies
pip install -r requirements.txt
4. Launch Jupyter Notebook
jupyter notebook
               
