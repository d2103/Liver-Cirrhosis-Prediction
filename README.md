# Liver Cirrhosis Prediction

A machine learning project for predicting liver cirrhosis stages using clinical data. This project implements and compares multiple classification algorithms to predict the severity stage of liver cirrhosis based on patient clinical features.

## Project Overview

Liver cirrhosis is a late stage of scarring (fibrosis) of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism. Early and accurate prediction of cirrhosis stages is crucial for effective treatment planning and patient management.

This project analyzes clinical data from liver cirrhosis patients and applies machine learning techniques to classify patients into three distinct stages of the disease based on their clinical and laboratory features.

## Dataset

The dataset contains clinical records of liver cirrhosis patients with the following characteristics:

- **Total Records**: 25,000 initial records (9,639 unique records after removing duplicates)
- **Features**: 19 clinical features
- **Target Variable**: `Stage` (3 classes: 1, 2, 3 representing different severity stages)

### Features Description

| Feature | Description |
|---------|-------------|
| `N_Days` | Number of days from enrollment |
| `Status` | Patient status (C = Censored, CL = Censored due to liver transplant, D = Death) |
| `Drug` | Treatment type (Placebo, D-penicillamine) |
| `Age` | Patient age in years |
| `Sex` | Patient gender (M/F) |
| `Ascites` | Presence of fluid in the abdomen (Y/N) |
| `Hepatomegaly` | Enlarged liver (Y/N) |
| `Spiders` | Spider angiomata presence (Y/N) |
| `Edema` | Swelling caused by fluid retention (N = No, S = Yes with diuretics, Y = Yes despite diuretics) |
| `Bilirubin` | Bilirubin levels in blood (mg/dl) |
| `Cholesterol` | Cholesterol levels (mg/dl) |
| `Albumin` | Albumin protein levels (gm/dl) |
| `Copper` | Copper levels in urine (μg/day) |
| `Alk_Phos` | Alkaline phosphatase levels (U/L) |
| `SGOT` | Serum glutamic-oxaloacetic transaminase level (U/ml) |
| `Tryglicerides` | Triglycerides levels (mg/dl) |
| `Platelets` | Platelet count per cubic ml/1000 |
| `Prothrombin` | Prothrombin time in seconds |
| `Stage` | **Target variable**: Liver cirrhosis stage (1, 2, 3) |

## Project Structure

```
Liver-Cirrhosis-Prediction/
├── data/
│   └── raw/
│       └── liver_cirrhosis.csv          # Original dataset
├── notebook/
│   └── liver_cirrhosis_stages_classification_full.ipynb  # Main analysis notebook
├── report/
│   └── final_report.pdf                 # Project report (Vietnamese)
└── README.md                            # This file
```

## Machine Learning Models

This project implements and compares three classification algorithms:

### 1. K-Nearest Neighbors (KNN)
- **Best Parameters**: k=7, Manhattan distance, distance-weighted voting
- **Test Accuracy**: ~79%
- **Cross-Validation Accuracy**: ~80%

### 2. Multi-Layer Perceptron (MLP)
- **Architecture**: Configurable hidden layers
- **Best Parameters**: (128, 128) hidden layers, alpha=0.1, learning_rate_init=0.001
- **Test Accuracy**: ~70-76%
- **Activation Function**: ReLU
- **Solver**: Adam optimizer

### 3. Support Vector Machine (SVM)
- **Kernels Tested**:
  - Linear: ~55% accuracy
  - Polynomial: ~72% accuracy
  - RBF (Radial Basis Function): ~74% accuracy (best among SVM variants)
  - Sigmoid: ~55% accuracy

## Data Preprocessing

The following preprocessing steps were applied:

1. **Data Cleaning**:
   - Removed duplicate records (reduced from 25,000 to 9,639 unique records)
   - Checked for missing values (none found)

2. **Feature Engineering**:
   - Converted `Age` from days to years
   - Binary encoding: `Ascites`, `Hepatomegaly`, `Spiders`, `Sex`, `Drug`
   - One-hot encoding: `Status` (Status_C, Status_CL, Status_D), `Edema` (Edema_N, Edema_S, Edema_Y)

3. **Feature Scaling**:
   - StandardScaler applied for all models

## Key Findings

### Clinical Insights

- **Stage Distribution**: The target variable is fairly balanced across the three stages
- **Age Factor**: Patients aged 46-61 show higher risk for Stage 3 cirrhosis
- **Clinical Symptoms**:
  - Ascites (fluid in abdomen) is more common in Stage 3
  - Hepatomegaly (enlarged liver) appears more frequently in Stage 1
  - Spider angiomata is predominantly seen in Stage 3
  - Edema (swelling) is a severe symptom more common in Stage 3

- **Laboratory Markers**:
  - Higher bilirubin levels correlate with advanced stages
  - Lower albumin levels in Stage 3 patients
  - Higher prothrombin time in Stage 3 (indicating blood clotting issues)

### Model Performance

The KNN model with Manhattan distance achieved the best performance (~79% accuracy), suggesting that the clinical features form distinct clusters that can be effectively classified using distance-based methods.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Usage

To run the analysis:

1. Open the Jupyter notebook:
```bash
jupyter notebook notebook/liver_cirrhosis_stages_classification_full.ipynb
```

2. Run all cells to execute the complete analysis pipeline including:
   - Data exploration and visualization
   - Preprocessing and feature engineering
   - Model training and hyperparameter tuning
   - Performance evaluation and comparison

## Results Summary

| Model | Best Test Accuracy | Key Hyperparameters |
|-------|-------------------|---------------------|
| KNN | ~79% | k=7, Manhattan distance, distance weights |
| MLP | ~75% | (128,128) layers, alpha=0.1, ReLU, Adam |
| SVM (RBF) | ~74% | RBF kernel, GridSearchCV optimized |

## Limitations and Future Work

- The dataset shows some class overlap which limits maximum achievable accuracy
- Future improvements could include:
  - Ensemble methods (Random Forest, XGBoost)
  - Deep learning architectures
  - Feature selection techniques
  - SMOTE for handling any class imbalance

## License

This project is for educational and research purposes.

## Contributors

This project was developed as part of a machine learning course assignment.

---

*Last Updated: April 2026*
