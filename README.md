# Feature Independence Validation Using Chi-Square Test in Machine Learning Models

A comprehensive, beginner-friendly, and interactive Streamlit web application designed for a second-year Computer Science and Engineering (CSE) college micro-project in Mathematics / Statistics for Machine Learning.

---

## 📌 Project Overview

In data preprocessing and feature engineering for machine learning, understanding statistical associations between categorical features is critical. Introducing correlated or redundant features can degrade classification performance, increase computational overhead, and introduce multicollinearity. 

This project provides an interactive graphical interface to validate **feature independence** using Pearson's **Chi-Square ($\chi^2$) Test of Independence**, cross-tabulation contingency tables, Cochran's assumption validation, and Cramer's V effect size computation.

---

## 🎯 Aim and Objectives

### Aim
To design, implement, and evaluate an interactive statistical web application that validates the independence of categorical attributes in academic datasets using the Chi-Square test, assisting in feature selection and redundancy elimination for machine learning models.

### Objectives
1. **Contingency Analysis**: Automatically construct observed joint frequency distributions and calculate marginal row/column proportions using `pandas.crosstab()`.
2. **Feature Binning**: Provide dynamic, user-configurable discretization methods (custom cutoffs, quantile splits, and equal-width intervals) to convert numerical continuous attributes into meaningful discrete categories.
3. **Hypothesis Testing**: Implement Pearson's Chi-Square Test using `scipy.stats.chi2_contingency()`, computing the test statistic ($\chi^2$), p-value, degrees of freedom ($df$), and critical values.
4. **Assumption Audit (Cochran's Rule)**: Explicitly evaluate validity assumptions ($\ge 80\%$ of cells having expected count $\ge 5$, and no cell $< 1$) and alert users to small-sample approximation risks.
5. **Association Strength**: Compute **Cramer's V** to quantify practical effect size independently of sample size.
6. **Interactive Visualization**: Render responsive, interactive Plotly visualizations (Grouped Bar Charts and Contingency Heatmaps).
7. **Report Generation**: Export comprehensive mathematical reports and cross-tabulation CSV files for documentation.

---

## 📐 Statistical Theory & Mathematical Formulations

### 1. Hypotheses Formulation
- **Null Hypothesis ($H_0$):** The two categorical variables are **statistically independent** (no relationship in category distributions).
- **Alternative Hypothesis ($H_1$):** The two categorical variables are **associated / dependent** (a statistically significant relationship exists).

### 2. Contingency Table (Cross-Tabulation)
For two categorical variables $X$ (with $r$ rows) and $Y$ (with $c$ columns), let $O_{ij}$ denote the observed frequency in cell $(i, j)$:

$$\text{Total Sample Size } N = \sum_{i=1}^{r} \sum_{j=1}^{c} O_{ij}$$

Row totals $R_i = \sum_{j=1}^{c} O_{ij}$ and Column totals $C_j = \sum_{i=1}^{r} O_{ij}$.

### 3. Expected Frequencies ($E_{ij}$)
Under the assumption of independence ($H_0$), the probability of an observation falling into cell $(i, j)$ is $P(X = i) \times P(Y = j)$. The expected cell count is:

$$E_{ij} = \frac{R_i \times C_j}{N}$$

### 4. Chi-Square Test Statistic ($\chi^2$)
Pearson's test statistic measures the sum of squared standardized deviations:

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Under $H_0$, $\chi^2$ asymptotically follows a Chi-Square distribution with degrees of freedom:

$$df = (r - 1) \times (c - 1)$$

### 5. Decision Rule
At a user-selected significance level $\alpha$ (e.g., $0.01$, $0.05$, $0.10$):
- **If $p\text{-value} < \alpha$ (or $\chi^2 > \chi^2_{\text{critical}}$):** **Reject $H_0$**. Statistically significant evidence of association.
- **If $p\text{-value} \ge \alpha$ (or $\chi^2 \le \chi^2_{\text{critical}}$):** **Fail to Reject $H_0$**. Insufficient evidence to establish association.

> **Important Academic Caveats:**
> - Failing to reject $H_0$ does **not** prove absolute independence; it merely means any existing association was undetectable given this sample size.
> - A statistically significant association does **not** imply a cause-and-effect relationship (*Correlation $\neq$ Causation*).

### 6. Cochran's Rule (1954) for Validity
The Chi-Square test is an asymptotic large-sample approximation. The test is valid when:
1. No expected frequency is less than 1.0 ($E_{ij} \ge 1$).
2. No more than 20% of the cells have expected frequencies less than 5.0.

### 7. Effect Size: Cramer's V
Because a large sample size ($N$) can make trivial associations statistically significant, Cramer's V measures the practical magnitude of the association:

$$V = \sqrt{\frac{\chi^2}{N \times \min(r - 1, c - 1)}}$$

| Value of $V$ (for $df^* = 1$) | Association Strength |
| :--- | :--- |
| $< 0.10$ | Negligible / Trivial |
| $0.10 \le V < 0.30$ | Small / Weak |
| $0.30 \le V < 0.50$ | Moderate |
| $\ge 0.50$ | Strong / Large |

---

## 🛠️ Tools and Technologies

- **Programming Language:** Python 3.10+
- **Application Framework:** Streamlit (Dynamic, responsive web dashboard)
- **Data Manipulation:** Pandas & NumPy
- **Statistical Engine:** SciPy (`scipy.stats.chi2_contingency`, `scipy.stats.chi2.ppf`)
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects

---

## 📁 Project Structure

```text
Student_Performance_Prediction/
├── app.py                             # Main Streamlit web application & statistical engine
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
├── dataset/
│   ├── student_data.csv               # Original student academic dataset (42 records)
│   └── demonstration_survey_data.csv  # Large synthetic academic survey dataset (350 records)
└── results/                           # Output directory for exported analysis reports
```

---

## 💻 Installation & Setup Guide (Windows)

Follow these step-by-step instructions in PowerShell or Command Prompt:

### 1. Open Terminal and Navigate to Project Directory
```powershell
cd d:\Documents\MATHS_3\micro_project\Student_Performance_Prediction
```

### 2. Create a Virtual Environment (Recommended)
Creating an isolated virtual environment prevents library conflicts:
```powershell
# Using Python directly or Windows py launcher
py -m venv venv
```

### 3. Activate the Virtual Environment
- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution policies block the script, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **Command Prompt (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### 4. Install Dependencies
Install all required libraries using `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*(Or without a venv: `py -m pip install -r requirements.txt`)*

---

## 🚀 How to Run the Dashboard

To launch the web application, execute:

```powershell
streamlit run app.py
```
*(Or: `py -m streamlit run app.py`)*

Once started, the application will automatically open in your default web browser at:
```text
http://localhost:8501
```

---

## 🖥️ How to Use the Dashboard

1. **Select or Upload Dataset**:
   - Use the **sidebar** to toggle between the default **Student Academic Dataset (42 records)**, the **Demonstration Academic Survey (350 records)**, or upload a custom `.csv` file.
2. **Configure Significance Level ($\alpha$)**:
   - Choose $\alpha = 0.01$ (conservative), $\alpha = 0.05$ (standard), or $\alpha = 0.10$.
3. **Inspect Dataset Overview**:
   - Review total records, feature count, missing cells, data types, and preview the raw data table.
4. **Choose Variables**:
   - Select **Feature 1 (Row variable)** and **Feature 2 (Column variable)**.
   - If a selected variable is numerical (e.g., `Attendance` or `Study_Hours`), configure the bin boundaries (cutoffs, quantiles, or equal intervals) to convert it into meaningful categories.
5. **Analyze Results**:
   - Review the observed cross-tabulation table with row and column totals.
   - Inspect the calculated $\chi^2$ statistic, degrees of freedom, p-value, and critical value.
   - Read the automated hypothesis decision and interpretation.
   - Check the **Cochran's Assumption Audit** to verify sample size adequacy.
   - Review **Cramer's V** to evaluate the practical strength of association.
6. **Explore Visualizations**:
   - Interact with the Plotly **Grouped Bar Chart** (toggle counts vs within-group percentages).
   - Inspect the **Contingency Heatmap** (switch between observed, expected, and standardized residuals).
7. **Export Reports**:
   - Download the full plaintext statistical summary report or export the contingency tables as CSV files.

---

## 📊 How to Interpret the Results

| Condition | P-Value vs $\alpha$ | Decision | Meaning |
| :--- | :--- | :--- | :--- |
| **Significant** | $p < \alpha$ | **Reject $H_0$** | Strong sample evidence that the two features are associated. |
| **Not Significant** | $p \ge \alpha$ | **Fail to Reject $H_0$** | Insufficient evidence; distributions do not deviate significantly from random chance. |

### Context in Machine Learning
- **Feature vs Target (Feature Selection):** A significant association indicates that the categorical feature contains discriminative signal for predicting the target class (similar to scikit-learn's `SelectKBest(score_func=chi2)`).
- **Feature vs Feature (Redundancy Check):** A very high Chi-Square and Cramer's V between two input features reveals redundancy. Dropping or merging one of the features can reduce multicollinearity and improve model generalization.

---

## ⚠️ Limitations of the Chi-Square Test

1. **Sample Size Sensitivity**:
   - In small samples ($N < 50$), expected cell counts often drop below 5, violating Cochran's criterion and inflating Type I error rates.
   - In extremely large samples ($N > 100,000$), even practically trivial differences achieve statistical significance ($p < 0.001$), which is why **Cramer's V** must always accompany p-values.
2. **Categorization Loss of Information**:
   - Discretizing continuous variables into discrete bins causes loss of granularity and can be sensitive to where boundaries are placed.
3. **Correlation, Not Causation**:
   - Chi-Square establishes non-random association, not directional causation. Unobserved confounding variables may drive the relationship.
4. **Order Ignorance**:
   - The standard Chi-Square test treats categories as nominal; it does not take into account ordinal hierarchies (e.g., Low < Medium < High).

---

## 🎓 College Viva / Demo Guide

When presenting this micro-project to your professor or external examiner:

1. **Introduce the Motivation:** Explain why feature independence validation is an essential mathematical step in machine learning before feeding categorical columns into algorithms.
2. **Demonstrate with Original Data (42 records):** Show `Attendance` vs `Performance`. Point out that the test achieves statistical significance, but emphasize your statistical maturity by highlighting the **Cochran's Rule Warning** (due to $N=42$).
3. **Demonstrate with the 350-record Dataset:** Switch to the demonstration survey in the sidebar to show how larger samples satisfy Cochran's assumptions ($0$ cells $< 5$) and cleanly validate feature pairs (e.g., `Attendance_Level` vs `Performance_Band`).
4. **Show Significance Level Impact:** Switch $\alpha$ from $0.05$ to $0.10$ on boundary cases (like `Learning_Style` vs `Performance_Band`) to explain how the critical threshold alters the decision.
5. **Highlight Effect Size vs P-Value:** Clarify that p-value tests *statistical significance* (is the effect real?), while Cramer's V tests *practical magnitude* (how strong is the relationship?).
6. **Export the Report:** Download the generated plaintext report to show a complete, publication-ready summary of findings.
#   F e a t u r e - I n d e p e n d e n c e - V a l i d a t o r  
 #   F e a t u r e - I n d e p e n d e n c e - V a l i d a t o r  
 