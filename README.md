# Feature Independence Validation Using Chi-Square Test

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6.svg)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-8%20Passed-success.svg)]()

An interactive statistical computing and hypothesis-testing web application developed for the **B.Tech Computer Science and Engineering (CSE) Mathematics & Statistics** micro-project. Built using **Python**, **Streamlit**, **Pandas**, **NumPy**, **SciPy**, and **Plotly**, this tool validates the statistical independence of categorical and discretized numerical features using **Pearson's Chi-Square Test of Independence ($\chi^2$)**.

---

## Table of Contents

1. [Project Overview & Academic Purpose](#1-project-overview--academic-purpose)
2. [Problem Statement](#2-problem-statement)
3. [Key Features & Capabilities](#3-key-features--capabilities)
4. [Mathematical Formulations & Hypothesis Testing](#4-mathematical-formulations--hypothesis-testing)
5. [Worked Numerical Example (Step-by-Step)](#5-worked-numerical-example-step-by-step)
6. [Discretization Strategies for Numerical Data](#6-discretization-strategies-for-numerical-data)
7. [Assumption Auditing & Cochran's Criteria](#7-assumption-auditing--cochrans-criteria)
8. [Project Folder Structure](#8-project-folder-structure)
9. [Dataset Descriptions & Verified Integrity](#9-dataset-descriptions--verified-integrity)
10. [Setup & Execution Guide (PowerShell / Windows)](#10-setup--execution-guide-powershell--windows)
11. [Application Dashboard Walkthrough](#11-application-dashboard-walkthrough)
12. [Automated Unit Testing Suite](#12-automated-unit-testing-suite)
13. [Limitations & Future Enhancements](#13-limitations--future-enhancements)
14. [Git Repository Workflow](#14-git-repository-workflow)

---

## 1. Project Overview & Academic Purpose

- **Project Title:** Feature Independence Validation Using Chi-Square Test
- **Academic Program:** B.Tech Computer Science & Engineering (CSE)
- **Course Domain:** Engineering Mathematics-III / Applied Statistics & Probability
- **Repository:** [`Jetson17/Feature-Independence-Validator`](https://github.com/Jetson17/Feature-Independence-Validator.git)

In exploratory data analysis and machine learning preprocessing, verifying feature independence is a critical preliminary step. Introducing correlated or mutually dependent categorical features increases model complexity, inflates multicollinearity, and distorts parameter estimates.

This application provides an interactive, mathematically rigorous environment where students, data scientists, and researchers can:
- Cross-tabulate categorical variables into contingency matrices.
- Dynamically discretize continuous numeric attributes into discrete intervals.
- Perform Pearson's Chi-Square Test of Independence with configurable significance levels ($\alpha$).
- Audit asymptotic test assumptions using **Cochran's (1954) criteria**.
- Automatically deploy exact test corrections (**Fisher's Exact Test** and **Yates' Continuity Correction**) when sample sizes are small.
- Measure practical association strength via **Cramér's V** effect size.
- Visualize bivariate distributions through dynamic Plotly charts and heatmaps.
- Export comprehensive plaintext diagnostic audits and CSV matrices for academic reporting.

> **Important Clarification:** This application is strictly an **inferential statistical hypothesis-testing and diagnostic engine**. It does not fit predictive machine learning models (e.g., Random Forests, Logistic Regression, or Neural Networks).

---

## 2. Problem Statement

Modern data pipelines frequently incorporate variables without empirical validation of their bivariate relationships:
1. **Feature Redundancy:** Correlated categorical attributes introduce redundant dimensions that inflate model variance without adding predictive signal.
2. **Small-Sample Approximation Collapse:** Traditional $\chi^2$ tests rely on asymptotic $\chi^2$ approximations. When cell expectations fall below 5, Type I error rates inflate, causing false positives.
3. **Conflating Significance with Practical Importance:** Statistical significance ($p < \alpha$) is heavily sample-size dependent; huge sample sizes can produce tiny $p$-values for trivial real-world associations. A standardized effect size (Cramér's V) is required to decouple magnitude from sample size.

This project addresses these challenges with an automated diagnostic workflow combining hypothesis testing, assumption audits, and effect size measurement.

---

## 3. Key Features & Capabilities

- **Dual Preloaded Datasets & Custom File Ingestion:**
  - Original 42-record student performance dataset (`dataset/student_data.csv`).
  - Synthetic 350-record survey dataset (`dataset/demonstration_survey_data.csv`) satisfying large-sample criteria.
  - User CSV uploader with real-time column detection and null-value handling.
- **Configurable Significance Levels ($\alpha$):** Instant recalculation for $\alpha = 0.01$ (conservative), $\alpha = 0.05$ (standard), or $\alpha = 0.10$ (exploratory).
- **Automated Feature Categorization & Binning:**
  - Automatic detection of numeric vs. categorical types.
  - Configurable Cutoffs discretization: Partition continuous values into 2, 3, or 4 categories with custom thresholds and domain presets.
  - Category consolidation toggle to merge rare classes (counts $< 2$) into an `"Other"` bin.
- **Contingency Matrix Generation:**
  - Observed count cross-tabulations with marginal totals ($R_i$, $C_j$, $N$).
- **Hypothesis Testing Engine:**
  - Calculates $\chi^2$, degrees of freedom ($df$), and $p$-value with high precision.
  - Clear visual decision banners (High-contrast monochrome panels for Reject $H_0$ and Fail to Reject $H_0$).
- **Interactive Visualizations:**
  - Grouped bar charts toggleable between absolute counts and conditional percentages (with original colors).
  - Multi-metric heatmaps displaying observed counts, expected counts, or standardized residuals.
- **Multi-Format Export Suite:**
  - Downloadable `.txt` audit reports with timestamps and complete testing outputs.
  - Downloadable observed and expected contingency tables in `.csv` format.

---

## 4. Mathematical Formulations & Hypothesis Testing

### 4.1 Statistical Hypotheses
For any two categorical attributes $X$ (with $r$ levels) and $Y$ (with $c$ levels):
- **Null Hypothesis ($H_0$):** Variables $X$ and $Y$ are **statistically independent**.
  $$P(X = i \cap Y = j) = P(X = i) \cdot P(Y = j) \quad \forall \; i, j$$
- **Alternative Hypothesis ($H_1$):** Variables $X$ and $Y$ are **statistically dependent / associated**.

### 4.2 Marginal Sums and Grand Total
For an $r \times c$ cross-tabulation table with observed cell counts $O_{ij}$:
- Row totals:
  $$R_i = \sum_{j=1}^c O_{ij}$$
- Column totals:
  $$C_j = \sum_{i=1}^r O_{ij}$$
- Grand sample size:
  $$N = \sum_{i=1}^r \sum_{j=1}^c O_{ij} = \sum_{i=1}^r R_i = \sum_{j=1}^c C_j$$

### 4.3 Expected Frequencies ($E_{ij}$)
Under the assumption of independence ($H_0$), the theoretical frequency expected in cell $(i, j)$ is:
$$E_{ij} = \frac{R_i \cdot C_j}{N}$$

### 4.4 Pearson's Chi-Square Test Statistic ($\chi^2$)
Measures the standardized squared deviation between observed and expected cell counts:
$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

### 4.5 Degrees of Freedom ($df$)
$$df = (r - 1) \cdot (c - 1)$$

### 4.6 Decision Rule
Let $\chi^2_{\text{crit}} = F_{\chi^2, df}^{-1}(1 - \alpha)$ denote the critical value from the Chi-Square cumulative distribution function:
- **If $p\text{-value} < \alpha$ (or $\chi^2 > \chi^2_{\text{crit}}$):** **Reject $H_0$**. Conclude there is statistically significant evidence of association.
- **If $p\text{-value} \ge \alpha$ (or $\chi^2 \le \chi^2_{\text{crit}}$):** **Fail to Reject $H_0$**. Insufficient evidence to reject independence; observed deviations are consistent with random chance.

### 4.7 Standardized Pearson Residuals ($r_{ij}$)
Standardized residuals identify which specific cell combinations drive the association:
$$r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}$$
- Values of $|r_{ij}| > 2.0$ indicate deviations that are statistically significant at roughly the $95\%$ confidence level ($p \approx 0.05$).
- Values of $|r_{ij}| > 3.0$ indicate extreme divergence ($p < 0.003$).

### 4.8 Cramér's V Effect Size
To measure the practical strength of association independent of sample size $N$:
$$V = \sqrt{\frac{\chi^2}{N \cdot \min(r - 1, c - 1)}}$$
where $k = \min(r - 1, c - 1)$. Association strength is classified using Cohen (1988) and Rea & Parker benchmark guidelines:

| Qualitative Strength | $k = 1$ ($2 \times 2$) | $k = 2$ ($3 \times 3$) | $k \ge 3$ (Larger) |
| :--- | :--- | :--- | :--- |
| **Negligible** | $V < 0.10$ | $V < 0.07$ | $V < 0.06$ |
| **Weak** | $0.10 \le V < 0.30$ | $0.07 \le V < 0.21$ | $0.06 \le V < 0.17$ |
| **Moderate** | $0.30 \le V < 0.50$ | $0.21 \le V < 0.35$ | $0.17 \le V < 0.29$ |
| **Strong** | $V \ge 0.50$ | $V \ge 0.35$ | $V \ge 0.29$ |

---

## 5. Worked Numerical Example (Step-by-Step)

The following representative calculation corresponds directly to the verified unit test case `test_representative_calculation_attendance_vs_performance` using `dataset/student_data.csv` ($N = 42$).

### 5.1 Variable Partitioning
- **Row Variable ($X$):** `Attendance` discretized at cutoff $75.0$:
  - Category 1: `High (≥75.0)` ($28$ students)
  - Category 2: `Low (<75.0)` ($14$ students)
- **Column Variable ($Y$):** `Performance` with 4 categories: `Average`, `Excellent`, `Good`, `Poor`.

### 5.2 Observed Contingency Matrix ($O_{ij}$)

| Attendance Category | Average | Excellent | Good | Poor | **Row Total ($R_i$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High (≥75.0)** | 4 | 12 | 12 | 0 | **28** |
| **Low (<75.0)** | 8 | 0 | 0 | 6 | **14** |
| **Column Total ($C_j$)** | **12** | **12** | **12** | **6** | **$N = 42$** |

### 5.3 Expected Frequency Matrix ($E_{ij}$)
Using $E_{ij} = (R_i \cdot C_j) / N$:

- $E_{1,1} = (28 \cdot 12) / 42 = 8.0000$
- $E_{1,2} = (28 \cdot 12) / 42 = 8.0000$
- $E_{1,3} = (28 \cdot 12) / 42 = 8.0000$
- $E_{1,4} = (28 \cdot 6) / 42 = 4.0000$
- $E_{2,1} = (14 \cdot 12) / 42 = 4.0000$
- $E_{2,2} = (14 \cdot 12) / 42 = 4.0000$
- $E_{2,3} = (14 \cdot 12) / 42 = 4.0000$
- $E_{2,4} = (14 \cdot 6) / 42 = 2.0000$

### 5.4 Cell Deviations $\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$

| Cell $(i, j)$ | $O_{ij}$ | $E_{ij}$ | $O_{ij} - E_{ij}$ | $(O_{ij} - E_{ij})^2$ | $\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $(1, 1)$ | 4 | 8.0 | -4.0 | 16.0 | **2.0000** |
| $(1, 2)$ | 12 | 8.0 | +4.0 | 16.0 | **2.0000** |
| $(1, 3)$ | 12 | 8.0 | +4.0 | 16.0 | **2.0000** |
| $(1, 4)$ | 0 | 4.0 | -4.0 | 16.0 | **4.0000** |
| $(2, 1)$ | 8 | 4.0 | +4.0 | 16.0 | **4.0000** |
| $(2, 2)$ | 0 | 4.0 | -4.0 | 16.0 | **4.0000** |
| $(2, 3)$ | 0 | 4.0 | -4.0 | 16.0 | **4.0000** |
| $(2, 4)$ | 6 | 2.0 | +4.0 | 16.0 | **8.0000** |

### 5.5 Test Statistics & Decision Summary
- **Chi-Square Statistic:**
  $$\chi^2 = 2.0 + 2.0 + 2.0 + 4.0 + 4.0 + 4.0 + 4.0 + 8.0 = \mathbf{30.0000}$$
- **Degrees of Freedom:**
  $$df = (2 - 1) \cdot (4 - 1) = \mathbf{3}$$
- **Critical Value ($\alpha = 0.05$):**
  $$\chi^2_{0.05, 3} = \mathbf{7.8147}$$
- **$P$-Value:**
  $$p = 1 - F_{\chi^2, 3}(30.0) \approx \mathbf{1.38 \times 10^{-6}} \quad (p < 0.0001)$$
- **Statistical Decision:** Because $\chi^2 = 30.0000 > 7.8147$ and $p < 0.05$, **Reject $H_0$** (Statistically Significant Association).
- **Cramér's V:**
  $$V = \sqrt{\frac{30.0}{42 \cdot \min(2 - 1, 4 - 1)}} = \sqrt{\frac{30}{42}} \approx \mathbf{0.8452} \quad \text{(Strong Association)}$$
- **Cochran Audit Result:** 4 out of 8 cells ($50.0\%$) have $E_{ij} < 5.0$, exceeding the $20\%$ ceiling. The dashboard triggers a diagnostic advisory recommending category consolidation or a larger sample.

---

## 6. Discretization Strategy for Numerical Data

When a user selects a continuous numeric column (e.g., `Attendance`, `Study_Hours`, `Exam_Score`), the application provides configurable cutoffs to discretize continuous values into meaningful academic categories:

```
Continuous Numerical Column
  └── Configurable Cutoffs (Custom Splits)
         ├── 2 Bins: Split at user threshold (e.g. Attendance ≥ 75.0)
         ├── 3 Bins: Low / Medium / High with dual threshold inputs
         └── 4 Bins: Tier 1 to Tier 4 across 3 thresholds
```

### Rare Category Consolidation
If categorical columns contain sparse classes, an optional toggle consolidates categories with observed counts $< 2$ into an `"Other"` group, helping prevent zero cells and sparse expectations.

---

## 7. Assumption Auditing & Cochran's Criteria

### Cochran's (1954) Rules
Because Pearson's Chi-Square statistic relies on a continuous theoretical distribution approximating discrete multinomial counts, test validity requires:
1. **Minimum Cell Expectation:** $E_{ij} \ge 1.0$ for **all** cells.
2. **Sparse Cell Constraint:** No more than **$20\%$** of cells should have $E_{ij} < 5.0$.

### Diagnostic Safeguards Implemented
- **Prominent Alert Box:** Automatically calculated and displayed in Section 5 of the dashboard.
- **Automated Fallback for $2 \times 2$ Tables:**
  - **Fisher's Exact Test:** Computes the exact hypergeometric probability without relying on large-sample approximations.
  - **Yates' Continuity Correction:** Subtracts $0.5$ from absolute deviations $|O_{ij} - E_{ij}|$ to reduce Type I error inflation.
- **Guidance for Larger Tables:** Recommends consolidating sparse categories (Section 2) or switching to larger datasets.

---

## 8. Project Folder Structure

```text
Student_Performance_Prediction/
│
├── dataset/
│   ├── student_data.csv               # Primary academic records (42 student records)
│   └── demonstration_survey_data.csv  # Large-sample survey dataset (350 survey records)
│
├── results/                           # Destination directory for exported reports/tables
├── venv/                              # Python isolated virtual environment
│
├── app.py                             # Main Streamlit web application & UI engine
├── test_statistical_engine.py         # Automated unit test suite (8 test cases)
├── requirements.txt                   # Production dependencies with minimum version bounds
├── .gitignore                         # Exclusions for virtual envs, pycache, and exports
└── README.md                          # Complete project documentation & guide
```

---

## 9. Dataset Descriptions & Verified Integrity

### 9.1 Student Academic Dataset (`dataset/student_data.csv`)
- **Row Count:** Exactly **42 data rows** (excluding header).
- **Attributes (5):**
  - `Attendance`: Numeric float/int ($60.0\%$ to $98.0\%$, Mean: $79.86\%$)
  - `Study_Hours`: Numeric float ($1.2$ to $5.0$ hrs/day, Mean: $3.01$ hrs)
  - `Previous_Marks`: Numeric integer ($52$ to $95$, Mean: $72.69$)
  - `Assignment_Score`: Numeric integer ($55$ to $97$, Mean: $75.55$)
  - `Performance`: Categorical target (`'Excellent'`: 12, `'Good'`: 12, `'Average'`: 12, `'Poor'`: 6)
- **Role:** Demonstrates small-sample conditions ($N = 42$) where sparse cells trigger Cochran's diagnostic alerts.

### 9.2 Demonstration Academic Survey (`dataset/demonstration_survey_data.csv`)
- **Row Count:** Exactly **350 data rows** (excluding header).
- **Attributes (8):**
  - `Attendance_Level`: Categorical (`'High (>=85%)'`, `'Moderate (70-84%)'`, `'Low (<70%)'`)
  - `Study_Environment`: Categorical (`'Quiet/Library'`, `'Home/Moderate'`, `'Noisy'`)
  - `Parent_Education`: Categorical (`'Bachelor Degree'`, `'High School'`, `'Postgraduate'`)
  - `Extracurricular`: Categorical (`'Participating'`, `'Not Participating'`)
  - `Learning_Style`: Categorical (`'Visual'`, `'Auditory'`, `'Kinesthetic'`)
  - `Weekly_Study_Hours`: Numeric float ($3.8$ to $28.0$ hrs/week)
  - `Exam_Score`: Numeric integer ($35$ to $95$)
  - `Performance_Band`: Categorical (`'Distinction'`, `'First Class'`, `'Second Class'`, `'Needs Improvement'`)
- **Role:** Demonstrates large-sample testing ($N = 350$) where cell frequencies satisfy Cochran's validity constraints without warnings.

---

## 10. Setup & Execution Guide (PowerShell / Windows)

Follow these exact steps in **Windows PowerShell** to activate the virtual environment and run the dashboard.

### Step 1: Open PowerShell and Navigate to the Repository Folder
Ensure your working directory is the repository folder containing `app.py`:

```powershell
# Navigate into the project folder
cd D:\Documents\MATHS_3\micro_project\Student_Performance_Prediction\Student_Performance_Prediction
```

> [!IMPORTANT]
> If your project was extracted into a nested directory (e.g. `...\Student_Performance_Prediction\Student_Performance_Prediction`), ensure you `cd` into the inner folder where `app.py` and `venv` reside.

### Step 2: Configure Script Execution Policy (If Needed)
If PowerShell displays `running scripts is disabled on this system`, bypass execution policy for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3: Activate the Virtual Environment
Activate the pre-configured virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```
*(When activated, your terminal prompt will display `(venv)` at the beginning).*

### Step 4: Verify Dependencies
If you need to install or verify dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### Step 5: Launch the Streamlit Web Application
Run the Streamlit application:

```powershell
streamlit run app.py
```

*Alternative direct execution if `streamlit` is not recognized globally:*
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

### Step 6: Access the Dashboard
Once launched, open your web browser and navigate to:
```text
http://localhost:8501
```

---

## 11. Application Dashboard Walkthrough

The web dashboard is organized into 6 focused, logical sections:

1. **Section 1 — Dataset Overview & Preview:**
   - Displays total records, column counts, missing value statistics, and column data types.
   - Includes an interactive scrollable table previewing the loaded dataset.
2. **Section 2 — Variable Selection & Feature Discretization:**
   - Independent dropdown selectors for Variable $X$ (Row) and Variable $Y$ (Column).
   - Real-time numerical discretization with live category frequency previews and configurable cutoffs.
   - Sparse category consolidation toggle.
3. **Section 3 — Contingency Table Creation:**
   - Clean cross-tabulation table with marginal row totals ($R_i$), column totals ($C_j$), and grand sample size ($N$).
4. **Section 4 — Hypothesis Testing & Chi-Square Results:**
   - 3 core KPI summary cards: $\chi^2$, $p$-value, and Degrees of Freedom ($df$).
   - High-contrast monochrome decision panel stating whether $H_0$ is rejected or not based on the significance threshold $\alpha$.
5. **Section 5 — Interactive Statistical Visualizations:**
   - **Grouped Bar Chart:** Categorical distribution chart using the original vibrant qualitative color palette (`Safe`), toggleable between absolute counts and within-group conditional percentages.
   - **Contingency Heatmap:** Multi-metric heatmap with original continuous color scales (`Blues` for observed counts, `Viridis` for expected frequencies, and `RdBu_r` for standardized residuals).
6. **Section 6 — Downloadable Statistical Analysis Report:**
   - One-click downloads for full text audit reports (`.txt`), observed contingency tables (`.csv`), and expected matrices (`.csv`).

---

## 12. Automated Unit Testing Suite

The repository includes an automated unit test suite in `test_statistical_engine.py` that verifies the statistical math without external dependencies.

### Running the Test Suite
Execute the tests using the virtual environment's Python binary:

```powershell
.\venv\Scripts\python.exe -m unittest test_statistical_engine.py -v
```

### Verified Test Suite Results (8 Tests Passed)

```text
test_cochran_criteria_validation (test_statistical_engine.TestStatisticalEngine.test_cochran_criteria_validation)
Test Cochran's (1954) validity rules on various expected matrices. ... ok
test_cramers_v_calculation (test_statistical_engine.TestStatisticalEngine.test_cramers_v_calculation)
Test Cramer's V calculation on known values. ... ok
test_cramers_v_interpretation (test_statistical_engine.TestStatisticalEngine.test_cramers_v_interpretation)
Test qualitative classification of Cramer's V. ... ok
test_demonstration_dataset_integrity (test_statistical_engine.TestStatisticalEngine.test_demonstration_dataset_integrity)
Verify demonstration_survey_data.csv existence, headers, and exact record count. ... ok
test_expected_frequencies_and_chi2 (test_statistical_engine.TestStatisticalEngine.test_expected_frequencies_and_chi2)
Test contingency expected frequencies and chi-square test statistic. ... ok
test_representative_calculation_attendance_vs_performance (test_statistical_engine.TestStatisticalEngine.test_representative_calculation_attendance_vs_performance)
Verify representative calculation: Attendance binned at 75 vs Performance on student_data.csv. ... ok
test_student_dataset_integrity (test_statistical_engine.TestStatisticalEngine.test_student_dataset_integrity)
Verify student_data.csv existence, headers, and exact record count. ... ok
test_text_report_and_csv_export_structure (test_statistical_engine.TestStatisticalEngine.test_text_report_and_csv_export_structure)
Verify that the generated text report and CSV export outputs conform to expected schema. ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.003s

OK
```

---

## 13. Limitations & Future Enhancements

### Limitations
1. **Information Loss via Discretization:** Converting continuous variables into discrete bins inevitably discards intra-bin variance.
2. **Sample Size Sensitivity:** Very small samples ($N < 30$) frequently violate Cochran's rule; very large samples ($N > 50,000$) can yield statistically significant $p$-values for negligible associations (highlighting the necessity of Cramér's V).
3. **Nominal Test Nature:** Standard Chi-Square treats categories as unordered nominal groups and does not leverage the ordinal hierarchy of grades (e.g., `'Poor'` $<$ `'Average'` $<$ `'Good'` $<$ `'Excellent'`).
4. **Association vs. Causation:** Statistical dependency confirms that distributions differ across groups, but cannot establish directional or causal mechanisms.

### Future Enhancements
- **Monte Carlo Exact Tests:** Implementing exact multinomial permutation tests for $r \times c$ tables larger than $2 \times 2$ when Cochran's rule is violated.
- **Ordinal Association Metrics:** Incorporating Goodman and Kruskal's Gamma ($\gamma$) and Spearman's Rank Correlation ($\rho$) for ordered academic levels.
- **Automated Feature Ranking:** Adding an automated screening module that tests and ranks all dataset features against a designated target class using Chi-Square scores.

---

## 14. Git Repository Workflow

### Repository URL
```text
https://github.com/Jetson17/Feature-Independence-Validator.git
```

### Common Commands

```powershell
# Check working tree status
git status

# Stage updated files
git add README.md app.py test_statistical_engine.py requirements.txt .gitignore

# Commit changes
git commit -m "Update README with verified dataset specs, walkthrough, and troubleshooting"

# Push to GitHub main branch
git push origin main
```

---

### Academic Notice
*This project was developed as a B.Tech Computer Science & Engineering Mathematics and Statistics micro-project. It illustrates the application of hypothesis testing, contingency table analysis, and statistical assumption auditing in data science.*
