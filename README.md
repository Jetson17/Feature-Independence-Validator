# Feature Independence Validation Using Chi-Square Test

An interactive statistical computing web application developed for the B.Tech Computer Science and Engineering (CSE) Mathematics and Statistics micro-project. Built with Python, Streamlit, Pandas, NumPy, SciPy, and Plotly, this tool validates the statistical independence of categorical features using Pearson's Chi-Square Test of Independence ($\chi^2$).

---

## 1. Project Title and Overview

**Title:** Feature Independence Validation Using Chi-Square Test  
**Degree:** B.Tech Computer Science & Engineering (Mathematics & Statistics Micro-Project)  

In statistical computing and data analysis, understanding how categorical features relate to one another is essential. When selecting features for predictive models or exploratory studies, researchers must verify whether two variables vary independently or exhibit statistically significant association.

This application provides an interactive web dashboard where users can load built-in academic datasets or upload custom CSV files, construct contingency cross-tabulation tables, discretize continuous numeric attributes, compute Pearson Chi-Square test statistics, evaluate Cochran's validity criteria, calculate Cramér's V effect size, interact with dynamic visualizations, and export comprehensive audit reports.

---

## 2. Problem Statement

In academic and industrial machine learning workflows:
1. **Feature Redundancy:** Introducing mutually dependent or redundant categorical features increases dimensionality, inflates variance, and complicates model interpretability.
2. **Arbitrary Feature Selection:** Variables are frequently fed into machine learning pipelines without mathematical verification of their statistical association.
3. **Small-Sample Approximation Breakdown:** Many practitioners compute asymptotic Chi-Square statistics without checking sample-size constraints (such as Cochran's rule), leading to unreliable conclusions when sparse expected frequencies inflate Type I error rates.

This project addresses these challenges by offering an automated, interactive statistical validation engine that couples hypothesis testing with diagnostic assumption audits and effect size measurements.

---

## 3. Objectives

1. **Hypothesis Testing:** Formulate and test statistical independence ($H_0$) against mutual association ($H_1$) for categorical feature pairs.
2. **Feature Discretization:** Provide dynamic binning strategies (custom cutoffs, quantile splits, and equal-width intervals) to convert continuous numerical attributes into discrete categories.
3. **Assumption Auditing:** Explicitly audit Cochran's (1954) criteria ($E_{ij} \ge 1.0$ and $\le 20\%$ cells $< 5.0$) and alert users to small-sample approximation risks.
4. **Effect Size Quantification:** Measure the practical strength of association using Cramér's V to decouple sample size effects from statistical significance.
5. **Interactive Visualization & Export:** Render reactive Plotly distribution charts and contingency heatmaps, and export publication-ready plaintext reports and CSV matrices.

---

## 4. Technologies and Libraries Used

The application is implemented using Python and a minimal set of direct libraries:

| Technology | Purpose & Role |
| :--- | :--- |
| **Python (3.10+)** | Core programming language for data structures, statistical logic, and testing routines. |
| **Streamlit** | High-performance reactive web framework for rendering the interactive dashboard and controls. |
| **Pandas** | Tabular data manipulation, CSV ingestion, missing-value filtering, and cross-tabulation (`pd.crosstab`). |
| **NumPy** | Numerical array processing, percentile calculations, and residual computations. |
| **SciPy (`scipy.stats`)** | Scientific computing engine providing `chi2_contingency`, `chi2.ppf` (critical value), and `fisher_exact`. |
| **Plotly** | Visualization engine generating interactive grouped bar charts and contingency heatmaps. |

> **Machine Learning Architecture Note:** This application does **not** train or deploy any machine learning predictive model (no scikit-learn classifiers, regression models, or neural networks). It is strictly a statistical hypothesis-testing and feature-independence diagnostic engine.

---

## 5. Features of the Application

- **Dual Preloaded Datasets & Custom Upload:** Toggle between an original 42-student academic record dataset, a 350-student demonstration survey dataset, or upload any user-supplied CSV file.
- **Configurable Significance Level ($\alpha$):** Choose between $\alpha = 0.01$, $\alpha = 0.05$ (standard), or $\alpha = 0.10$ with instant recalculation.
- **Variable Selection & Auto-Categorization:** Select any two distinct columns ($X \neq Y$). Numerical variables are automatically detected and converted into discrete categories.
- **Flexible Numerical Binning:** Choose between domain-based cutoffs, quantile divisions (median, tertiles, quartiles), or equal-width intervals.
- **Category Consolidation:** Optional toggle to combine sparse categories with counts $< 2$ into an `"Other"` group to alleviate sparse expected cells.
- **Contingency Tables:** View raw observed counts with marginal totals ($R_i, C_j, N$) or normalized conditional distributions (row %, column %, table %).
- **Hypothesis Testing Engine:** Computes $\chi^2$, degrees of freedom ($df$), $p$-value, critical threshold ($\chi^2_{\text{crit}}$), and displays plain-English decisions.
- **Cochran's Assumption Audit:** Evaluates whether cell expectations meet asymptotic criteria and recommends Fisher's Exact Test for $2 \times 2$ tables when violated.
- **Standardized Pearson Residuals:** Highlights individual cell combinations that deviate significantly ($|r_{ij}| > 2.0$) from theoretical independence.
- **Cramér's V Effect Size:** Quantifies association strength on a normalized scale ($0$ to $1$) with dimensional benchmark interpretations.
- **Interactive Visualizations:** Grouped bar charts (frequency vs. percentage) and multi-metric heatmaps (observed, expected, residuals).
- **Comprehensive Exports:** Download full analysis reports (`.txt`), observed contingency tables (`.csv`), and expected frequencies matrices (`.csv`).

---

## 6. Explanation of the Chi-Square Test of Independence

Pearson's Chi-Square Test of Independence is a non-parametric statistical technique used to determine whether there is a significant association between two nominal or ordinal categorical variables.

Under the assumption of independence, the probability that an observation falls into any given cell of a two-way cross-tabulation is simply the product of its marginal probabilities:
$$P(X = i \text{ and } Y = j) = P(X = i) \times P(Y = j)$$

The test evaluates whether the observed cell frequencies ($O_{ij}$) deviate from theoretical frequencies ($E_{ij}$) expected under random chance by more than would be anticipated from sampling variability alone.

---

## 7. Null Hypothesis and Alternative Hypothesis

For any two categorical attributes $X$ (with $r$ rows) and $Y$ (with $c$ columns):

- **Null Hypothesis ($H_0$):** Variables $X$ and $Y$ are **statistically independent**.
  - *Interpretation:* Knowledge of category membership in $X$ provides no information regarding category membership in $Y$.
- **Alternative Hypothesis ($H_1$):** Variables $X$ and $Y$ are **statistically associated / dependent**.
  - *Interpretation:* Category membership in $X$ is significantly related to category membership in $Y$.

---

## 8. Mathematical Formulations

### 1. Observed Margins and Sample Size
For an $r \times c$ contingency table with observed counts $O_{ij}$:
- Row totals: $R_i = \sum_{j=1}^c O_{ij}$
- Column totals: $C_j = \sum_{i=1}^r O_{ij}$
- Total sample size: $N = \sum_{i=1}^r \sum_{j=1}^c O_{ij}$

### 2. Expected Frequencies ($E_{ij}$)
Under $H_0$, the expected frequency for cell $(i, j)$ is:
$$E_{ij} = \frac{R_i \times C_j}{N}$$

### 3. Pearson Chi-Square Statistic ($\chi^2$)
Measures the sum of normalized squared deviations across all cells:
$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

### 4. Degrees of Freedom ($df$)
$$df = (r - 1) \times (c - 1)$$

### 5. P-Value & Decision Rule
Let $F_{\chi^2, df}$ denote the cumulative distribution function of a Chi-Square distribution with $df$ degrees of freedom:
$$p\text{-value} = 1 - F_{\chi^2, df}(\chi^2)$$

- **If $p\text{-value} < \alpha$ (or $\chi^2 > \chi^2_{\text{crit}}$):** **Reject $H_0$**. Conclude there is statistically significant evidence of association.
- **If $p\text{-value} \ge \alpha$ (or $\chi^2 \le \chi^2_{\text{crit}}$):** **Fail to Reject $H_0$**. Conclude the observed variation is consistent with random chance.

### 6. Standardized Pearson Residuals ($r_{ij}$)
$$r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}$$
Cells with $|r_{ij}| > 2.0$ indicate statistically significant deviation at approximately the $95\%$ confidence level.

### 7. Effect Size: Cramér's V
To evaluate practical magnitude independent of sample size $N$:
$$V = \sqrt{\frac{\chi^2}{N \times \min(r - 1, c - 1)}}$$

| Association Strength | $k = 1$ ($2 \times 2$) | $k = 2$ ($3 \times 3$) | $k \ge 3$ (Larger) |
| :--- | :--- | :--- | :--- |
| **Negligible** | $V < 0.10$ | $V < 0.07$ | $V < 0.06$ |
| **Weak** | $0.10 \le V < 0.30$ | $0.07 \le V < 0.21$ | $0.06 \le V < 0.17$ |
| **Moderate** | $0.30 \le V < 0.50$ | $0.21 \le V < 0.35$ | $0.17 \le V < 0.29$ |
| **Strong** | $V \ge 0.50$ | $V \ge 0.35$ | $V \ge 0.29$ |

*(where $k = \min(r - 1, c - 1)$)*

---

## 9. The Three Numerical Binning Methods

When numerical features are selected, the application discretizes continuous scales into discrete groups:

1. **Configurable Cutoffs (Custom Splits):**
   - User defines boundary cutoffs manually (or uses domain presets).
   - *Example:* Attendance can be partitioned into 2 categories (`Low (<75)` vs `High (≥75)`) or 3 categories (`Low`, `Medium`, `High`).
   - *Advantage:* Aligns with academic thresholds and institutional grading policies.
2. **Quantile Bins (Equal Frequency):**
   - Splits data using sample quantiles into 2 (median), 3 (tertiles), or 4 (quartiles) groups.
   - *Advantage:* Ensures approximately equal numbers of observations in each category, preventing sparse marginal distributions.
3. **Equal-Width Intervals:**
   - Divides the numerical range $[\min, \max]$ into equal-span intervals.
   - *Advantage:* Preserves the natural scale distribution across categories.

---

## 10. Significance Level ($\alpha$) & Cochran's Assumption Checks

### Significance Level ($\alpha$)
The significance level represents the acceptable probability of a Type I error (false positive). The dashboard provides 3 options:
- $\alpha = 0.01$ (Conservative: 1% false positive risk)
- $\alpha = 0.05$ (Standard: 5% false positive risk)
- $\alpha = 0.10$ (Exploratory: 10% false positive risk)

### Cochran's Assumption Audit (1954)
Because the Chi-Square statistic relies on a continuous large-sample approximation, the test is valid when:
1. **Minimum Cell Expectation:** $E_{ij} \ge 1.0$ across all cells.
2. **Sparse Cell Percentage:** No more than $20\%$ of cells have $E_{ij} < 5.0$.

When Cochran's criteria are violated:
- The dashboard issues a prominent warning callout.
- For $2 \times 2$ tables, **Fisher's Exact Test** and **Yates' continuity-corrected Chi-Square** are automatically calculated.
- For larger tables, category consolidation or larger datasets are recommended.

---

## 11. Project Folder Structure

```text
Feature-Independence-Validator/
├── app.py                      # Main Streamlit web application & UI
├── test_statistical_engine.py  # Automated unit test suite (8 test cases)
├── requirements.txt            # Production dependencies with minimum version bounds
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git exclusions for venvs, bytecode, and exports
└── dataset/
    ├── student_data.csv              # Student academic dataset (42 records)
    └── demonstration_survey_data.csv # Academic survey dataset (350 records)
```

---

## 12. Dataset Descriptions & Verified Record Counts

### 1. Student Academic Dataset (`dataset/student_data.csv`)
- **Verified Row Count:** Exactly **42 data rows** (excluding header).
- **Columns (5):**
  - `Attendance` (Numeric, float: 60.0 to 95.0)
  - `Study_Hours` (Numeric, float: 1.5 to 5.0)
  - `Previous_Marks` (Numeric, integer: 50 to 90)
  - `Assignment_Score` (Numeric, integer: 55 to 92)
  - `Performance` (Categorical: `'Excellent'`, `'Good'`, `'Average'`, `'Poor'`)
- **Purpose:** Demonstrates real student performance metrics and small-sample conditions ($N = 42$), triggering Cochran warnings to test statistical diagnostics.

### 2. Demonstration Academic Survey (`dataset/demonstration_survey_data.csv`)
- **Verified Row Count:** Exactly **350 data rows** (excluding header).
- **Columns (8):**
  - `Attendance_Level` (Categorical: `'High (>=85%)'`, `'Moderate (70-84%)'`, `'Low (<70%)'`)
  - `Study_Environment` (Categorical: `'Quiet/Private'`, `'Home/Moderate'`, `'Noisy'`, `'Library/Campus'`)
  - `Parent_Education` (Categorical: `'High School'`, `'Bachelor Degree'`, `'Master/PhD'`, `'Other'`)
  - `Extracurricular` (Categorical: `'Participating'`, `'Not Participating'`)
  - `Learning_Style` (Categorical: `'Visual'`, `'Auditory'`, `'Kinesthetic'`, `'Reading/Writing'`)
  - `Weekly_Study_Hours` (Numeric, float: 4.5 to 28.0)
  - `Exam_Score` (Numeric, integer: 35 to 95)
  - `Performance_Band` (Categorical: `'Distinction'`, `'First Class'`, `'Second Class'`, `'Needs Improvement'`)
- **Purpose:** Demonstrates large-sample testing ($N = 350$) where cell counts satisfy Cochran's validity criteria ($0$ cells $< 5.0$).

---

## 13. Installation Instructions for Windows PowerShell

Ensure Python 3.10+ is installed and accessible in PowerShell:

```powershell
python --version
```
*(If `python` is not recognized, use `py --version` or verify your PATH configuration).*

---

## 14. Virtual Environment Creation & Activation

Navigate into the project directory and create an isolated virtual environment:

```powershell
# Navigate into the project folder
cd <path-to-project-folder>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

> **Execution Policy Tip:** If PowerShell displays a script execution policy restriction, enable local scripts for the current process:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

---

## 15. Dependency Installation & Application Launch Commands

With your virtual environment activated:

```powershell
# Upgrade package installer
python -m pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

### Opening the Interface
Once launched, the terminal displays the local application URL:
```text
  Local URL: http://localhost:8501
```
Open **`http://localhost:8501`** in any modern web browser.

---

## 16. Instructions for Uploading a Custom CSV Dataset

To analyze custom data:
1. Open the **sidebar** in the web dashboard.
2. Under **“Select Dataset Source”**, select **“Upload Custom CSV File”**.
3. Click **“Browse files”** and upload your `.csv` file.
4. Requirements:
   - File must be valid comma-separated values (`.csv`).
   - Must contain at least 2 distinct columns.
   - Can include both categorical (text/string) and continuous numerical columns.
   - Missing cells (`NaN`, null) are automatically counted and excluded from the test matrix.

---

## 17. How to Interpret Results and Visualizations

- **Primary Metrics Cards:**
  - $\chi^2$: Higher values indicate larger deviations between observed and expected frequencies.
  - $p$-value: Compare against $\alpha$. If $p < \alpha$, the association is statistically significant.
  - Cramér's V: Indicates practical effect magnitude (from $0.00$ to $1.00$).
- **Decision Banner:**
  - Green banner: **Reject $H_0$** — Statistically significant association detected.
  - Grey banner: **Fail to Reject $H_0$** — Insufficient evidence of association.
- **Grouped Bar Chart:**
  - Toggle between raw counts and within-group conditional percentages to observe category distributions.
- **Contingency Heatmap:**
  - Switch between observed counts ($O_{ij}$), expected counts ($E_{ij}$), and standardized residuals ($r_{ij}$).
  - Cells in the residuals heatmap with $|r_{ij}| > 2.0$ indicate category pairs with significant positive or negative association.

---

## 18. How to Export the TXT Report and CSV Tables

In **Section 7** of the dashboard:
1. **Download Full Report (`.txt`):** Exports a formatted text report with timestamps, parameters, observed tables, expected tables, residuals, test statistics, and Cochran audit status.
2. **Download Contingency Table (`.csv`):** Exports the observed cross-tabulation table with marginal row and column totals.
3. **Download Expected Frequencies (`.csv`):** Exports the theoretical expectation matrix $E_{ij}$ rounded to 4 decimals.

---

## 19. How to Run Automated Unit Tests

A comprehensive unit test suite is included in `test_statistical_engine.py`. Run the suite with:

```powershell
python -m unittest test_statistical_engine.py -v
```

### Verified Test Suite Output (8 Tests Executed)
```text
test_cochran_criteria_validation ... ok
test_cramers_v_calculation ... ok
test_cramers_v_interpretation ... ok
test_demonstration_dataset_integrity ... ok
test_expected_frequencies_and_chi2 ... ok
test_representative_calculation_attendance_vs_performance ... ok
test_student_dataset_integrity ... ok
test_text_report_and_csv_export_structure ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.001s

OK
```

---

## 20. Limitations and Future Enhancements

### Limitations
1. **Discretization Information Loss:** Converting continuous numbers into bins causes loss of numerical precision.
2. **Sample Size Sensitivity:** Small samples ($N < 50$) often violate Cochran's rule; very large samples ($N > 100,000$) make trivial differences achieve $p < 0.001$.
3. **Nominal Independence:** Standard Chi-Square ignores ordinal hierarchy (e.g. `'Poor'` $<$ `'Good'`).
4. **Association $\neq$ Causation:** The test proves statistical dependency, not a causal relationship.
5. **No Model Training:** Does not fit parameters, predict target labels, or evaluate model accuracy.

### Future Enhancements
1. **Exact Multinomial Tests:** Add Monte Carlo simulation for tables larger than $2 \times 2$ when Cochran's rule fails.
2. **Ordinal Association Measures:** Add Spearman's rank correlation or Goodman and Kruskal's gamma for ordered categorical features.
3. **Automated Feature Selector:** Add automated batch ranking of all features against a target class label using Chi-Square scores.

---

## 21. GitHub Usage Instructions

### Cloning the Repository
```powershell
git clone https://github.com/Jetson17/Feature-Independence-Validator.git
cd Feature-Independence-Validator
```

### Checking Git Working Tree
```powershell
git status
```

### Staging and Committing Changes
```powershell
git add app.py test_statistical_engine.py requirements.txt .gitignore README.md
git commit -m "Complete project audit, add test suite, and update documentation"
```

### Pushing to GitHub
```powershell
git push origin main
```

---

## Academic Project Metadata

- **Course:** B.Tech Computer Science & Engineering | Mathematics & Statistics
- **Project Title:** Feature Independence Validation Using Chi-Square Test
- **Student Name:** `[Insert Student Name]`
- **Register / Roll Number:** `[Insert Roll / Register Number]`
- **Faculty Guide / Mentor:** `[Insert Faculty Guide Name]`
- **Department:** Department of Computer Science & Engineering
- **Institution:** `[Insert College / University Name]`
- **Academic Year:** `2025–2026`

---

## License

This repository is submitted for college course evaluation. It currently has no explicit open-source license. All rights are reserved by the author(s).
