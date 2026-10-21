"""
Feature Independence Validation Using Chi-Square Test in Machine Learning Models
Streamlit Micro-Project Web Application
"""

import io
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Chi-Square Feature Independence Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished academic dashboard aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 900px;
        margin-bottom: 0.8rem;
    }
    .hero-badge {
        display: inline-block;
        background-color: #3b82f6;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.5rem;
    }
    .hero-badge-alt {
        display: inline-block;
        background-color: #10b981;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Card Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    /* Status Badges */
    .decision-box-reject {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1.5px solid #10b981;
        border-radius: 10px;
        padding: 1.4rem;
        margin: 1rem 0;
    }
    .decision-box-fail {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1.5px solid #3b82f6;
        border-radius: 10px;
        padding: 1.4rem;
        margin: 1rem 0;
    }
    .decision-box-warn {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 10px;
        padding: 1.4rem;
        margin: 1rem 0;
    }
    
    /* Clean table captions and section headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Streamlit tabs and container improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER FUNCTIONS & STATISTICAL CALCULATIONS
# ==============================================================================

@st.cache_data
def load_default_student_data():
    """Load the original student academic dataset."""
    try:
        return pd.read_csv("dataset/student_data.csv")
    except Exception as e:
        st.error(f"Error loading student_data.csv: {e}")
        return None

@st.cache_data
def load_demonstration_data():
    """Load the synthetic demonstration survey dataset."""
    try:
        return pd.read_csv("dataset/demonstration_survey_data.csv")
    except Exception as e:
        st.error(f"Error loading demonstration_survey_data.csv: {e}")
        return None

def compute_cramers_v(chi2_stat, n, r, c):
    """
    Compute Cramer's V effect size measure.
    V = sqrt(chi2 / (n * min(r - 1, c - 1)))
    """
    if n <= 0:
        return 0.0
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    v = np.sqrt(max(0.0, chi2_stat / (n * k)))
    return float(v)

def interpret_cramers_v(v, r, c):
    """
    Determine the qualitative strength of Cramer's V based on degrees of freedom.
    Using Cohen (1988) and Rea & Parker benchmark tables.
    """
    k = min(r - 1, c - 1)
    if k <= 1:
        if v < 0.10:
            return "Negligible", "The association magnitude is trivial or practically negligible."
        elif v < 0.30:
            return "Small / Weak", "There is a weak association between the two features."
        elif v < 0.50:
            return "Moderate", "There is a moderate association between the two features."
        else:
            return "Strong / Large", "There is a strong association between the two features."
    elif k == 2:
        if v < 0.07:
            return "Negligible", "The association magnitude is trivial or practically negligible."
        elif v < 0.21:
            return "Small / Weak", "There is a weak association between the two features."
        elif v < 0.35:
            return "Moderate", "There is a moderate association between the two features."
        else:
            return "Strong / Large", "There is a strong association between the two features."
    else:
        if v < 0.06:
            return "Negligible", "The association magnitude is trivial or practically negligible."
        elif v < 0.17:
            return "Small / Weak", "There is a weak association between the two features."
        elif v < 0.29:
            return "Moderate", "There is a moderate association between the two features."
        else:
            return "Strong / Large", "There is a strong association between the two features."

def generate_text_report(
    project_title,
    dataset_name,
    var_x,
    var_y,
    binning_info_x,
    binning_info_y,
    total_records,
    analyzed_records,
    excluded_records,
    observed_df,
    expected_df,
    residuals_df,
    chi2_stat,
    dof,
    p_value,
    alpha,
    critical_val,
    decision,
    cramers_v,
    effect_label,
    cochran_valid,
    cells_below_5_pct,
    min_expected
):
    """Generate a clean, structured plaintext report for export."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 78)
    report.append(f"{project_title.upper()}")
    report.append("STATISTICAL ANALYSIS & FEATURE INDEPENDENCE VALIDATION REPORT")
    report.append("=" * 78)
    report.append(f"Generated On          : {now}")
    report.append(f"Dataset Selected      : {dataset_name}")
    report.append(f"Feature X (Row)       : {var_x} {binning_info_x}")
    report.append(f"Feature Y (Column)    : {var_y} {binning_info_y}")
    report.append(f"Total Dataset Records : {total_records}")
    report.append(f"Analyzed Records (N)  : {analyzed_records}")
    report.append(f"Excluded (Missing)    : {excluded_records}")
    report.append(f"Significance Level (a): {alpha}")
    report.append("-" * 78)
    
    report.append("\n1. OBSERVED CONTINGENCY TABLE (pd.crosstab)")
    report.append("-" * 78)
    report.append(observed_df.to_string())
    
    report.append("\n\n2. EXPECTED FREQUENCIES TABLE (E_ij = R_i * C_j / N)")
    report.append("-" * 78)
    report.append(expected_df.round(3).to_string())
    
    report.append("\n\n3. STANDARDIZED RESIDUALS ((O - E) / sqrt(E))")
    report.append("-" * 78)
    report.append(residuals_df.round(3).to_string())
    
    report.append("\n\n4. CHI-SQUARE TEST OF INDEPENDENCE RESULTS")
    report.append("-" * 78)
    report.append(f"Chi-Square Statistic (X^2) : {chi2_stat:.5f}")
    report.append(f"Degrees of Freedom (df)     : {dof}")
    report.append(f"P-Value                     : {p_value:.6e} ({p_value:.5f})")
    report.append(f"Critical Value (X^2_crit)   : {critical_val:.5f} (at a = {alpha})")
    report.append(f"Effect Size (Cramer's V)    : {cramers_v:.4f} [{effect_label}]")
    
    report.append("\n5. HYPOTHESES & STATISTICAL DECISION")
    report.append("-" * 78)
    report.append("H0 (Null Hypothesis)       : Feature X and Feature Y are independent.")
    report.append("H1 (Alternative Hypothesis): Feature X and Feature Y are associated.")
    report.append(f"Decision Rule              : If p-value < {alpha}, Reject H0; Else Fail to Reject H0.")
    report.append(f"Final Decision             : {decision.upper()}")
    
    if "REJECT" in decision.upper() and "FAIL" not in decision.upper():
        report.append(
            f"Interpretation             : At the {alpha} significance level, the p-value ({p_value:.5f}) is less\n"
            f"                             than {alpha}. There is statistically significant evidence to reject H0.\n"
            f"                             A significant association exists between {var_x} and {var_y}."
        )
    else:
        report.append(
            f"Interpretation             : At the {alpha} significance level, the p-value ({p_value:.5f}) is greater\n"
            f"                             than or equal to {alpha}. We fail to reject H0. The data does not provide\n"
            f"                             sufficient statistical evidence of association."
        )
    report.append("\n* CAUTION: Statistical association does not imply a causal relationship.")
    report.append("           Failing to reject H0 does not prove absolute independence.")
    
    report.append("\n6. STATISTICAL ASSUMPTION AUDIT (Cochran's Rule)")
    report.append("-" * 78)
    report.append(f"Minimum Expected Frequency : {min_expected:.3f}")
    report.append(f"Cells with Expected < 5    : {cells_below_5_pct:.1f}%")
    if cochran_valid:
        report.append("Assumption Audit Status    : PASSED. Asymptotic Chi-Square approximation is reliable.")
    else:
        report.append("Assumption Audit Status    : WARNING. Small expected frequencies detected.")
        report.append("                             More than 20% of cells have expected frequency < 5, or min < 1.")
        report.append("                             Consider combining sparse categories or expanding sample size.")
    
    report.append("\n7. MACHINE LEARNING FEATURE VALIDATION CONTEXT")
    report.append("-" * 78)
    if "REJECT" in decision.upper() and "FAIL" not in decision.upper():
        report.append(
            "Feature Engineering Note   : The significant association suggests these features share predictive\n"
            "                             information. If one is the target, this feature is informative.\n"
            "                             If both are input features, evaluate whether retaining both introduces\n"
            "                             redundancy or multicollinearity in classification models."
        )
    else:
        report.append(
            "Feature Engineering Note   : The lack of detectable association suggests these categorical features\n"
            "                             behave independently in this sample, carrying non-overlapping signals."
        )
    report.append("=" * 78)
    report.append("END OF REPORT")
    report.append("=" * 78)
    
    return "\n".join(report)


# ==============================================================================
# 3. APPLICATION HEADER & HERO SECTION
# ==============================================================================

st.markdown("""
<div class="hero-container">
    <div>
        <span class="hero-badge">Statistical Hypothesis Testing</span>
        <span class="hero-badge-alt">Machine Learning Feature Validation</span>
    </div>
    <div class="hero-title">Feature Independence Validation Using Chi-Square Test</div>
    <div class="hero-subtitle">
        An interactive micro-project dashboard for evaluating statistical independence, category cross-tabulation, 
        and association strength between categorical features in machine learning datasets.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.title("🛠️ Configuration")

dataset_choice = st.sidebar.selectbox(
    "Select Dataset Source",
    options=[
        "Student Academic Dataset (Default - 42 records)",
        "Demonstration Academic Survey (Synthetic - 350 records)",
        "Upload Custom CSV File"
    ],
    index=0,
    help="Choose between the original student dataset, a larger demonstration survey, or your own CSV."
)

df_raw = None
dataset_label = ""

if dataset_choice == "Student Academic Dataset (Default - 42 records)":
    df_raw = load_default_student_data()
    dataset_label = "Student Academic Dataset (dataset/student_data.csv)"
    st.sidebar.info("💡 **Original student dataset loaded**: 42 student records with academic performance metrics.")

elif dataset_choice == "Demonstration Academic Survey (Synthetic - 350 records)":
    df_raw = load_demonstration_data()
    dataset_label = "Demonstration Academic Survey (dataset/demonstration_survey_data.csv)"
    st.sidebar.success("📊 **Demonstration dataset loaded**: 350 records with multiple categorical variables ideal for large-sample Chi-Square validation.")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload any comma-separated values (.csv) file containing categorical or numerical columns."
    )
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            dataset_label = f"Uploaded CSV: {uploaded_file.name}"
            st.sidebar.success(f"✅ Loaded `{uploaded_file.name}` ({len(df_raw)} records)")
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded CSV: {e}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file in the sidebar to proceed, or select one of the provided datasets.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.error("Dataset could not be loaded or is empty. Please verify the dataset path or file format.")
    st.stop()

st.sidebar.divider()

# Significance level alpha
st.sidebar.subheader("🎯 Significance Level (α)")
alpha = st.sidebar.radio(
    "Select α (Threshold for rejecting H₀):",
    options=[0.01, 0.05, 0.10],
    index=1,
    format_func=lambda x: f"α = {x} ({int(x*100)}% Type I Error Rate)",
    help="The probability of rejecting the null hypothesis when it is actually true (Type I error rate). α = 0.05 is standard."
)

st.sidebar.divider()

# Expandable Guide for Viva/Presentation
with st.sidebar.expander("📖 Micro-Project Guide & Theory", expanded=False):
    st.markdown(r"""
    **Chi-Square (χ²) Test of Independence:**
    - Non-parametric test to determine whether two categorical variables are statistically associated.
    
    **Hypotheses:**
    - **$H_0$ (Null):** The variables are independent (no association).
    - **$H_1$ (Alternative):** The variables are associated.
    
    **Decision Rule:**
    - If **p-value < α**, reject $H_0$.
    - Otherwise, fail to reject $H_0$.
    
    **Cochran's Assumption (1954):**
    - Minimum expected cell count must be $\ge 1$.
    - At least 80% of expected cell counts must be $\ge 5$.
    
    **Cramer's V:**
    - Effect size measure quantifying association magnitude from 0 (no association) to 1 (complete association).
    """)


# ==============================================================================
# 5. DATASET OVERVIEW SECTION
# ==============================================================================

st.markdown('<div class="section-title">📂 Dataset Overview</div>', unsafe_allow_html=True)

total_records = len(df_raw)
total_columns = len(df_raw.columns)
missing_cells = int(df_raw.isna().sum().sum())
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df_raw.columns if c not in num_cols]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records (N)", total_records)
with col2:
    st.metric("Total Features (P)", total_columns)
with col3:
    st.metric("Missing Values", missing_cells)
with col4:
    st.metric("Categorical Features", len(cat_cols))
with col5:
    st.metric("Numerical Features", len(num_cols))

overview_tabs = st.tabs(["📋 Dataset Preview", "🔍 Column Metadata & Data Types", "📥 Export Dataset"])

with overview_tabs[0]:
    st.dataframe(df_raw.head(15), width="stretch")
    st.caption(f"Showing first 15 of {total_records} rows from `{dataset_label}`.")

with overview_tabs[1]:
    meta_records = []
    for col in df_raw.columns:
        dt = str(df_raw[col].dtype)
        n_unique = df_raw[col].nunique()
        n_null = int(df_raw[col].isna().sum())
        null_pct = (n_null / total_records) * 100 if total_records > 0 else 0
        sample_vals = ", ".join(map(str, df_raw[col].dropna().unique()[:4]))
        col_type = "Numerical" if col in num_cols else "Categorical"
        meta_records.append({
            "Column Name": col,
            "Detected Nature": col_type,
            "Data Type": dt,
            "Non-Null Count": total_records - n_null,
            "Null Count": n_null,
            "Null %": f"{null_pct:.1f}%",
            "Unique Values": n_unique,
            "Sample Values": sample_vals
        })
    meta_df = pd.DataFrame(meta_records)
    st.dataframe(meta_df, width="stretch", hide_index=True)

with overview_tabs[2]:
    csv_data = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Dataset (CSV)",
        data=csv_data,
        file_name=f"dataset_export_{datetime.date.today()}.csv",
        mime="text/csv",
        help="Download the active dataset as a CSV file."
    )


# ==============================================================================
# 6. FEATURE SELECTION & NUMERICAL CATEGORIZATION
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">⚙️ Feature Selection & Categorization</div>', unsafe_allow_html=True)
st.write(
    "Select two distinct features from the dataset to test for statistical independence. "
    "If you select a numerical feature, use the interactive binning controls to convert it into meaningful categories."
)

col_sel1, col_sel2 = st.columns(2)

all_columns = list(df_raw.columns)
default_idx1 = 0
default_idx2 = 1 if len(all_columns) > 1 else 0

# Pick sensible defaults if available
if "Performance" in all_columns and "Attendance" in all_columns:
    default_idx1 = all_columns.index("Attendance")
    default_idx2 = all_columns.index("Performance")
elif "Attendance_Level" in all_columns and "Performance_Band" in all_columns:
    default_idx1 = all_columns.index("Attendance_Level")
    default_idx2 = all_columns.index("Performance_Band")

with col_sel1:
    feature_x = st.selectbox(
        "Select Feature 1 (Variable X / Row Variable):",
        options=all_columns,
        index=default_idx1,
        key="feature_x_sel"
    )

with col_sel2:
    feature_y = st.selectbox(
        "Select Feature 2 (Variable Y / Column Variable):",
        options=all_columns,
        index=default_idx2,
        key="feature_y_sel"
    )

if feature_x == feature_y:
    st.error("⚠️ **Invalid Selection**: Feature 1 and Feature 2 must be different columns. Please pick two distinct variables.")
    st.stop()


# Helper function to process/bin a feature
def process_feature(df, feature_name, col_key_suffix):
    """
    Examines a column. If categorical, returns series and metadata.
    If numerical, provides configurable binning controls and returns binned series.
    """
    col_series = df[feature_name].copy()
    is_numerical = pd.api.types.is_numeric_dtype(col_series)
    
    if not is_numerical:
        # Categorical variable
        st.success(f"✅ **`{feature_name}`** is categorical with **{col_series.nunique()}** distinct observed categories.")
        return col_series.astype(str), f"[Categorical, {col_series.nunique()} categories]"
    
    # Numerical variable -> Needs explicit categorization
    st.info(f"ℹ️ **`{feature_name}`** is numerical (Range: {col_series.min():.1f} to {col_series.max():.1f}). Converting into categories:")
    
    bin_method = st.radio(
        f"Binning Method for `{feature_name}`:",
        options=["Configurable Cutoffs (Custom)", "Quantile Bins (Equal Frequency)", "Equal Width Intervals"],
        key=f"bin_method_{col_key_suffix}",
        horizontal=True
    )
    
    c_min = float(col_series.min())
    c_max = float(col_series.max())
    
    if bin_method == "Configurable Cutoffs (Custom)":
        num_bins = st.selectbox(
            f"Number of categories for `{feature_name}`:",
            options=[2, 3, 4],
            index=1,
            key=f"num_bins_{col_key_suffix}"
        )
        
        # Provide domain-appropriate presets for known columns
        presets = {}
        if feature_name == "Attendance" and c_max <= 100:
            if num_bins == 3:
                presets = {"cut1": 70.0, "cut2": 85.0, "labels": "Low (<70%), Medium (70-84%), High (>=85%)"}
            elif num_bins == 2:
                presets = {"cut1": 75.0, "labels": "Low (<75%), High (>=75%)"}
        elif feature_name == "Study_Hours":
            if num_bins == 3:
                presets = {"cut1": 2.5, "cut2": 4.0, "labels": "Low (<2.5h), Medium (2.5-4h), High (>=4h)"}
        elif "Marks" in feature_name or "Score" in feature_name:
            if num_bins == 3:
                presets = {"cut1": 65.0, "cut2": 80.0, "labels": "Poor/Average (<65), Good (65-79), Excellent (>=80)"}
        
        if num_bins == 2:
            default_cut = presets.get("cut1", round(float(col_series.median()), 1))
            cut1 = st.slider(
                f"Boundary Cutoff for `{feature_name}`:",
                min_value=float(c_min),
                max_value=float(c_max),
                value=float(np.clip(default_cut, c_min, c_max)),
                key=f"cut1_{col_key_suffix}"
            )
            bins = [-np.inf, cut1, np.inf]
            labels = [f"Low (<{cut1:.1f})", f"High (≥{cut1:.1f})"]
            
        elif num_bins == 3:
            p33 = float(np.percentile(col_series.dropna(), 33))
            p66 = float(np.percentile(col_series.dropna(), 66))
            default_cut1 = presets.get("cut1", round(p33, 1))
            default_cut2 = presets.get("cut2", round(p66, 1))
            if default_cut1 >= default_cut2:
                default_cut2 = default_cut1 + 1.0
            
            c_cut1, c_cut2 = st.columns(2)
            with c_cut1:
                cut1 = st.number_input(
                    f"Cutoff 1 (Low/Medium Boundary):",
                    value=float(default_cut1),
                    key=f"cut1_3_{col_key_suffix}"
                )
            with c_cut2:
                cut2 = st.number_input(
                    f"Cutoff 2 (Medium/High Boundary):",
                    value=float(default_cut2),
                    key=f"cut2_3_{col_key_suffix}"
                )
            if cut1 >= cut2:
                st.warning("⚠️ Cutoff 1 should be strictly less than Cutoff 2. Adjusting Cutoff 2 automatically.")
                cut2 = cut1 + 1.0
            bins = [-np.inf, cut1, cut2, np.inf]
            labels = [f"Low (<{cut1:.1f})", f"Medium ({cut1:.1f}-{cut2:.1f})", f"High (≥{cut2:.1f})"]
            
        else: # 4 bins
            p25 = float(np.percentile(col_series.dropna(), 25))
            p50 = float(np.percentile(col_series.dropna(), 50))
            p75 = float(np.percentile(col_series.dropna(), 75))
            c_cut1, c_cut2, c_cut3 = st.columns(3)
            with c_cut1:
                cut1 = st.number_input(f"Cutoff 1:", value=round(p25, 1), key=f"cut1_4_{col_key_suffix}")
            with c_cut2:
                cut2 = st.number_input(f"Cutoff 2:", value=round(p50, 1), key=f"cut2_4_{col_key_suffix}")
            with c_cut3:
                cut3 = st.number_input(f"Cutoff 3:", value=round(p75, 1), key=f"cut3_4_{col_key_suffix}")
            bins = [-np.inf, cut1, cut2, cut3, np.inf]
            labels = [f"Tier 1 (<{cut1:.1f})", f"Tier 2 ({cut1:.1f}-{cut2:.1f})", f"Tier 3 ({cut2:.1f}-{cut3:.1f})", f"Tier 4 (≥{cut3:.1f})"]
            
        binned = pd.cut(col_series, bins=bins, labels=labels, right=False)
        bin_desc = f"[Binned into {num_bins} categories: {', '.join(labels)}]"
        
    elif bin_method == "Quantile Bins (Equal Frequency)":
        num_q = st.selectbox(
            f"Quantile Divisions for `{feature_name}`:",
            options=[2, 3, 4],
            index=1,
            format_func=lambda x: f"{x} Quantiles ({'Median split' if x==2 else 'Tertiles' if x==3 else 'Quartiles'})",
            key=f"num_q_{col_key_suffix}"
        )
        if num_q == 2:
            q_labels = ["Low (Q1)", "High (Q2)"]
        elif num_q == 3:
            q_labels = ["Low (T1)", "Medium (T2)", "High (T3)"]
        else:
            q_labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
        try:
            binned = pd.qcut(col_series, q=num_q, labels=q_labels, duplicates='drop')
            bin_desc = f"[Quantile binned: {num_q} bins]"
        except Exception as e:
            st.warning(f"Could not calculate quantiles directly ({e}). Falling back to equal-width bins.")
            binned = pd.cut(col_series, bins=num_q, labels=q_labels, right=False)
            bin_desc = f"[Equal width binned: {num_q} bins]"
            
    else: # Equal Width Intervals
        num_w = st.selectbox(
            f"Number of Equal-Width Intervals for `{feature_name}`:",
            options=[2, 3, 4],
            index=1,
            key=f"num_w_{col_key_suffix}"
        )
        labels = [f"Interval {i+1}" for i in range(num_w)]
        binned = pd.cut(col_series, bins=num_w, labels=labels, right=False)
        bin_desc = f"[Equal width binned: {num_w} intervals]"
        
    # Show category distribution
    counts = binned.value_counts().sort_index()
    dist_str = " | ".join([f"**{cat}**: {count}" for cat, count in counts.items()])
    st.caption(f"Categorization counts: {dist_str}")
    
    return binned.astype(str), bin_desc


st.write("---")
col_proc1, col_proc2 = st.columns(2)

with col_proc1:
    st.markdown(f"#### Variable X: `{feature_x}`")
    series_x, info_x = process_feature(df_raw, feature_x, "x")

with col_proc2:
    st.markdown(f"#### Variable Y: `{feature_y}`")
    series_y, info_y = process_feature(df_raw, feature_y, "y")


# ==============================================================================
# 7. DATA CLEANING & CONTINGENCY TABLE CREATION
# ==============================================================================

# Align series and handle missing values explicitly
analysis_df = pd.DataFrame({
    feature_x: series_x,
    feature_y: series_y
})

initial_count = len(analysis_df)
# Drop NaN or literal 'nan' strings
valid_mask = (
    analysis_df[feature_x].notna() &
    analysis_df[feature_y].notna() &
    (analysis_df[feature_x] != "nan") &
    (analysis_df[feature_y] != "nan")
)
clean_df = analysis_df[valid_mask]
analyzed_count = len(clean_df)
excluded_count = initial_count - analyzed_count

if excluded_count > 0:
    st.warning(f"⚠️ **Missing Value Handling**: {excluded_count} record(s) excluded due to missing/invalid values. Analyzing **{analyzed_count}** valid observations.")
else:
    st.caption(f"✅ **Data Completeness**: All {analyzed_count} records have complete data for `{feature_x}` and `{feature_y}`.")

# Validate minimum unique categories
unique_x = clean_df[feature_x].nunique()
unique_y = clean_df[feature_y].nunique()

if unique_x < 2 or unique_y < 2:
    st.error(
        f"❌ **Insufficient Categories for Testing**: "
        f"`{feature_x}` has {unique_x} category, and `{feature_y}` has {unique_y} category. "
        f"Chi-Square Test requires at least 2 observed categories per variable. "
        f"Please adjust your binning cutoffs or choose different variables."
    )
    st.stop()

# Observed contingency table (cross-tabulation)
observed_crosstab = pd.crosstab(
    clean_df[feature_x],
    clean_df[feature_y],
    margins=False
)

observed_with_margins = pd.crosstab(
    clean_df[feature_x],
    clean_df[feature_y],
    margins=True,
    margins_name="Total"
)

st.divider()
st.markdown('<div class="section-title">📊 Contingency Table (Cross-Tabulation)</div>', unsafe_allow_html=True)

tab_obs1, tab_obs2, tab_obs3 = st.tabs(["🔢 Observed Counts", "📈 Proportions & Percentages", "📥 Export Table"])

with tab_obs1:
    st.write(f"Observed joint frequency distribution $O_{{ij}}$ for **{feature_x}** (rows) vs **{feature_y}** (columns):")
    st.dataframe(observed_with_margins, width="stretch")
    st.caption(f"Table dimensions: {observed_crosstab.shape[0]} rows × {observed_crosstab.shape[1]} columns (excluding marginal totals). Sample size N = {analyzed_count}.")

with tab_obs2:
    prop_type = st.radio(
        "Display Proportions As:",
        options=["Row Percentages (%)", "Column Percentages (%)", "Total Table Percentages (%)"],
        horizontal=True
    )
    if prop_type == "Row Percentages (%)":
        row_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='index') * 100
        st.dataframe(row_props.round(2).astype(str) + "%", width="stretch")
        st.caption("Each row sums to 100.0%. Shows the conditional distribution of Feature Y given Feature X.")
    elif prop_type == "Column Percentages (%)":
        col_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='columns') * 100
        st.dataframe(col_props.round(2).astype(str) + "%", width="stretch")
        st.caption("Each column sums to 100.0%. Shows the conditional distribution of Feature X given Feature Y.")
    else:
        tot_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='all') * 100
        st.dataframe(tot_props.round(2).astype(str) + "%", width="stretch")
        st.caption("All cells sum to 100.0%. Shows the joint relative frequency distribution.")

with tab_obs3:
    csv_crosstab = observed_with_margins.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Contingency Table (CSV)",
        data=csv_crosstab,
        file_name=f"contingency_table_{feature_x}_vs_{feature_y}.csv",
        mime="text/csv",
        help="Download the observed contingency table as a CSV file."
    )


# ==============================================================================
# 8. CHI-SQUARE COMPUTATION & STATISTICAL AUDIT
# ==============================================================================

# SciPy Chi-Square test
chi2_stat, p_value, dof, expected_matrix = stats.chi2_contingency(observed_crosstab, correction=False)

# Convert expected frequencies to DataFrame with matching index and columns
expected_df = pd.DataFrame(
    expected_matrix,
    index=observed_crosstab.index,
    columns=observed_crosstab.columns
)

# Standardized Pearson residuals: (O - E) / sqrt(E)
residuals_df = (observed_crosstab - expected_df) / np.sqrt(expected_df)

# Critical value at selected alpha
critical_val = stats.chi2.ppf(1 - alpha, dof)

# Effect size: Cramer's V
r_dim, c_dim = observed_crosstab.shape
cramers_v = compute_cramers_v(chi2_stat, analyzed_count, r_dim, c_dim)
effect_label, effect_desc = interpret_cramers_v(cramers_v, r_dim, c_dim)

# Cochran's (1954) criteria validation:
# 1. No expected frequency < 1
# 2. At most 20% of expected frequencies < 5
min_expected = float(expected_matrix.min())
cells_below_1 = int((expected_matrix < 1.0).sum())
cells_below_5 = int((expected_matrix < 5.0).sum())
total_cells = int(expected_matrix.size)
cells_below_5_pct = (cells_below_5 / total_cells) * 100.0

cochran_valid = (cells_below_1 == 0) and (cells_below_5_pct <= 20.0)

# Statistical Decision
reject_null = p_value < alpha
if reject_null:
    decision_text = "Reject Null Hypothesis (H₀)"
    decision_status = "Statistically Significant Association"
else:
    decision_text = "Fail to Reject Null Hypothesis (H₀)"
    decision_status = "No Statistically Significant Association"


# ==============================================================================
# 9. RESULTS & STATISTICAL DECISION DASHBOARD
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">🧪 Chi-Square Test of Independence Results</div>', unsafe_allow_html=True)

# Metrics Cards
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Chi-Square (χ²)", f"{chi2_stat:.4f}")
with m_col2:
    p_format = f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}"
    st.metric("P-Value", p_format)
with m_col3:
    st.metric("Degrees of Freedom (df)", dof)
with m_col4:
    st.metric("Critical Value (χ²_crit)", f"{critical_val:.4f}")
with m_col5:
    st.metric("Cramer's V (Effect)", f"{cramers_v:.4f}")

# Hypothesis Decision Container
if reject_null:
    st.markdown(f"""
    <div class="decision-box-reject">
        <h3 style="color: #065f46; margin-top:0;">✅ Statistical Decision: {decision_text}</h3>
        <p style="color: #047857; font-size: 1.05rem; margin-bottom: 0.5rem;">
            <strong>Result:</strong> Statistically significant association detected between 
            <code>{feature_x}</code> and <code>{feature_y}</code> at the α = {alpha} level.
        </p>
        <p style="color: #064e3b; margin-bottom: 0.2rem;">
            <strong>Interpretation:</strong> Because the p-value (<strong>{p_format}</strong>) is strictly less than 
            the chosen significance level (<strong>α = {alpha}</strong>), and the calculated test statistic 
            (<strong>χ² = {chi2_stat:.4f}</strong>) exceeds the critical value (<strong>χ²_crit = {critical_val:.4f}</strong>), 
            we reject the null hypothesis of independence. The distribution of categories in <code>{feature_x}</code> 
            is statistically dependent on <code>{feature_y}</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="decision-box-fail">
        <h3 style="color: #1e40af; margin-top:0;">⚖️ Statistical Decision: {decision_text}</h3>
        <p style="color: #1d4ed8; font-size: 1.05rem; margin-bottom: 0.5rem;">
            <strong>Result:</strong> Insufficient statistical evidence to reject H₀ at the α = {alpha} level.
        </p>
        <p style="color: #1e3a8a; margin-bottom: 0.2rem;">
            <strong>Interpretation:</strong> Because the p-value (<strong>{p_format}</strong>) is greater than or equal to 
            the significance level (<strong>α = {alpha}</strong>), we fail to reject the null hypothesis. 
            The sample does not exhibit a statistically significant association between <code>{feature_x}</code> and <code>{feature_y}</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Academic & Statistical Rigor Alerts
col_note1, col_note2 = st.columns(2)
with col_note1:
    st.warning(
        "⚠️ **Association Does Not Imply Causation**: "
        "A statistically significant Chi-Square test confirms that category proportions differ beyond random chance, "
        "but it does not establish that one feature causes changes in the other."
    )
with col_note2:
    st.info(
        "📘 **Absence of Evidence is Not Evidence of Absence**: "
        "Failing to reject H₀ does not mathematically prove complete independence; "
        "it only indicates that any potential association is too small to be detected at this sample size and significance level."
    )


# ==============================================================================
# 10. STATISTICAL ASSUMPTION AUDIT (Cochran's Criterion)
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">🛡️ Statistical Assumption Audit (Cochran\'s Criterion)</div>', unsafe_allow_html=True)

if cochran_valid:
    st.success(
        f"✅ **Assumptions Fully Satisfied**: Minimum expected cell frequency is **{min_expected:.2f}** (≥ 1.0), "
        f"and {cells_below_5} of {total_cells} cells ({cells_below_5_pct:.1f}%) have expected count < 5.0 (≤ 20.0%). "
        f"The asymptotic Chi-Square distribution approximation is reliable."
    )
else:
    st.markdown(f"""
    <div class="decision-box-warn">
        <h4 style="color: #92400e; margin-top:0;">⚠️ Small Expected Frequencies Detected (Cochran's Rule Warning)</h4>
        <p style="color: #b45309; margin-bottom: 0.4rem;">
            The standard asymptotic Chi-Square test relies on large-sample approximations. 
            Cochran (1954) established that:
        </p>
        <ul style="color: #78350f; margin-bottom: 0.5rem;">
            <li>No expected cell frequency should be strictly less than 1.0 (Current minimum: <strong>{min_expected:.2f}</strong>).</li>
            <li>No more than 20% of expected cell frequencies should be less than 5.0 (Current: <strong>{cells_below_5_pct:.1f}%</strong> — {cells_below_5} of {total_cells} cells).</li>
        </ul>
        <p style="color: #92400e; margin-bottom: 0;">
            <strong>Impact:</strong> The reported p-value should be interpreted cautiously and not treated as definitive.
            <strong>Recommended Solutions:</strong>
            (1) Consolidate sparse categories into fewer bins (e.g. 2 bins instead of 3 or 4);
            (2) Increase the sample size by testing the 350-record Demonstration dataset;
            (3) In a viva/presentation, explain that small datasets naturally trigger Cochran warnings, demonstrating statistical rigor!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Expected Frequencies Table & Residuals Tabs
exp_tabs = st.tabs(["🧮 Expected Frequencies Matrix (E_ij)", "🎯 Standardized Residuals", "📏 Effect Size (Cramer's V)"])

with exp_tabs[0]:
    st.write("Expected frequencies calculated under the theoretical assumption of complete independence:")
    st.markdown("$$E_{ij} = \\frac{R_i \\times C_j}{N}$$")
    st.dataframe(expected_df.round(2), width="stretch")
    st.caption("Each cell represents the count expected if no association existed between the two variables.")

with exp_tabs[1]:
    st.write("Standardized Pearson residuals identify which specific category combinations deviate most from expectation:")
    st.markdown("$$r_{ij} = \\frac{O_{ij} - E_{ij}}{\\sqrt{E_{ij}}}$$")
    st.dataframe(residuals_df.round(2), width="stretch")
    st.caption(
        "Interpretation: Residuals with absolute value > 2.0 (or > -2.0) indicate cells that contribute "
        "significantly to the overall Chi-Square statistic at approximately 95% confidence."
    )

with exp_tabs[2]:
    st.write("### Cramer's V Effect Size Measure")
    st.markdown("$$V = \\sqrt{\\frac{\\chi^2}{N \\times \\min(r - 1, c - 1)}}$$")
    v_c1, v_c2 = st.columns([1, 2])
    with v_c1:
        st.metric("Cramer's V Value", f"{cramers_v:.4f}")
        st.info(f"**Classification:** {effect_label}")
    with v_c2:
        st.write(f"**Interpretation:** {effect_desc}")
        st.write(
            "Unlike the p-value (which is heavily influenced by sample size N), "
            "Cramer's V provides a normalized effect size between 0.0 (no association) "
            "and 1.0 (perfect association) that can be compared across different studies and models."
        )


# ==============================================================================
# 11. INTERACTIVE PLOTLY VISUALIZATIONS
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">📉 Interactive Statistical Visualizations</div>', unsafe_allow_html=True)

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("1. Grouped Category Distribution")
    chart_mode = st.radio(
        "Bar Chart Display Mode:",
        options=["Absolute Frequency (Counts)", "Within-Group Relative Percentage (%)"],
        horizontal=True,
        key="bar_mode_toggle"
    )
    
    # Prepare data for grouped bar chart
    if chart_mode == "Absolute Frequency (Counts)":
        chart_df = clean_df.groupby([feature_x, feature_y]).size().reset_index(name="Count")
        fig_bar = px.bar(
            chart_df,
            x=feature_x,
            y="Count",
            color=feature_y,
            barmode="group",
            title=f"Distribution of {feature_y} across {feature_x} (Counts)",
            labels={feature_x: str(feature_x), "Count": "Observed Frequency", feature_y: str(feature_y)},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
    else:
        chart_df = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='index').reset_index()
        chart_df = pd.melt(chart_df, id_vars=feature_x, var_name=feature_y, value_name="Percentage")
        chart_df["Percentage"] = chart_df["Percentage"] * 100
        fig_bar = px.bar(
            chart_df,
            x=feature_x,
            y="Percentage",
            color=feature_y,
            barmode="group",
            title=f"Relative Distribution of {feature_y} within {feature_x} (%)",
            labels={feature_x: str(feature_x), "Percentage": "Within-Group Percentage (%)", feature_y: str(feature_y)},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_bar.update_layout(yaxis=dict(ticksuffix="%"))
        
    fig_bar.update_layout(
        template="plotly_white",
        legend_title_text=str(feature_y),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig_bar, width="stretch")

with vis_col2:
    st.subheader("2. Contingency Table Heatmap")
    heatmap_view = st.radio(
        "Heatmap Data Metric:",
        options=["Observed Frequencies (O_ij)", "Expected Frequencies (E_ij)", "Standardized Residuals (O-E)/√E"],
        horizontal=True,
        key="heatmap_mode_toggle"
    )
    
    if heatmap_view == "Observed Frequencies (O_ij)":
        heat_matrix = observed_crosstab.values
        heat_text = heat_matrix.astype(str)
        heat_title = f"Observed Counts Heatmap: {feature_x} vs {feature_y}"
        color_scale = "Blues"
    elif heatmap_view == "Expected Frequencies (E_ij)":
        heat_matrix = expected_df.values
        heat_text = np.round(heat_matrix, 2).astype(str)
        heat_title = f"Expected Counts Heatmap: {feature_x} vs {feature_y}"
        color_scale = "Teal"
    else:
        heat_matrix = residuals_df.values
        heat_text = np.round(heat_matrix, 2).astype(str)
        heat_title = f"Standardized Residuals Heatmap: {feature_x} vs {feature_y}"
        color_scale = "RdBu_r"
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_matrix,
        x=list(observed_crosstab.columns),
        y=list(observed_crosstab.index),
        text=heat_text,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale=color_scale,
        colorbar=dict(title="Value")
    ))
    
    fig_heat.update_layout(
        title=heat_title,
        xaxis_title=str(feature_y),
        yaxis_title=str(feature_x),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_heat, width="stretch")


# ==============================================================================
# 12. MACHINE LEARNING CONTEXT & PRACTICAL SIGNIFICANCE
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">🤖 Machine Learning & Feature Engineering Application</div>', unsafe_allow_html=True)

with st.expander("💡 Why Feature Independence Validation Matters in Machine Learning", expanded=True):
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        st.markdown("""
        #### 1. Categorical Feature Selection
        - In classification algorithms (e.g., Decision Trees, Random Forests, Logistic Regression), we want features that are **strongly associated with the target variable**.
        - Scikit-learn's `SelectKBest(score_func=chi2)` utilizes this exact test to rank categorical predictors and filter out uninformative noise variables.
        - **Practical Rule:** If testing a candidate predictor against the target label yields a significant p-value ($p < \\alpha$), the feature carries discriminative predictive power!
        """)
    with col_ml2:
        st.markdown("""
        #### 2. Multicollinearity & Redundancy Detection
        - When testing **two input features** against one another, a high Chi-Square statistic and large Cramer's V indicate **feature redundancy**.
        - Retaining two strongly associated input features can inflate model variance, complicate feature importance attribution, and unnecessarily increase dimensionality.
        - **Feature Engineering Rule:** If two categorical features exhibit near-perfect association, one can often be pruned or combined without sacrificing model accuracy.
        """)


# ==============================================================================
# 13. DOWNLOADABLE ANALYSIS REPORT & SUMMARY
# ==============================================================================

st.divider()
st.markdown('<div class="section-title">📄 Downloadable Statistical Analysis Report</div>', unsafe_allow_html=True)

full_report_text = generate_text_report(
    project_title="Feature Independence Validation Using Chi-Square Test in Machine Learning Models",
    dataset_name=dataset_label,
    var_x=feature_x,
    var_y=feature_y,
    binning_info_x=info_x,
    binning_info_y=info_y,
    total_records=total_records,
    analyzed_records=analyzed_count,
    excluded_records=excluded_count,
    observed_df=observed_with_margins,
    expected_df=expected_df,
    residuals_df=residuals_df,
    chi2_stat=chi2_stat,
    dof=dof,
    p_value=p_value,
    alpha=alpha,
    critical_val=critical_val,
    decision=decision_text,
    cramers_v=cramers_v,
    effect_label=effect_label,
    cochran_valid=cochran_valid,
    cells_below_5_pct=cells_below_5_pct,
    min_expected=min_expected
)

rep_col1, rep_col2 = st.columns([2, 1])

with rep_col1:
    st.text_area(
        "Preview of Generated Statistical Report:",
        value=full_report_text,
        height=320,
        disabled=True
    )

with rep_col2:
    st.write("### Export Options")
    st.write("Download the complete mathematical and statistical analysis summary for your project documentation or lab record:")
    
    st.download_button(
        label="📥 Download Full Analysis Report (.txt)",
        data=full_report_text.encode('utf-8'),
        file_name=f"ChiSquare_Report_{feature_x}_vs_{feature_y}.txt",
        mime="text/plain",
        width="stretch"
    )
    
    st.download_button(
        label="📥 Download Observed Contingency Table (.csv)",
        data=observed_with_margins.to_csv().encode('utf-8'),
        file_name=f"Contingency_Table_{feature_x}_vs_{feature_y}.csv",
        mime="text/csv",
        width="stretch"
    )
    
    st.download_button(
        label="📥 Download Expected Frequencies (.csv)",
        data=expected_df.round(4).to_csv().encode('utf-8'),
        file_name=f"Expected_Frequencies_{feature_x}_vs_{feature_y}.csv",
        mime="text/csv",
        width="stretch"
    )


# ==============================================================================
# 14. FOOTER
# ==============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    <strong>Feature Independence Validation Using Chi-Square Test in Machine Learning Models</strong><br>
    College Micro-Project | Computer Science & Engineering | Built with Python, Streamlit, Pandas, SciPy & Plotly
</div>
""", unsafe_allow_html=True)