"""
Feature Independence Validation Using Chi-Square Test
Streamlit Micro-Project Web Application
Department of Computer Science & Engineering | Mathematics & Statistics
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
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional academic dashboard aesthetics
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
        padding: 1.6rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 950px;
        margin-bottom: 0.6rem;
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .hero-badge-sub {
        display: inline-block;
        background-color: #059669;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .section-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 50%;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.35rem;
    }

    /* Decision Containers */
    .decision-box-reject {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .decision-box-fail {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #64748b;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .decision-box-warn {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }

    /* Compact Note Callouts */
    .academic-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.45;
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
        st.error(f"Error loading dataset/student_data.csv: {e}")
        return None

@st.cache_data
def load_demonstration_data():
    """Load the synthetic demonstration survey dataset."""
    try:
        return pd.read_csv("dataset/demonstration_survey_data.csv")
    except Exception as e:
        st.error(f"Error loading dataset/demonstration_survey_data.csv: {e}")
        return None

def compute_cramers_v(chi2_stat, n, r, c):
    """
    Compute Cramer's V effect size measure.
    Formula: V = sqrt(chi2 / (n * min(r - 1, c - 1)))
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
    Determine the qualitative strength of Cramer's V based on degrees of freedom k = min(r-1, c-1).
    Using standard Cohen (1988) and Rea & Parker benchmark guidelines.
    """
    k = min(r - 1, c - 1)
    if k <= 1:
        if v < 0.10:
            return "Negligible", "Practically negligible association between the two features."
        elif v < 0.30:
            return "Weak", "Weak association between the two features."
        elif v < 0.50:
            return "Moderate", "Moderate association between the two features."
        else:
            return "Strong", "Strong association between the two features."
    elif k == 2:
        if v < 0.07:
            return "Negligible", "Practically negligible association between the two features."
        elif v < 0.21:
            return "Weak", "Weak association between the two features."
        elif v < 0.35:
            return "Moderate", "Moderate association between the two features."
        else:
            return "Strong", "Strong association between the two features."
    else:
        if v < 0.06:
            return "Negligible", "Practically negligible association between the two features."
        elif v < 0.17:
            return "Weak", "Weak association between the two features."
        elif v < 0.29:
            return "Moderate", "Moderate association between the two features."
        else:
            return "Strong", "Strong association between the two features."

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
    min_expected,
    alt_test_notes=""
):
    """Generate a clean, structured plaintext report for documentation export."""
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
    
    report.append("\n1. OBSERVED CONTINGENCY TABLE (Cross-Tabulation with Margins)")
    report.append("-" * 78)
    report.append(observed_df.to_string())
    
    report.append("\n\n2. EXPECTED FREQUENCIES TABLE (E_ij = R_i * C_j / N)")
    report.append("-" * 78)
    report.append(expected_df.round(3).to_string())
    
    report.append("\n\n3. STANDARDIZED PEARSON RESIDUALS ((O - E) / sqrt(E))")
    report.append("-" * 78)
    report.append(residuals_df.round(3).to_string())
    
    # Display small p-values in scientific notation to avoid rounding to 0.00000
    p_val_display = f"{p_value:.6e}" if p_value < 0.0001 else f"{p_value:.5f}"

    report.append("\n\n4. CHI-SQUARE TEST OF INDEPENDENCE RESULTS")
    report.append("-" * 78)
    report.append(f"Chi-Square Statistic (X^2) : {chi2_stat:.5f}")
    report.append(f"Degrees of Freedom (df)     : {dof}")
    report.append(f"P-Value                     : {p_val_display}")
    report.append(f"Critical Value (X^2_crit)   : {critical_val:.5f} (at a = {alpha})")
    report.append(f"Effect Size (Cramer's V)    : {cramers_v:.4f} [{effect_label}]")
    
    report.append("\n5. HYPOTHESES & STATISTICAL DECISION")
    report.append("-" * 78)
    report.append("H0 (Null Hypothesis)       : Feature X and Feature Y are statistically independent.")
    report.append("H1 (Alternative Hypothesis): Feature X and Feature Y are statistically associated.")
    report.append(f"Decision Rule              : If p-value < {alpha} (or X^2 > X^2_crit), Reject H0; Else Fail to Reject H0.")
    report.append(f"Final Decision             : {decision.upper()}")
    
    if "REJECT" in decision.upper() and "FAIL" not in decision.upper():
        report.append(
            f"Interpretation             : At the {alpha} significance level, the p-value ({p_val_display}) is less\n"
            f"                             than {alpha}. There is statistically significant evidence to reject H0.\n"
            f"                             A significant association exists between {var_x} and {var_y}."
        )
    else:
        report.append(
            f"Interpretation             : At the {alpha} significance level, the p-value ({p_val_display}) is greater\n"
            f"                             than or equal to {alpha}. We fail to reject H0. The sample does not provide\n"
            f"                             sufficient statistical evidence of association."
        )
    report.append("\n* CAUTION: Statistical association does not imply a causal relationship.")
    report.append("           Failing to reject H0 does not prove absolute independence.")
    
    report.append("\n6. STATISTICAL ASSUMPTION AUDIT (Cochran's Rule)")
    report.append("-" * 78)
    report.append(f"Minimum Expected Frequency : {min_expected:.3f} (Required >= 1.0)")
    report.append(f"Cells with Expected < 5    : {cells_below_5_pct:.1f}% (Required <= 20.0%)")
    if cochran_valid:
        report.append("Assumption Audit Status    : PASSED. Asymptotic Chi-Square approximation is reliable.")
    else:
        report.append("Assumption Audit Status    : WARNING. Small expected frequencies detected.")
        report.append("                             More than 20% of cells have expected frequency < 5, or min < 1.")
        report.append("                             The standard asymptotic p-value should be interpreted with caution.")
    if alt_test_notes:
        report.append(f"\nAlternative Test Findings  :\n{alt_test_notes}")
        
    report.append("\n7. DOWNLOADABLE STATISTICAL ANALYSIS REPORT")
    report.append("-" * 78)
    report.append(
        "This report was compiled and exported from the Chi-Square Feature Independence\n"
        "Validator dashboard. All statistical computations, contingency distributions,\n"
        "and diagnostic validity checks are finalized and verified.\n\n"
        "Associated Export Files:\n"
        f"  - Full Analysis Report (.txt)         : ChiSquare_Report_{var_x}_vs_{var_y}.txt\n"
        f"  - Observed Contingency Table (.csv)   : Contingency_Table_{var_x}_vs_{var_y}.csv\n"
        f"  - Expected Frequencies Matrix (.csv) : Expected_Frequencies_{var_x}_vs_{var_y}.csv"
    )
    report.append("=" * 78)
    report.append("END OF REPORT")
    report.append("=" * 78)
    
    return "\n".join(report)


# ==============================================================================
# 3. APPLICATION HEADER & INTRODUCTION
# ==============================================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badges">
        <span class="hero-badge">Hypothesis Testing</span>
        <span class="hero-badge-sub">Feature Independence Validation</span>
    </div>
    <div class="hero-title">Feature Independence Validation Using Chi-Square Test</div>
    <div class="hero-subtitle">
        An interactive micro-project dashboard for evaluating statistical independence, contingency cross-tabulation, 
        and association strength between categorical features in machine learning datasets.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.title("Controls & Settings")

dataset_choice = st.sidebar.selectbox(
    "Select Dataset Source",
    options=[
        "Student Academic Dataset (Default - 42 records)",
        "Demonstration Academic Survey (Synthetic - 350 records)",
        "Upload Custom CSV File"
    ],
    index=0,
    help="Choose between the original student dataset, a larger demonstration survey, or your custom CSV."
)

df_raw = None
dataset_label = ""

if dataset_choice == "Student Academic Dataset (Default - 42 records)":
    df_raw = load_default_student_data()
    dataset_label = "Student Academic Dataset (dataset/student_data.csv)"
    st.sidebar.caption("Loaded 42 student records with academic performance metrics.")

elif dataset_choice == "Demonstration Academic Survey (Synthetic - 350 records)":
    df_raw = load_demonstration_data()
    dataset_label = "Demonstration Academic Survey (dataset/demonstration_survey_data.csv)"
    st.sidebar.caption("Loaded 350 survey records ideal for large-sample Chi-Square validation.")

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
            st.sidebar.success(f"Loaded `{uploaded_file.name}` ({len(df_raw)} records)")
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file in the sidebar to proceed, or choose one of the built-in datasets.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.error("Dataset could not be loaded or is empty. Please verify the dataset path or file format.")
    st.stop()

st.sidebar.divider()

# Significance level alpha
st.sidebar.subheader("Significance Level (α)")
alpha = st.sidebar.radio(
    "Select α threshold:",
    options=[0.01, 0.05, 0.10],
    index=1,
    format_func=lambda x: f"α = {x} ({int(x*100)}% Type I Error Rate)",
    help="The probability of rejecting the null hypothesis when it is actually true. α = 0.05 is standard."
)

st.sidebar.divider()

# Theory Guide Expander (Placed in sidebar expander to prevent a permanently long sidebar)
with st.sidebar.expander("Theoretical Guide & Key Formulas", expanded=False):
    st.markdown(r"""
    **Chi-Square ($\chi^2$) Test of Independence:**
    $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
    
    **Degrees of Freedom:**
    $$df = (r - 1) \times (c - 1)$$
    
    **Expected Frequencies:**
    $$E_{ij} = \frac{R_i \times C_j}{N}$$
    
    **Decision Rule:**
    - If **$p\text{-value} < \alpha$**, Reject $H_0$ (Association detected).
    - Otherwise, Fail to Reject $H_0$ (Independence not rejected).
    
    **Cochran's Rule (1954):**
    - $E_{ij} \ge 1.0$ for all cells.
    - At least 80% of cells have $E_{ij} \ge 5.0$.
    
    **Cramér's V:**
    $$V = \sqrt{\frac{\chi^2}{N \times \min(r - 1, c - 1)}}$$
    """)


# ==============================================================================
# SECTION 1: DATASET OVERVIEW & PREVIEW
# ==============================================================================

st.markdown('<div class="section-header"><span class="section-num">1</span> Dataset Overview & Inspection</div>', unsafe_allow_html=True)

total_records = len(df_raw)
total_columns = len(df_raw.columns)
missing_cells = int(df_raw.isna().sum().sum())
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df_raw.columns if c not in num_cols]

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Total Records (N)", total_records)
with m_col2:
    st.metric("Total Features (P)", total_columns)
with m_col3:
    st.metric("Missing Values", missing_cells)
with m_col4:
    st.metric("Categorical Features", len(cat_cols))
with m_col5:
    st.metric("Numerical Features", len(num_cols))

st.dataframe(df_raw.head(10), use_container_width=True)
st.caption(f"Showing first 10 rows from `{dataset_label}`.")

# Optional Column Metadata Expander
with st.expander("View Column Metadata & Data Types", expanded=False):
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
            "Type": col_type,
            "Data Type": dt,
            "Non-Null": total_records - n_null,
            "Null Count": n_null,
            "Null %": f"{null_pct:.1f}%",
            "Unique Values": n_unique,
            "Sample Values": sample_vals
        })
    meta_df = pd.DataFrame(meta_records)
    st.dataframe(meta_df, use_container_width=True, hide_index=True)

# Export Dataset
csv_data = df_raw.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Current Dataset (CSV)",
    data=csv_data,
    file_name=f"dataset_export_{datetime.date.today()}.csv",
    mime="text/csv",
    help="Export active dataset as a CSV file."
)


# ==============================================================================
# SECTION 2: VARIABLE SELECTION & FEATURE CATEGORIZATION
# ==============================================================================

st.divider()
st.markdown('<div class="section-header"><span class="section-num">2</span> Variable Selection & Feature Discretization</div>', unsafe_allow_html=True)
st.write(
    "Select two distinct variables to test for statistical independence. "
    "If a numerical feature is chosen, it will be automatically converted into discrete categories."
)

col_sel1, col_sel2 = st.columns(2)

all_columns = list(df_raw.columns)
default_idx1 = 0
default_idx2 = 1 if len(all_columns) > 1 else 0

# Sensible default selection
if "Attendance" in all_columns and "Performance" in all_columns:
    default_idx1 = all_columns.index("Attendance")
    default_idx2 = all_columns.index("Performance")
elif "Attendance_Level" in all_columns and "Performance_Band" in all_columns:
    default_idx1 = all_columns.index("Attendance_Level")
    default_idx2 = all_columns.index("Performance_Band")

with col_sel1:
    feature_x = st.selectbox(
        "Feature 1 (Row Variable X):",
        options=all_columns,
        index=default_idx1,
        key="feature_x_sel"
    )

with col_sel2:
    feature_y = st.selectbox(
        "Feature 2 (Column Variable Y):",
        options=all_columns,
        index=default_idx2,
        key="feature_y_sel"
    )

# Strict validation that variables are different
if feature_x == feature_y:
    st.error("Validation Error: Feature 1 and Feature 2 must be different variables. Please pick two distinct columns.")
    st.stop()


def process_feature(df, feature_name, col_key_suffix):
    """
    Examines a column. If categorical, returns clean series and metadata.
    If numerical, provides safe, configurable binning controls and returns discretized series.
    """
    col_series = df[feature_name].copy()
    is_numerical = pd.api.types.is_numeric_dtype(col_series)
    
    if not is_numerical:
        n_cats = col_series.nunique()
        st.success(f"`{feature_name}` is categorical with **{n_cats}** distinct categories.")
        
        # Optional consolidation of rare categories if many categories exist
        consolidate_rare = st.checkbox(
            f"Consolidate sparse categories in `{feature_name}` (counts < 2 into 'Other')",
            value=False,
            key=f"rare_toggle_{col_key_suffix}",
            help="Combines infrequent categories to help satisfy Cochran's minimum cell frequency criteria."
        )
        if consolidate_rare:
            counts = col_series.value_counts()
            rare_cats = counts[counts < 2].index.tolist()
            if len(rare_cats) > 0:
                col_series = col_series.replace(rare_cats, "Other")
                st.caption(f"Consolidated {len(rare_cats)} sparse categories: {', '.join(map(str, rare_cats))}")
                
        return col_series.astype(str), f"[Categorical, {col_series.nunique()} categories]"
    
    # Numerical feature -> Discretization
    c_min = float(col_series.min())
    c_max = float(col_series.max())
    
    if c_min == c_max:
        st.warning(f"`{feature_name}` has constant values ({c_min}). Cannot discretize a constant feature.")
        return col_series.astype(str), "[Constant numerical feature]"
    
    st.info(f"`{feature_name}` is numerical (Range: {c_min:.1f} to {c_max:.1f}). Discretize into categories:")
    
    bin_method = st.radio(
        f"Binning Method for `{feature_name}`:",
        options=["Configurable Cutoffs (Custom)", "Quantile Bins (Equal Frequency)", "Equal Width Intervals"],
        key=f"bin_method_{col_key_suffix}",
        horizontal=True
    )
    
    if bin_method == "Configurable Cutoffs (Custom)":
        num_bins = st.selectbox(
            f"Number of categories for `{feature_name}`:",
            options=[2, 3, 4],
            index=1,
            key=f"num_bins_{col_key_suffix}",
            help="Note: Using 2 categories yields larger cell frequencies, which helps satisfy Cochran's assumptions on small samples."
        )
        
        # Domain presets
        presets = {}
        if feature_name == "Attendance" and c_max <= 100:
            if num_bins == 3:
                presets = {"cut1": 70.0, "cut2": 85.0}
            elif num_bins == 2:
                presets = {"cut1": 75.0}
        elif feature_name == "Study_Hours":
            if num_bins == 3:
                presets = {"cut1": 2.5, "cut2": 4.0}
        elif "Marks" in feature_name or "Score" in feature_name:
            if num_bins == 3:
                presets = {"cut1": 65.0, "cut2": 80.0}
        
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
                    f"Cutoff 1 (Low/Medium):",
                    value=float(default_cut1),
                    key=f"cut1_3_{col_key_suffix}"
                )
            with c_cut2:
                cut2 = st.number_input(
                    f"Cutoff 2 (Medium/High):",
                    value=float(default_cut2),
                    key=f"cut2_3_{col_key_suffix}"
                )
            if cut1 >= cut2:
                st.warning("Cutoff 1 must be strictly less than Cutoff 2. Cutoff 2 has been adjusted.")
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
            if cut1 >= cut2:
                cut2 = cut1 + 0.5
            if cut2 >= cut3:
                cut3 = cut2 + 0.5
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
            st.warning(f"Could not compute equal quantiles ({e}). Using equal-width bins.")
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
        
    counts = binned.value_counts().sort_index()
    dist_str = " | ".join([f"**{cat}**: {count}" for cat, count in counts.items()])
    st.caption(f"Category counts: {dist_str}")
    
    return binned.astype(str), bin_desc


col_proc1, col_proc2 = st.columns(2)
with col_proc1:
    st.markdown(f"**Variable X:** `{feature_x}`")
    series_x, info_x = process_feature(df_raw, feature_x, "x")

with col_proc2:
    st.markdown(f"**Variable Y:** `{feature_y}`")
    series_y, info_y = process_feature(df_raw, feature_y, "y")


# ==============================================================================
# SECTION 3: CONTINGENCY TABLE CREATION
# ==============================================================================

# Align series and handle missing values
analysis_df = pd.DataFrame({
    feature_x: series_x,
    feature_y: series_y
})

initial_count = len(analysis_df)
valid_mask = (
    analysis_df[feature_x].notna() &
    analysis_df[feature_y].notna() &
    (~analysis_df[feature_x].astype(str).str.strip().str.lower().isin(["nan", "none", "null", ""])) &
    (~analysis_df[feature_y].astype(str).str.strip().str.lower().isin(["nan", "none", "null", ""]))
)
clean_df = analysis_df[valid_mask]
analyzed_count = len(clean_df)
excluded_count = initial_count - analyzed_count

if excluded_count > 0:
    st.warning(f"Missing Value Handling: {excluded_count} record(s) excluded due to missing/invalid values. Analyzing **{analyzed_count}** valid observations.")
else:
    st.caption(f"All {analyzed_count} records have complete data for `{feature_x}` and `{feature_y}`.")

# Validate minimum unique categories
unique_x = clean_df[feature_x].nunique()
unique_y = clean_df[feature_y].nunique()

if unique_x < 2 or unique_y < 2:
    st.error(
        f"Insufficient Categories: `{feature_x}` has {unique_x} category, and `{feature_y}` has {unique_y} category. "
        f"The Chi-Square Test requires at least 2 categories per variable. Please adjust your binning or select another feature."
    )
    st.stop()

# Observed contingency table
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
st.markdown('<div class="section-header"><span class="section-num">3</span> Observed Contingency Table (Cross-Tabulation)</div>', unsafe_allow_html=True)

tab_obs1, tab_obs2 = st.tabs(["Observed Counts with Marginal Totals", "Proportions & Relative Frequencies (%)"])

with tab_obs1:
    st.write(f"Cross-tabulation matrix $O_{{ij}}$ showing joint frequencies for **{feature_x}** (rows) vs **{feature_y}** (columns):")
    st.dataframe(observed_with_margins, use_container_width=True)
    st.caption(f"Dimensions: {observed_crosstab.shape[0]} rows × {observed_crosstab.shape[1]} columns. Total sample size N = {analyzed_count}.")

with tab_obs2:
    prop_type = st.radio(
        "Display Proportions As:",
        options=["Row Percentages (%)", "Column Percentages (%)", "Total Table Percentages (%)"],
        horizontal=True
    )
    if prop_type == "Row Percentages (%)":
        row_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='index') * 100
        st.dataframe(row_props.round(2).astype(str) + "%", use_container_width=True)
        st.caption("Each row sums to 100.0%. Shows the conditional distribution of Feature Y given Feature X.")
    elif prop_type == "Column Percentages (%)":
        col_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='columns') * 100
        st.dataframe(col_props.round(2).astype(str) + "%", use_container_width=True)
        st.caption("Each column sums to 100.0%. Shows the conditional distribution of Feature X given Feature Y.")
    else:
        tot_props = pd.crosstab(clean_df[feature_x], clean_df[feature_y], normalize='all') * 100
        st.dataframe(tot_props.round(2).astype(str) + "%", use_container_width=True)
        st.caption("All cells sum to 100.0%. Shows the joint relative frequency distribution.")

csv_crosstab = observed_with_margins.to_csv().encode('utf-8')
st.download_button(
    label="Download Contingency Table (CSV)",
    data=csv_crosstab,
    file_name=f"contingency_table_{feature_x}_vs_{feature_y}.csv",
    mime="text/csv",
    help="Download observed contingency table as CSV."
)


# ==============================================================================
# SECTION 4: CHI-SQUARE TEST OF INDEPENDENCE & DECISION
# ==============================================================================

# SciPy Chi-Square computation
chi2_stat, p_value, dof, expected_matrix = stats.chi2_contingency(observed_crosstab, correction=False)

expected_df = pd.DataFrame(
    expected_matrix,
    index=observed_crosstab.index,
    columns=observed_crosstab.columns
)

# Standardized Pearson residuals: (O - E) / sqrt(E)
residuals_df = (observed_crosstab - expected_df) / np.sqrt(expected_df)

# Critical value
critical_val = stats.chi2.ppf(1 - alpha, dof)

# Effect size: Cramer's V
r_dim, c_dim = observed_crosstab.shape
cramers_v = compute_cramers_v(chi2_stat, analyzed_count, r_dim, c_dim)
effect_label, effect_desc = interpret_cramers_v(cramers_v, r_dim, c_dim)

# Cochran's (1954) criteria validation:
# 1. No expected frequency < 1.0
# 2. At most 20% of expected frequencies < 5.0
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

# Alternative exact tests if 2x2
is_2x2 = (r_dim == 2 and c_dim == 2)
p_fisher = None
chi2_yates = None
p_yates = None

if is_2x2:
    try:
        fisher_res = stats.fisher_exact(observed_crosstab.values)
        p_fisher = float(fisher_res.pvalue)
        chi2_yates, p_yates, _, _ = stats.chi2_contingency(observed_crosstab, correction=True)
    except Exception:
        pass

st.divider()
st.markdown('<div class="section-header"><span class="section-num">4</span> Hypothesis Testing & Chi-Square Results</div>', unsafe_allow_html=True)

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

# Hypothesis & Decision Banner
if reject_null:
    st.markdown(f"""
    <div class="decision-box-reject">
        <h4 style="color: #15803d; margin: 0 0 0.4rem 0;">Statistical Decision: {decision_text}</h4>
        <p style="color: #166534; font-size: 0.95rem; margin-bottom: 0.35rem;">
            <strong>Result:</strong> Statistically significant association detected between 
            <code>{feature_x}</code> and <code>{feature_y}</code> at the α = {alpha} significance level.
        </p>
        <p style="color: #14532d; font-size: 0.9rem; margin-bottom: 0;">
            <strong>Test Conclusion:</strong> Because the p-value (<strong>{p_format}</strong>) is strictly less than 
            α = {alpha}, and the test statistic (<strong>χ² = {chi2_stat:.4f}</strong>) exceeds the critical value 
            (<strong>χ²_crit = {critical_val:.4f}</strong>), we reject the null hypothesis of independence.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="decision-box-fail">
        <h4 style="color: #334155; margin: 0 0 0.4rem 0;">Statistical Decision: {decision_text}</h4>
        <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0.35rem;">
            <strong>Result:</strong> Insufficient statistical evidence to reject H₀ at the α = {alpha} significance level.
        </p>
        <p style="color: #1e293b; font-size: 0.9rem; margin-bottom: 0;">
            <strong>Test Conclusion:</strong> Because the p-value (<strong>{p_format}</strong>) is greater than or equal to 
            α = {alpha}, we fail to reject H₀. The observed variation is consistent with random chance.
        </p>
    </div>
    """, unsafe_allow_html=True)

col_warn1, col_warn2 = st.columns(2)
with col_warn1:
    st.caption(
        "**Note on Causation:** Association does not imply causation. A significant Chi-Square test indicates "
        "that category distributions differ, but does not prove that one variable causes changes in the other."
    )
with col_warn2:
    st.caption(
        "**Note on Independence:** Failing to reject H₀ does not prove absolute mathematical independence; "
        "it only indicates that any association present was undetectable at this sample size and significance level."
    )


# ==============================================================================
# SECTION 5: ASSUMPTION AUDIT & DIAGNOSTIC TABLES
# ==============================================================================

st.divider()
st.markdown('<div class="section-header"><span class="section-num">5</span> Statistical Assumption Audit (Cochran\'s Rule) & Diagnostics</div>', unsafe_allow_html=True)

alt_report_notes = ""

if cochran_valid:
    st.success(
        f"Assumptions Fully Satisfied: Minimum expected cell count is {min_expected:.2f} (≥ 1.0), "
        f"and {cells_below_5} of {total_cells} cells ({cells_below_5_pct:.1f}%) have expected count < 5.0 (≤ 20.0%). "
        f"The asymptotic Chi-Square distribution approximation is reliable."
    )
else:
    st.markdown(f"""
    <div class="decision-box-warn">
        <h4 style="color: #92400e; margin: 0 0 0.35rem 0;">⚠️ Cochran's Assumption Warning: Small Expected Frequencies Detected</h4>
        <p style="color: #78350f; font-size: 0.92rem; margin-bottom: 0.4rem;">
            Pearson's Chi-Square test relies on large-sample asymptotic approximation. Cochran (1954) established that:
        </p>
        <ul style="color: #78350f; font-size: 0.88rem; margin-bottom: 0.4rem;">
            <li>No expected cell count should be less than 1.0 (Current minimum: <strong>{min_expected:.2f}</strong>).</li>
            <li>No more than 20% of expected cell counts should be less than 5.0 (Current: <strong>{cells_below_5_pct:.1f}%</strong> — {cells_below_5} of {total_cells} cells).</li>
        </ul>
        <p style="color: #92400e; font-size: 0.9rem; margin-bottom: 0;">
            <strong>Caution:</strong> When expected counts are small, the true distribution deviates from theoretical χ², 
            which can inflate the Type I error rate. The reported p-value should be interpreted cautiously rather than as definitive proof.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Practical remedies
    if is_2x2 and p_fisher is not None:
        p_fisher_fmt = f"{p_fisher:.4e}" if p_fisher < 0.0001 else f"{p_fisher:.4f}"
        p_yates_fmt = f"{p_yates:.4e}" if p_yates < 0.0001 else f"{p_yates:.4f}"
        st.info(
            f"**Exact Alternative for 2×2 Table:** "
            f"Because this is a 2×2 contingency table, you can rely on **Fisher's Exact Test**, which computes the exact hypergeometric probability without asymptotic assumptions: "
            f"**Fisher's Exact p-value = {p_fisher_fmt}** (Yates' corrected χ² = {chi2_yates:.4f}, p = {p_yates_fmt})."
        )
        p_fisher_rep = f"{p_fisher:.6e}" if p_fisher < 0.0001 else f"{p_fisher:.6f}"
        p_yates_rep = f"{p_yates:.6e}" if p_yates < 0.0001 else f"{p_yates:.6f}"
        alt_report_notes = (
            f"Fisher's Exact Test p-value : {p_fisher_rep}\n"
            f"Yates' Corrected Chi-Square : {chi2_yates:.4f} (p = {p_yates_rep})"
        )
    else:
        st.info(
            "**Recommended Remedies:** "
            "(1) Consolidate sparse categories into fewer categories (e.g. choose 2 bins for numerical variables in Section 2); "
            "(2) Evaluate the larger 350-record Demonstration survey dataset from the sidebar; "
            "(3) In your college micro-project presentation, explain that small sample sizes naturally trigger Cochran warnings, demonstrating statistical rigor!"
        )

# Diagnostic Tabs
exp_tabs = st.tabs([
    "Expected Frequencies Matrix (E_ij)",
    "Standardized Residuals (r_ij)",
    "Effect Size Interpretation (Cramér's V)",
    "Alternative Tests & Small-Sample Notes"
])

with exp_tabs[0]:
    st.write("Expected frequencies calculated under the assumption of complete independence:")
    st.markdown(r"$$E_{ij} = \frac{R_i \times C_j}{N}$$")
    st.dataframe(expected_df.round(2), use_container_width=True)
    st.caption("Each cell represents the expected count under H₀.")

with exp_tabs[1]:
    st.write("Standardized Pearson residuals identify which category combinations deviate most from expectation:")
    st.markdown(r"$$r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}$$")
    st.dataframe(residuals_df.round(2), use_container_width=True)
    st.caption("Residuals with absolute value > 2.0 indicate cells contributing significantly to the association at ~95% confidence.")

with exp_tabs[2]:
    st.write("**Cramér's V Effect Size Measure:**")
    st.markdown(r"$$V = \sqrt{\frac{\chi^2}{N \times \min(r - 1, c - 1)}}$$")
    v_c1, v_c2 = st.columns([1, 2])
    with v_c1:
        st.metric("Cramér's V Value", f"{cramers_v:.4f}")
        st.info(f"**Classification:** {effect_label}")
    with v_c2:
        st.write(f"**Interpretation:** {effect_desc}")
        st.caption("Unlike the p-value (which shrinks with large N), Cramér's V measures the true magnitude of association between 0 (none) and 1 (perfect).")

with exp_tabs[3]:
    st.write("**Small-Sample & Alternative Testing Diagnostics:**")
    if is_2x2 and p_fisher is not None:
        p_fisher_disp = f"{p_fisher:.4e}" if p_fisher < 0.0001 else f"{p_fisher:.4f}"
        p_yates_disp = f"{p_yates:.4e}" if p_yates < 0.0001 else f"{p_yates:.4f}"
        c_alt1, c_alt2 = st.columns(2)
        with c_alt1:
            st.metric("Fisher's Exact Test (p-value)", p_fisher_disp)
            st.caption("Exact conditional hypergeometric test. Recommended when Cochran's assumptions fail on 2×2 tables.")
        with c_alt2:
            st.metric("Yates' Continuity-Corrected χ²", f"{chi2_yates:.4f}")
            st.caption(f"Continuity-adjusted p-value = {p_yates_disp}")
    else:
        st.write(
            "For contingency tables larger than 2×2 where Cochran's rule fails, exact tests require extensive permutations. "
            "The standard statistical remedy is category consolidation (merging infrequent categories into an 'Other' group or reducing bin counts)."
        )


# ==============================================================================
# SECTION 6: INTERACTIVE VISUALIZATIONS
# ==============================================================================

st.divider()
st.markdown('<div class="section-header"><span class="section-num">6</span> Interactive Statistical Visualizations</div>', unsafe_allow_html=True)

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("Grouped Distribution Chart")
    chart_mode = st.radio(
        "Display Format:",
        options=["Absolute Frequency (Counts)", "Within-Group Relative Percentage (%)"],
        horizontal=True,
        key="bar_mode_toggle"
    )
    
    if chart_mode == "Absolute Frequency (Counts)":
        chart_df = clean_df.groupby([feature_x, feature_y]).size().reset_index(name="Count")
        fig_bar = px.bar(
            chart_df,
            x=feature_x,
            y="Count",
            color=feature_y,
            barmode="group",
            title=f"Distribution of {feature_y} across {feature_x}",
            labels={feature_x: str(feature_x), "Count": "Observed Frequency", feature_y: str(feature_y)},
            color_discrete_sequence=px.colors.qualitative.Safe
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
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_layout(yaxis=dict(ticksuffix="%"))
        
    fig_bar.update_layout(
        template="plotly_white",
        legend_title_text=str(feature_y),
        margin=dict(l=30, r=30, t=40, b=30),
        hovermode="x unified"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with vis_col2:
    st.subheader("Contingency Heatmap")
    heatmap_view = st.radio(
        "Heatmap Metric:",
        options=["Observed Frequencies (O_ij)", "Expected Frequencies (E_ij)", "Standardized Residuals (r_ij)"],
        horizontal=True,
        key="heatmap_mode_toggle"
    )
    
    if heatmap_view == "Observed Frequencies (O_ij)":
        heat_matrix = observed_crosstab.values
        heat_text = heat_matrix.astype(str)
        heat_title = f"Observed Counts: {feature_x} vs {feature_y}"
        color_scale = "Blues"
    elif heatmap_view == "Expected Frequencies (E_ij)":
        heat_matrix = expected_df.values
        heat_text = np.round(heat_matrix, 2).astype(str)
        heat_title = f"Expected Counts: {feature_x} vs {feature_y}"
        color_scale = "Viridis"
    else:
        heat_matrix = residuals_df.values
        heat_text = np.round(heat_matrix, 2).astype(str)
        heat_title = f"Standardized Residuals: {feature_x} vs {feature_y}"
        color_scale = "RdBu_r"
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_matrix,
        x=list(observed_crosstab.columns),
        y=list(observed_crosstab.index),
        text=heat_text,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale=color_scale,
        colorbar=dict(title="Value")
    ))
    
    fig_heat.update_layout(
        title=heat_title,
        xaxis_title=str(feature_y),
        yaxis_title=str(feature_x),
        template="plotly_white",
        margin=dict(l=30, r=30, t=40, b=30)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# SECTION 7: DOWNLOADABLE ANALYSIS REPORT & SUMMARY
# ==============================================================================

st.divider()
st.markdown('<div class="section-header"><span class="section-num">7</span> Downloadable Statistical Analysis Report</div>', unsafe_allow_html=True)

full_report_text = generate_text_report(
    project_title="Feature Independence Validation Using Chi-Square Test",
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
    min_expected=min_expected,
    alt_test_notes=alt_report_notes
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
    st.write("**Export Options:**")
    st.caption("Export the complete statistical report or raw tables for documentation and lab submissions:")
    
    st.download_button(
        label="Download Full Report (.txt)",
        data=full_report_text.encode('utf-8'),
        file_name=f"ChiSquare_Report_{feature_x}_vs_{feature_y}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.download_button(
        label="Download Contingency Table (.csv)",
        data=observed_with_margins.to_csv().encode('utf-8'),
        file_name=f"Contingency_Table_{feature_x}_vs_{feature_y}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.download_button(
        label="Download Expected Frequencies (.csv)",
        data=expected_df.round(4).to_csv().encode('utf-8'),
        file_name=f"Expected_Frequencies_{feature_x}_vs_{feature_y}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 0.5rem 0 1.5rem 0;">
    <strong>Feature Independence Validation Using Chi-Square Test</strong><br>
    B.Tech CSE Mathematics & Statistics Micro-Project | Python, Streamlit, Pandas, SciPy & Plotly
</div>
""", unsafe_allow_html=True)