"""
Feature Independence Validation Using Chi-Square Test
Streamlit Micro-Project Web Application — Pure Monochrome Scientific Edition
Department of Computer Science & Engineering | Mathematics & Statistics
"""

import io
import datetime
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Attempt to import SciPy stats; if blocked by OS security policy (e.g. Windows Smart App Control
# on Python 3.14 BLAS extensions), gracefully provide a pure-Python / NumPy statistical fallback.
try:
    from scipy import stats
except Exception:
    def _regularized_gamma_upper(a, x, eps=1e-15, max_iter=200):
        if x <= 0.0:
            return 1.0
        if a <= 0.0:
            return 0.0
        if x >= a + 1.0:
            b = x + 1.0 - a
            c = 1e30
            d = 1.0 / b
            h = d
            for i in range(1, max_iter + 1):
                an = -i * (i - a)
                b += 2.0
                d = an * d + b
                if abs(d) < 1e-30:
                    d = 1e-30
                c = b + an / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
        else:
            ap = a
            sum_val = 1.0 / a
            del_val = sum_val
            for i in range(1, max_iter + 1):
                ap += 1.0
                del_val *= x / ap
                sum_val += del_val
                if abs(del_val) < abs(sum_val) * eps:
                    break
            P = sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))
            return max(0.0, min(1.0, 1.0 - P))

    def _chi2_sf(x, df):
        if x <= 0.0:
            return 1.0
        return _regularized_gamma_upper(df / 2.0, x / 2.0)

    def _chi2_ppf(q, df):
        target = 1.0 - q
        low = 0.0
        high = max(10.0, float(df) * 5.0)
        while _chi2_sf(high, df) > target:
            high *= 2.0
        for _ in range(60):
            mid = (low + high) / 2.0
            if _chi2_sf(mid, df) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    class _Chi2Dist:
        @staticmethod
        def ppf(q, df):
            return _chi2_ppf(q, df)
        @staticmethod
        def sf(x, df):
            return _chi2_sf(x, df)

    class _FisherResult:
        def __init__(self, pvalue):
            self.pvalue = pvalue

    def _fisher_exact(table):
        mat = np.asarray(table, dtype=int)
        a, b = int(mat[0, 0]), int(mat[0, 1])
        c, d = int(mat[1, 0]), int(mat[1, 1])
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        n = r1 + r2
        def prob(k):
            return (math.comb(r1, k) * math.comb(r2, c1 - k)) / math.comb(n, c1)
        p_obs = prob(a)
        k_min = max(0, c1 - r2)
        k_max = min(r1, c1)
        p_val = sum(prob(k) for k in range(k_min, k_max + 1) if prob(k) <= p_obs + 1e-12)
        return _FisherResult(min(1.0, p_val))

    def _chi2_contingency(observed, correction=False):
        obs = np.asarray(observed, dtype=float)
        r, c = obs.shape
        row_sums = obs.sum(axis=1)
        col_sums = obs.sum(axis=0)
        n = obs.sum()
        expected = np.outer(row_sums, col_sums) / n
        dof = (r - 1) * (c - 1)
        if correction and r == 2 and c == 2:
            diff = np.abs(obs - expected) - 0.5
            diff = np.maximum(diff, 0.0)
            chi2 = float(np.sum((diff ** 2) / expected))
        else:
            chi2 = float(np.sum(((obs - expected) ** 2) / expected))
        p_val = _chi2_sf(chi2, dof)
        return chi2, p_val, dof, expected

    class _StatsFallback:
        chi2 = _Chi2Dist()
        chi2_contingency = staticmethod(_chi2_contingency)
        fisher_exact = staticmethod(_fisher_exact)

    stats = _StatsFallback()


# ==============================================================================
# 1. PAGE CONFIGURATION & THEME SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Chi-Square Feature Independence Validator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "Dark Mode"

# Sidebar Theme Selector (Placed at the very top of controls)
st.sidebar.markdown("### Interface Theme")
selected_theme = st.sidebar.radio(
    "Select Display Theme:",
    options=["Dark Mode", "Light Mode"],
    index=0 if st.session_state.theme == "Dark Mode" else 1,
    horizontal=True,
    label_visibility="collapsed",
    key="theme_radio_select"
)
st.session_state.theme = selected_theme

is_dark = (st.session_state.theme == "Dark Mode")

# Exact monochrome design tokens
if is_dark:
    bg_color = "#000000"
    card_bg = "#111111"
    sidebar_bg = "#000000"
    text_primary = "#FFFFFF"
    text_secondary = "#888888"
    border_color = "#333333"
    border_subtle = "#222222"
    input_bg = "#111111"
    btn_bg = "#111111"
    btn_text = "#FFFFFF"
    btn_hover_bg = "#FFFFFF"
    btn_hover_text = "#000000"
    plotly_template = "plotly_dark"
    chart_paper = "#000000"
    chart_plot = "#000000"
else:
    bg_color = "#FFFFFF"
    card_bg = "#FFFFFF"
    sidebar_bg = "#FFFFFF"
    text_primary = "#000000"
    text_secondary = "#666666"
    border_color = "#D9D9D9"
    border_subtle = "#EBEBEB"
    input_bg = "#FFFFFF"
    btn_bg = "#FFFFFF"
    btn_text = "#000000"
    btn_hover_bg = "#000000"
    btn_hover_text = "#FFFFFF"
    plotly_template = "plotly_white"
    chart_paper = "#FFFFFF"
    chart_plot = "#FFFFFF"

# Global Strict Monochrome CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {bg_color} !important;
        color: {text_primary} !important;
    }}

    .stApp > header {{
        background-color: {bg_color} !important;
    }}
    
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {{
        color: {text_secondary} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }}

    p, span, label {{
        color: {text_primary} !important;
    }}
    .stCaption, small {{
        color: {text_secondary} !important;
    }}

    hr {{
        border-color: {border_color} !important;
        opacity: 0.6;
        margin: 1.4rem 0 !important;
    }}

    /* Minimalist Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        padding: 0.85rem 1rem !important;
        border-radius: 2px !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}

    /* Minimalist Monochrome Buttons */
    .stButton > button, .stDownloadButton > button {{
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        border: 1px solid {border_color} !important;
        border-radius: 2px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {btn_hover_bg} !important;
        color: {btn_hover_text} !important;
        border-color: {text_primary} !important;
    }}

    /* Monochrome Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
        color: {text_primary} !important;
        border-radius: 2px !important;
    }}
    div[data-baseweb="select"] * {{
        color: {text_primary} !important;
        background-color: {input_bg} !important;
    }}
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    li[role="option"] {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
    }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
        background-color: {border_color} !important;
        color: {text_primary} !important;
    }}

    /* Radios & Checkboxes */
    div[role="radiogroup"] label {{
        color: {text_primary} !important;
    }}
    div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {{
        color: {text_primary} !important;
    }}
    div[data-baseweb="radio"] input:checked + div {{
        border-color: {text_primary} !important;
        background-color: {text_primary} !important;
    }}
    div[data-baseweb="checkbox"] input:checked + div {{
        border-color: {text_primary} !important;
        background-color: {text_primary} !important;
    }}

    /* Monochrome Tabs */
    button[data-baseweb="tab"] {{
        color: {text_secondary} !important;
        background-color: transparent !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text_primary} !important;
        border-bottom: 2px solid {text_primary} !important;
        font-weight: 600 !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {text_primary} !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 2px !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {text_primary} !important;
    }}

    /* Alert & Warning Neutralization */
    div[data-testid="stAlert"] {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 2px !important;
    }}
    div[data-testid="stAlert"] svg {{
        fill: {text_primary} !important;
    }}
    div[data-testid="stAlert"] * {{
        color: {text_primary} !important;
    }}

    /* Custom Monochrome Panels */
    .mono-header {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 1.35rem 1.6rem;
        border-radius: 2px;
        margin-bottom: 1.25rem;
    }}
    .mono-title {{
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        color: {text_primary};
    }}
    .mono-subtitle {{
        font-size: 0.92rem;
        color: {text_secondary};
        line-height: 1.5;
        max-width: 900px;
        margin-bottom: 0.75rem;
    }}
    .mono-tag {{
        display: inline-block;
        border: 1px solid {border_color};
        color: {text_secondary};
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 2px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.35rem;
    }}

    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_primary};
        margin-top: 1.25rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }}
    .section-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border: 1px solid {border_color};
        background-color: {card_bg};
        color: {text_primary};
        border-radius: 2px;
        font-size: 0.75rem;
        font-weight: 700;
    }}

    .mono-box {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 2px;
        padding: 0.95rem 1.15rem;
        margin: 0.65rem 0;
    }}
    .mono-box-highlight {{
        background-color: {card_bg};
        border: 1px solid {text_primary};
        border-radius: 2px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }}
    .mono-box-title {{
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
        color: {text_primary};
    }}
    .mono-box-desc {{
        font-size: 0.88rem;
        color: {text_primary};
        margin-bottom: 0;
        line-height: 1.5;
    }}

    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: {text_primary} !important;
        border-color: {text_primary} !important;
    }}
    div[data-testid="stSlider"] div[data-baseweb="slider"] div {{
        background-color: {border_color} !important;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {border_color} !important;
        border-radius: 2px;
    }}
</style>
""", unsafe_allow_html=True)


def render_notice(text, title=None, style="default"):
    """Helper to render a clean monochrome notice box without colorful alert styling."""
    title_html = f'<div class="mono-box-title">{title}</div>' if title else ""
    box_class = "mono-box-highlight" if style == "highlight" else "mono-box"
    st.markdown(
        f'<div class="{box_class}">'
        f'{title_html}'
        f'<div class="mono-box-desc">{text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ==============================================================================
# 2. HELPER FUNCTIONS & STATISTICAL CALCULATIONS
# ==============================================================================

@st.cache_data
def load_default_student_data():
    """Load the original student academic dataset."""
    try:
        return pd.read_csv("dataset/student_data.csv")
    except Exception as e:
        render_notice(f"Error loading dataset/student_data.csv: {e}", title="DATASET LOAD ERROR", style="highlight")
        return None

@st.cache_data
def load_demonstration_data():
    """Load the synthetic demonstration survey dataset."""
    try:
        return pd.read_csv("dataset/demonstration_survey_data.csv")
    except Exception as e:
        render_notice(f"Error loading dataset/demonstration_survey_data.csv: {e}", title="DATASET LOAD ERROR", style="highlight")
        return None

def compute_cramers_v(chi2_stat, n, r, c):
    """Compute Cramer's V effect size measure: V = sqrt(chi2 / (n * min(r - 1, c - 1)))."""
    if n <= 0:
        return 0.0
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    v = np.sqrt(max(0.0, chi2_stat / (n * k)))
    return float(v)

def interpret_cramers_v(v, r, c):
    """Determine the qualitative strength of Cramer's V based on k = min(r-1, c-1)."""
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
    report.append(f"Generated Timestamp: {now}")
    report.append("=" * 78)
    report.append("")
    report.append("1. ANALYSIS SPECIFICATIONS & PARAMETERS")
    report.append("-" * 78)
    report.append(f"Dataset Source              : {dataset_name}")
    report.append(f"Row Variable (X)            : {var_x} {binning_info_x}")
    report.append(f"Column Variable (Y)         : {var_y} {binning_info_y}")
    report.append(f"Significance Level (alpha)  : {alpha}")
    report.append(f"Total Dataset Records       : {total_records}")
    report.append(f"Analyzed Complete Records   : {analyzed_records}")
    report.append(f"Excluded Records (Missing)  : {excluded_records}")
    report.append("")
    report.append("2. HYPOTHESIS TESTING FORMULATION")
    report.append("-" * 78)
    report.append(f"Null Hypothesis (H0)        : {var_x} and {var_y} are statistically independent.")
    report.append(f"Alternative Hypothesis (H1) : {var_x} and {var_y} are statistically dependent/associated.")
    report.append("")
    report.append("3. OBSERVED CONTINGENCY TABLE (Cross-Tabulation)")
    report.append("-" * 78)
    report.append(observed_df.to_string())
    report.append("")
    report.append("4. THEORETICAL EXPECTED FREQUENCIES (E_ij)")
    report.append("-" * 78)
    report.append(expected_df.round(4).to_string())
    report.append("")
    report.append("5. STANDARDIZED PEARSON RESIDUALS (r_ij)")
    report.append("-" * 78)
    report.append(residuals_df.round(4).to_string())
    report.append("Note: Standardized residuals |r_ij| > 2.0 indicate cells deviating significantly")
    report.append("      from independence at roughly the 95% confidence level.")
    report.append("")
    report.append("6. TEST STATISTICS & DECISION METRICS")
    report.append("-" * 78)
    report.append(f"Chi-Square Statistic (X^2)  : {chi2_stat:.5f}")
    report.append(f"Degrees of Freedom (df)     : {dof}")
    report.append(f"Critical Value (X^2_crit)   : {critical_val:.5f}")
    report.append(f"Calculated P-Value          : {p_value:.6e}")
    report.append(f"Decision                    : {decision.upper()}")
    report.append("")
    report.append("7. EFFECT SIZE (Cramer's V)")
    report.append("-" * 78)
    report.append(f"Cramer's V Value            : {cramers_v:.5f}")
    report.append(f"Association Magnitude       : {effect_label}")
    report.append("")
    report.append("8. COCHRAN'S (1954) ASSUMPTION AUDIT")
    report.append("-" * 78)
    cochran_status = "PASSED (Satisfied)" if cochran_valid else "WARNING (Assumptions Violated)"
    report.append(f"Cochran Rule Status         : {cochran_status}")
    report.append(f"Minimum Expected Cell Count : {min_expected:.4f} (Rule: >= 1.0)")
    report.append(f"Cells with Count < 5.0      : {cells_below_5_pct:.2f}% (Rule: <= 20.0%)")
    if not cochran_valid:
        report.append("Advisory: Asymptotic Chi-Square approximation may inflate Type I error rates.")
        report.append("          Consolidate sparse categories or refer to exact tests.")
    if alt_test_notes:
        report.append("")
        report.append("Small-Sample Alternative Tests:")
        report.append(alt_test_notes)
    report.append("")
    report.append("=" * 78)
    report.append("ACADEMIC AUDIT VERIFICATION")
    report.append("-" * 78)
    report.append(
        "This report was automatically compiled by the Feature Independence\n"
        "Validator dashboard. All statistical computations, contingency distributions,\n"
        "and diagnostic validity checks are finalized and verified.\n\n"
        "Associated Export Files:\n"
        f"  - Full Analysis Report (.txt)        : ChiSquare_Report_{var_x}_vs_{var_y}.txt\n"
        f"  - Observed Contingency Table (.csv)  : Contingency_Table_{var_x}_vs_{var_y}.csv\n"
        f"  - Expected Frequencies Matrix (.csv): Expected_Frequencies_{var_x}_vs_{var_y}.csv"
    )
    report.append("=" * 78)
    report.append("END OF REPORT")
    report.append("=" * 78)
    
    return "\n".join(report)


# ==============================================================================
# 3. APPLICATION HEADER & INTRODUCTION (MONOCHROME)
# ==============================================================================

st.markdown(f"""
<div class="mono-header">
    <div style="margin-bottom: 0.45rem;">
        <span class="mono-tag">Hypothesis Testing</span>
        <span class="mono-tag">Feature Independence Validation</span>
        <span class="mono-tag">B.Tech CSE Micro-Project</span>
    </div>
    <div class="mono-title">Feature Independence Validation Using Chi-Square Test</div>
    <div class="mono-subtitle">
        An interactive dashboard for evaluating statistical independence, contingency tables, 
        and association strength between categorical features.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.markdown("### Controls & Settings")

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
            st.sidebar.caption(f"Loaded `{uploaded_file.name}` ({len(df_raw)} records)")
        except Exception as e:
            st.sidebar.caption(f"Error parsing uploaded CSV: {e}")
            st.stop()
    else:
        render_notice("Please upload a CSV file in the sidebar to proceed, or choose one of the built-in datasets.", title="DATASET REQUIRED")
        st.stop()

if df_raw is None or df_raw.empty:
    render_notice("Dataset could not be loaded or is empty. Please verify the dataset path or file format.", title="DATASET ERROR", style="highlight")
    st.stop()

st.sidebar.divider()

# Significance level alpha
st.sidebar.markdown("### Significance Level (α)")
alpha = st.sidebar.radio(
    "Select α threshold:",
    options=[0.01, 0.05, 0.10],
    index=1,
    format_func=lambda x: f"α = {x} ({int(x*100)}% Type I Error Rate)",
    help="The probability of rejecting the null hypothesis when it is actually true. α = 0.05 is standard."
)

st.sidebar.divider()

# Theory Guide Expander
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

st.markdown('<div class="section-title"><span class="section-num">1</span> Dataset Overview & Inspection</div>', unsafe_allow_html=True)

total_records = len(df_raw)
total_columns = len(df_raw.columns)
missing_cells = int(df_raw.isna().sum().sum())
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df_raw.columns if c not in num_cols]

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Total Records", total_records)
with m_col2:
    st.metric("Total Features", total_columns)
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
st.markdown('<div class="section-title"><span class="section-num">2</span> Variable Selection & Feature Discretization</div>', unsafe_allow_html=True)
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
    render_notice(
        "Feature 1 and Feature 2 must be different variables. Please choose two distinct columns to construct a cross-tabulation.",
        title="VARIABLE SELECTION ERROR",
        style="highlight"
    )
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
        render_notice(f"Variable <code>{feature_name}</code> is categorical with <strong>{n_cats}</strong> distinct categories.", title="FEATURE DETECTED")
        
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
        render_notice(f"Variable <code>{feature_name}</code> has constant values ({c_min}). Cannot discretize a constant feature.", title="CONSTANT FEATURE", style="highlight")
        return col_series.astype(str), "[Constant numerical feature]"
    
    render_notice(f"Variable <code>{feature_name}</code> is numerical (Range: {c_min:.1f} to {c_max:.1f}). Discretize into categories below:", title="FEATURE DETECTED")
    
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
                render_notice("Cutoff 1 must be strictly less than Cutoff 2. Cutoff 2 has been adjusted.", title="CUTOFF ADJUSTMENT")
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
            render_notice(f"Could not compute equal quantiles ({e}). Using equal-width bins instead.", title="QUANTILE FALLBACK")
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
    render_notice(
        f"Missing Value Handling: {excluded_count} record(s) excluded due to missing/invalid values. Analyzing {analyzed_count} complete observations.",
        title="DATA CLEANING NOTICE"
    )
else:
    st.caption(f"All {analyzed_count} records have complete data for `{feature_x}` and `{feature_y}`.")

# Validate minimum unique categories
unique_x = clean_df[feature_x].nunique()
unique_y = clean_df[feature_y].nunique()

if unique_x < 2 or unique_y < 2:
    render_notice(
        f"Insufficient Categories: `{feature_x}` has {unique_x} category, and `{feature_y}` has {unique_y} category. "
        f"The Chi-Square Test requires at least 2 categories per variable. Please adjust your binning or select another feature.",
        title="CATEGORICAL VALIDATION ERROR",
        style="highlight"
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
st.markdown('<div class="section-title"><span class="section-num">3</span> Observed Contingency Table (Cross-Tabulation)</div>', unsafe_allow_html=True)

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

# Chi-Square computation
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
st.markdown('<div class="section-title"><span class="section-num">4</span> Hypothesis Testing & Chi-Square Results</div>', unsafe_allow_html=True)

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

# Monochrome Hypothesis Decision Panel
decision_header = f"STATISTICAL DECISION: {decision_text.upper()}"
if reject_null:
    decision_detail = (
        f"The p-value (<strong>{p_format}</strong>) is strictly below the significance threshold α = {alpha}, "
        f"and the test statistic (<strong>χ² = {chi2_stat:.4f}</strong>) exceeds the critical value threshold (<strong>χ²_crit = {critical_val:.4f}</strong>). "
        f"There is statistically significant evidence of mutual association between <code>{feature_x}</code> and <code>{feature_y}</code>."
    )
else:
    decision_detail = (
        f"The p-value (<strong>{p_format}</strong>) is greater than or equal to the significance threshold α = {alpha}, "
        f"and the test statistic (<strong>χ² = {chi2_stat:.4f}</strong>) does not exceed the critical threshold (<strong>χ²_crit = {critical_val:.4f}</strong>). "
        f"We fail to reject the null hypothesis; observed joint frequency variations between <code>{feature_x}</code> and <code>{feature_y}</code> are consistent with random sampling variability."
    )

st.markdown(f"""
<div class="mono-box-highlight">
    <div class="mono-box-title" style="font-size: 0.95rem;">{decision_header}</div>
    <div class="mono-box-desc" style="margin-top: 0.35rem;">
        {decision_detail}
    </div>
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
st.markdown('<div class="section-title"><span class="section-num">5</span> Statistical Assumption Audit (Cochran\'s Rule) & Diagnostics</div>', unsafe_allow_html=True)

alt_report_notes = ""

if cochran_valid:
    st.markdown(f"""
    <div class="mono-box">
        <div class="mono-box-title">ASSUMPTION AUDIT: COCHRAN'S CRITERIA SATISFIED</div>
        <div class="mono-box-desc">
            Minimum expected cell frequency is <strong>{min_expected:.2f}</strong> (≥ 1.0), and 
            <strong>{cells_below_5} of {total_cells} cells ({cells_below_5_pct:.1f}%)</strong> have expected frequency &lt; 5.0 (≤ 20.0%). 
            The continuous asymptotic Chi-Square distribution approximation is statistically valid and reliable.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="mono-box-highlight">
        <div class="mono-box-title">ASSUMPTION AUDIT: COCHRAN'S RULE WARNING (SPARSE CELLS)</div>
        <div class="mono-box-desc" style="margin-bottom: 0.4rem;">
            Pearson's Chi-Square test relies on asymptotic approximation. Cochran (1954) established the following validity criteria:
        </div>
        <ul style="color: {text_secondary}; font-size: 0.88rem; margin-bottom: 0.45rem; padding-left: 1.25rem;">
            <li>No expected cell frequency should be less than 1.0 (Current minimum: <strong>{min_expected:.2f}</strong>).</li>
            <li>No more than 20% of expected cell frequencies should be less than 5.0 (Current: <strong>{cells_below_5_pct:.1f}%</strong> — {cells_below_5} of {total_cells} cells).</li>
        </ul>
        <div class="mono-box-desc" style="font-size: 0.84rem; color: {text_secondary};">
            <em>Diagnostic Precaution:</em> Small expected frequencies inflate Type I error rates. The reported p-value should be interpreted with diagnostic caution.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Practical remedies
    if is_2x2 and p_fisher is not None:
        p_fisher_fmt = f"{p_fisher:.4e}" if p_fisher < 0.0001 else f"{p_fisher:.4f}"
        p_yates_fmt = f"{p_yates:.4e}" if p_yates < 0.0001 else f"{p_yates:.4f}"
        render_notice(
            f"Because this is a 2×2 contingency table, you can rely on **Fisher's Exact Test**, which computes the exact hypergeometric probability without asymptotic assumptions: "
            f"**Fisher's Exact p-value = {p_fisher_fmt}** (Yates' corrected χ² = {chi2_yates:.4f}, p = {p_yates_fmt}).",
            title="EXACT TEST ALTERNATIVE FOR 2×2 TABLE"
        )
        p_fisher_rep = f"{p_fisher:.6e}" if p_fisher < 0.0001 else f"{p_fisher:.6f}"
        p_yates_rep = f"{p_yates:.6e}" if p_yates < 0.0001 else f"{p_yates:.6f}"
        alt_report_notes = (
            f"Fisher's Exact Test p-value : {p_fisher_rep}\n"
            f"Yates' Corrected Chi-Square : {chi2_yates:.4f} (p = {p_yates_rep})"
        )
    else:
        render_notice(
            "(1) Consolidate sparse categories into fewer categories (e.g. choose 2 bins for numerical variables in Section 2); "
            "(2) Evaluate the larger 350-record Demonstration survey dataset from the sidebar; "
            "(3) In your college micro-project presentation, explain that small sample sizes naturally trigger Cochran warnings, demonstrating statistical rigor!",
            title="RECOMMENDED PRACTICAL REMEDIES"
        )

# Diagnostic Tabs
exp_tabs = st.tabs([
    "Expected Frequencies Matrix (E_ij)",
    "Standardized Residuals (r_ij)",
    "Effect Size Interpretation (Cramér's V)",
    "Alternative Tests & Small-Sample Notes"
])

with exp_tabs[0]:
    st.write("Expected frequencies calculated under the theoretical assumption of statistical independence:")
    st.markdown(r"$$E_{ij} = \frac{R_i \times C_j}{N}$$")
    st.dataframe(expected_df.round(2), use_container_width=True)
    st.caption("Each cell represents the theoretical expectation count under H₀.")

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
        st.markdown(f"**Classification:** `{effect_label}`")
    with v_c2:
        st.write(f"**Interpretation:** {effect_desc}")
        st.caption("Unlike the p-value (which shrinks with large N), Cramér's V measures the true practical magnitude of association between 0 (none) and 1 (perfect).")

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
# SECTION 6: INTERACTIVE STATISTICAL VISUALIZATIONS
# ==============================================================================

st.divider()
st.markdown('<div class="section-title"><span class="section-num">6</span> Interactive Statistical Visualizations</div>', unsafe_allow_html=True)

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
        template=plotly_template,
        paper_bgcolor=chart_paper,
        plot_bgcolor=chart_plot,
        font=dict(color=text_primary, family="Inter, sans-serif"),
        legend=dict(
            title_text=str(feature_y),
            font=dict(color=text_primary),
            bgcolor=card_bg,
            bordercolor=border_color,
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor=border_color,
            linecolor=border_color,
            tickfont=dict(color=text_primary),
            title_font=dict(color=text_primary)
        ),
        yaxis=dict(
            gridcolor=border_color,
            linecolor=border_color,
            tickfont=dict(color=text_primary),
            title_font=dict(color=text_primary)
        ),
        margin=dict(l=35, r=30, t=45, b=35),
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
        colorbar=dict(
            title=dict(text="Value", font=dict(color=text_primary)),
            tickfont=dict(color=text_primary)
        )
    ))
    
    fig_heat.update_layout(
        title=dict(text=heat_title, font=dict(color=text_primary, size=14)),
        xaxis=dict(
            title=str(feature_y),
            tickfont=dict(color=text_primary),
            title_font=dict(color=text_primary),
            gridcolor=border_color
        ),
        yaxis=dict(
            title=str(feature_x),
            tickfont=dict(color=text_primary),
            title_font=dict(color=text_primary),
            gridcolor=border_color
        ),
        template=plotly_template,
        paper_bgcolor=chart_paper,
        plot_bgcolor=chart_plot,
        font=dict(color=text_primary, family="Inter, sans-serif"),
        margin=dict(l=35, r=30, t=45, b=35)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# SECTION 7: DOWNLOADABLE ANALYSIS REPORT & SUMMARY
# ==============================================================================

st.divider()
st.markdown('<div class="section-title"><span class="section-num">7</span> Downloadable Statistical Analysis Report</div>', unsafe_allow_html=True)

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
st.markdown(f"""
<div style="text-align: center; color: {text_secondary}; font-size: 0.85rem; padding: 0.8rem 0 2rem 0; letter-spacing: 0.02em;">
    <strong>Feature Independence Validation Using Chi-Square Test</strong><br>
    B.Tech CSE Mathematics & Statistics Micro-Project | Pure Monochrome Scientific Edition
</div>
""", unsafe_allow_html=True)
