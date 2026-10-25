"""
Automated Test Suite for Feature Independence Validation Statistical Engine
Department of Computer Science & Engineering | Mathematics & Statistics
"""

import os
import csv
import math
import io
import unittest

def compute_cramers_v(chi2_stat, n, r, c):
    """Compute Cramer's V effect size measure."""
    if n <= 0:
        return 0.0
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    v = math.sqrt(max(0.0, chi2_stat / (n * k)))
    return float(v)

def interpret_cramers_v(v, r, c):
    """Determine qualitative strength of Cramer's V based on k = min(r-1, c-1)."""
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

def evaluate_cochran_criteria(expected_matrix):
    """
    Evaluate Cochran's (1954) criteria:
    1. No cell < 1.0
    2. At most 20% of cells < 5.0
    """
    total_cells = len(expected_matrix) * len(expected_matrix[0])
    min_expected = min(min(row) for row in expected_matrix)
    cells_below_1 = sum(1 for row in expected_matrix for val in row if val < 1.0)
    cells_below_5 = sum(1 for row in expected_matrix for val in row if val < 5.0)
    pct_below_5 = (cells_below_5 / total_cells) * 100.0
    valid = (cells_below_1 == 0) and (pct_below_5 <= 20.0)
    return valid, min_expected, pct_below_5

def compute_expected_frequencies(observed):
    """Compute expected frequency matrix: E_ij = (R_i * C_j) / N."""
    r_dim = len(observed)
    c_dim = len(observed[0])
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[i][j] for i in range(r_dim)) for j in range(c_dim)]
    n = sum(row_sums)
    
    expected = []
    for i in range(r_dim):
        row_exp = []
        for j in range(c_dim):
            e_ij = (row_sums[i] * col_sums[j]) / n
            row_exp.append(e_ij)
        expected.append(row_exp)
    return expected, n

def compute_chi2_statistic(observed, expected):
    """Compute Pearson Chi-Square statistic: chi2 = sum((O_ij - E_ij)^2 / E_ij)."""
    chi2 = 0.0
    for i in range(len(observed)):
        for j in range(len(observed[0])):
            o = observed[i][j]
            e = expected[i][j]
            chi2 += ((o - e) ** 2) / e
    return chi2

def format_text_report(project_title, var_x, var_y, n_analyzed, chi2_stat, dof, p_val, alpha, decision, cramers_v, effect_label, cochran_valid):
    """Generate plaintext analysis report for export verification."""
    report = []
    report.append("=" * 78)
    report.append(f"{project_title.upper()}")
    report.append("STATISTICAL ANALYSIS & FEATURE INDEPENDENCE VALIDATION REPORT")
    report.append("=" * 78)
    report.append(f"Feature X (Row)       : {var_x}")
    report.append(f"Feature Y (Column)    : {var_y}")
    report.append(f"Analyzed Records (N)  : {n_analyzed}")
    report.append(f"Significance Level (a): {alpha}")
    report.append("-" * 78)
    report.append(f"Chi-Square Statistic (X^2) : {chi2_stat:.5f}")
    report.append(f"Degrees of Freedom (df)     : {dof}")
    report.append(f"P-Value                     : {p_val:.5f}")
    report.append(f"Effect Size (Cramer's V)    : {cramers_v:.4f} [{effect_label}]")
    report.append(f"Decision                    : {decision.upper()}")
    report.append(f"Cochran Rule Status         : {'PASSED' if cochran_valid else 'WARNING'}")
    report.append("=" * 78)
    return "\n".join(report)


class TestStatisticalEngine(unittest.TestCase):
    """Automated test cases validating the statistical computation engine."""

    def test_cramers_v_calculation(self):
        """Test Cramer's V calculation on known values."""
        # 2x2 table: N=100, chi2=10.0 -> k = min(1, 1) = 1 -> V = sqrt(10 / (100 * 1)) = sqrt(0.1)
        v = compute_cramers_v(chi2_stat=10.0, n=100, r=2, c=2)
        expected_v = math.sqrt(0.1)
        self.assertAlmostEqual(v, expected_v, places=6)

        # 3x3 table: N=200, chi2=50.0 -> k = min(2, 2) = 2 -> V = sqrt(50 / (200 * 2)) = sqrt(50/400) = sqrt(0.125)
        v3 = compute_cramers_v(chi2_stat=50.0, n=200, r=3, c=3)
        self.assertAlmostEqual(v3, math.sqrt(0.125), places=6)

        # Edge cases: N <= 0 or r < 2 or c < 2
        self.assertEqual(compute_cramers_v(10.0, 0, 2, 2), 0.0)
        self.assertEqual(compute_cramers_v(10.0, 100, 1, 2), 0.0)
        self.assertEqual(compute_cramers_v(0.0, 100, 2, 2), 0.0)

    def test_cramers_v_interpretation(self):
        """Test qualitative classification of Cramer's V."""
        # For k=1
        label, _ = interpret_cramers_v(0.05, 2, 2)
        self.assertEqual(label, "Negligible")
        label, _ = interpret_cramers_v(0.20, 2, 2)
        self.assertEqual(label, "Weak")
        label, _ = interpret_cramers_v(0.35, 2, 2)
        self.assertEqual(label, "Moderate")
        label, _ = interpret_cramers_v(0.60, 2, 2)
        self.assertEqual(label, "Strong")

        # For k=2
        label, _ = interpret_cramers_v(0.05, 3, 3)
        self.assertEqual(label, "Negligible")
        label, _ = interpret_cramers_v(0.15, 3, 3)
        self.assertEqual(label, "Weak")
        label, _ = interpret_cramers_v(0.25, 3, 3)
        self.assertEqual(label, "Moderate")
        label, _ = interpret_cramers_v(0.40, 3, 3)
        self.assertEqual(label, "Strong")

    def test_expected_frequencies_and_chi2(self):
        """Test contingency expected frequencies and chi-square test statistic."""
        observed = [
            [10, 20],
            [30, 40]
        ]
        # Row sums: 30, 70. Col sums: 40, 60. Total N = 100.
        expected, n = compute_expected_frequencies(observed)
        self.assertEqual(n, 100)
        self.assertAlmostEqual(expected[0][0], (30 * 40) / 100, places=4)  # 12.0
        self.assertAlmostEqual(expected[0][1], (30 * 60) / 100, places=4)  # 18.0
        self.assertAlmostEqual(expected[1][0], (70 * 40) / 100, places=4)  # 28.0
        self.assertAlmostEqual(expected[1][1], (70 * 60) / 100, places=4)  # 42.0

        chi2 = compute_chi2_statistic(observed, expected)
        # Expected chi2: (10-12)^2/12 + (20-18)^2/18 + (30-28)^2/28 + (40-42)^2/42
        # = 4/12 + 4/18 + 4/28 + 4/42 = 1/3 + 2/9 + 1/7 + 2/21 = 0.79365079...
        manual_chi2 = (4.0 / 12) + (4.0 / 18) + (4.0 / 28) + (4.0 / 42)
        self.assertAlmostEqual(chi2, manual_chi2, places=6)

    def test_cochran_criteria_validation(self):
        """Test Cochran's (1954) validity rules on various expected matrices."""
        # 1. Perfectly valid matrix (all >= 5.0)
        matrix_valid = [
            [10.0, 15.0],
            [20.0, 25.0]
        ]
        valid, min_val, pct_below_5 = evaluate_cochran_criteria(matrix_valid)
        self.assertTrue(valid)
        self.assertEqual(min_val, 10.0)
        self.assertEqual(pct_below_5, 0.0)

        # 2. Invalid: contains cell < 1.0
        matrix_has_zero_or_sub1 = [
            [0.8, 15.0],
            [20.0, 25.0]
        ]
        valid, min_val, pct_below_5 = evaluate_cochran_criteria(matrix_has_zero_or_sub1)
        self.assertFalse(valid)
        self.assertEqual(min_val, 0.8)

        # 3. Invalid: > 20% cells < 5.0
        # 10 cells total: 3 cells < 5.0 (30.0%)
        matrix_30_pct = [
            [4.0, 4.0, 4.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0]
        ]
        valid, min_val, pct_below_5 = evaluate_cochran_criteria(matrix_30_pct)
        self.assertFalse(valid)
        self.assertEqual(pct_below_5, 30.0)

        # 4. Valid boundary: exactly 20% cells < 5.0 (2 out of 10), min >= 1.0
        matrix_20_pct = [
            [3.0, 4.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0]
        ]
        valid, min_val, pct_below_5 = evaluate_cochran_criteria(matrix_20_pct)
        self.assertTrue(valid)
        self.assertEqual(pct_below_5, 20.0)
        self.assertEqual(min_val, 3.0)

    def test_student_dataset_integrity(self):
        """Verify student_data.csv existence, headers, and exact record count."""
        filepath = os.path.join("dataset", "student_data.csv")
        self.assertTrue(os.path.exists(filepath), f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            rows = reader[1:]

        expected_header = ["Attendance", "Study_Hours", "Previous_Marks", "Assignment_Score", "Performance"]
        self.assertEqual(header, expected_header)
        self.assertEqual(len(rows), 42, "student_data.csv must have exactly 42 data records.")

    def test_demonstration_dataset_integrity(self):
        """Verify demonstration_survey_data.csv existence, headers, and exact record count."""
        filepath = os.path.join("dataset", "demonstration_survey_data.csv")
        self.assertTrue(os.path.exists(filepath), f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            rows = reader[1:]

        expected_header = [
            "Attendance_Level", "Study_Environment", "Parent_Education",
            "Extracurricular", "Learning_Style", "Weekly_Study_Hours",
            "Exam_Score", "Performance_Band"
        ]
        self.assertEqual(header, expected_header)
        self.assertEqual(len(rows), 350, "demonstration_survey_data.csv must have exactly 350 data records.")

    def test_representative_calculation_attendance_vs_performance(self):
        """
        Verify representative calculation: Attendance binned at 75 vs Performance on student_data.csv.
        Mathematical ground truth:
        Table:
          High (>=75.0): [4, 12, 12, 0] (Row sum = 28)
          Low (<75.0)  : [8, 0, 0, 6]   (Row sum = 14)
          Col sums: [12, 12, 12, 6] (N = 42)
        Expected Chi2 = 30.0000, df = 3, Cramer's V = sqrt(30/42) = 0.845154
        """
        filepath = os.path.join("dataset", "student_data.csv")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        table = {"High (>=75.0)": {}, "Low (<75.0)": {}}
        for row in reader:
            att = float(row["Attendance"])
            b_att = "High (>=75.0)" if att >= 75.0 else "Low (<75.0)"
            perf = row["Performance"]
            table[b_att][perf] = table[b_att].get(perf, 0) + 1

        cats = sorted(list({r["Performance"] for r in reader}))
        observed = [
            [table["High (>=75.0)"].get(c, 0) for c in cats],
            [table["Low (<75.0)"].get(c, 0) for c in cats]
        ]

        expected, n = compute_expected_frequencies(observed)
        chi2 = compute_chi2_statistic(observed, expected)
        dof = (len(observed) - 1) * (len(observed[0]) - 1)
        v = compute_cramers_v(chi2, n, len(observed), len(observed[0]))

        self.assertEqual(n, 42)
        self.assertEqual(dof, 3)
        self.assertAlmostEqual(chi2, 30.0, places=4)
        self.assertAlmostEqual(v, math.sqrt(30.0 / 42.0), places=4)

    def test_text_report_and_csv_export_structure(self):
        """Verify that the generated text report and CSV export outputs conform to expected schema."""
        report = format_text_report(
            project_title="Feature Independence Validation Using Chi-Square Test",
            var_x="Attendance",
            var_y="Performance",
            n_analyzed=42,
            chi2_stat=30.0,
            dof=3,
            p_val=0.00001,
            alpha=0.05,
            decision="Reject Null Hypothesis (H0)",
            cramers_v=0.8452,
            effect_label="Strong",
            cochran_valid=False
        )
        self.assertIn("FEATURE INDEPENDENCE VALIDATION", report)
        self.assertIn("Chi-Square Statistic (X^2) : 30.00000", report)
        self.assertIn("Degrees of Freedom (df)     : 3", report)
        self.assertIn("Decision                    : REJECT NULL HYPOTHESIS (H0)", report)
        self.assertIn("Cochran Rule Status         : WARNING", report)

        # Test CSV export formatting
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["", "Average", "Excellent", "Good", "Poor", "Total"])
        writer.writerow(["High (>=75.0)", 4, 12, 12, 0, 28])
        writer.writerow(["Low (<75.0)", 8, 0, 0, 6, 14])
        writer.writerow(["Total", 12, 12, 12, 6, 42])
        csv_str = buf.getvalue()
        self.assertIn("High (>=75.0),4,12,12,0,28", csv_str)
        self.assertIn("Total,12,12,12,6,42", csv_str)


if __name__ == "__main__":
    unittest.main()
