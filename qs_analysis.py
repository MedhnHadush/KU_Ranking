import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def clean_rank_columns(df, rank_columns):
    """
    Removes '=' from all values in the specified rank columns.
    """
    for col in rank_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('=', '', regex=False)
    return df

def analyze_top_150_qs(filepath):
    """
    Reads the given QS file (CSV), extracts the top 150 universities by RANK_2026,
    and computes the average and median of the specified columns.
    Also computes these statistics for universities ranked between 100 and 150 (inclusive),
    and for Khalifa University of Science and Technology. Results are shown as a DataFrame.
    Also prints the difference between Khalifa University and the mean/median of 100-150 for each column.
    Also prints the rank of each indicator for Khalifa University and its overall rank if available.
    """
    df = pd.read_csv(filepath)
    # Clean rank columns
    df = clean_rank_columns(df, ['RANK_2026'])
    
    # Convert RANK_2026 to numeric for sorting and filtering
    df['RANK_2026'] = pd.to_numeric(df['RANK_2026'], errors='coerce')
    
    # Sort by the correct rank column
    if 'RANK_2026' in df.columns:
        df_sorted = df.sort_values('RANK_2026')
    else:
        raise ValueError("Column 'RANK_2026' not found in the file.")

    # Select the top 150 universities (by numeric rank)
    top_150 = df_sorted.head(150)
    between_100_150 = top_150[(top_150['RANK_2026'] >= 100) & (top_150['RANK_2026'] <= 150)]

    # Columns to analyze
    columns = [
        'Overall_Score', 'SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF',
        'SCORE_IF', 'SCORE_IS', 'SCORE_ISD', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST'
    ]
    # Filter to only columns that exist in the file
    columns = [col for col in columns if col in df.columns]

    # Helper to get stats
    def get_stats(subdf):
        stats = {}
        for col in columns:
            vals = pd.to_numeric(subdf[col], errors='coerce')
            stats[col] = {'mean': np.nanmean(vals), 'median': np.nanmedian(vals)}
        return stats

    stats_top_150 = get_stats(top_150)
    stats_100_150 = get_stats(between_100_150)

    # Khalifa University
    ku_row = None
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if not ku_row.empty:
            stats_ku = get_stats(ku_row)
        else:
            stats_ku = {col: {'mean': np.nan, 'median': np.nan} for col in columns}
            ku_row = None
    else:
        stats_ku = {col: {'mean': np.nan, 'median': np.nan} for col in columns}
        ku_row = None

    # Build DataFrame for display
    table = pd.DataFrame({
        'Top 150 Mean': {col: stats_top_150[col]['mean'] for col in columns},
        'Top 150 Median': {col: stats_top_150[col]['median'] for col in columns},
        'Ranks 100-150 Mean': {col: stats_100_150[col]['mean'] for col in columns},
        'Ranks 100-150 Median': {col: stats_100_150[col]['median'] for col in columns},
        'Khalifa Univ.': {col: stats_ku[col]['mean'] for col in columns},
    })
    print(table.round(2))

    # Calculate and print differences
    diff_mean = {}
    diff_median = {}
    for col in columns:
        ku_val = stats_ku[col]['mean']
        mean_100_150 = stats_100_150[col]['mean']
        median_100_150 = stats_100_150[col]['median']
        diff_mean[col] = ku_val - mean_100_150 if not np.isnan(ku_val) and not np.isnan(mean_100_150) else np.nan
        diff_median[col] = ku_val - median_100_150 if not np.isnan(ku_val) and not np.isnan(median_100_150) else np.nan
    diff_table = pd.DataFrame({
        'Khalifa - Mean(100-150)': diff_mean,
        'Khalifa - Median(100-150)': diff_median
    })
    print("\nDifference between Khalifa University and 100-150 Mean/Median:")
    print(diff_table.round(2))

    # Print the rank of each indicator for Khalifa University and its overall rank as a DataFrame
    if ku_row is not None and not ku_row.empty:
        print("\nKhalifa University of Science and Technology Ranks:")
        rank_dict = {}
        for col in columns:
            rank_col = col.replace('SCORE', 'RANK')
            if rank_col in ku_row.columns:
                rank_dict[rank_col] = ku_row.iloc[0][rank_col]
            else:
                rank_dict[rank_col] = 'N/A'
        # Overall rank
        overall_rank = None
        for possible in ['RANK_2026', 'Rank_2026', 'Rank', 'RANK']:
            if possible in ku_row.columns:
                overall_rank = ku_row.iloc[0][possible]
                break
        rank_df = pd.DataFrame({'Khalifa Rank': rank_dict})
        print(rank_df.transpose())
        print(f"Overall Rank: {overall_rank if overall_rank is not None else 'N/A'}")
    else:
        print("Khalifa University of Science and Technology not found for rank display.")

    return table, diff_table

def highest_rank_jump(filepath):
    """
    Calculates the highest positive jump (improvement) between Rank_2025 and Rank_2026.
    Prints the university/universities with the highest jump and the jump value.
    Also prints the average jump across all universities.
    """
    df = pd.read_csv(filepath)
    # Clean rank columns
    df = clean_rank_columns(df, ['RANK_2025', 'RANK_2026'])
    if 'RANK_2025' not in df.columns or 'RANK_2026' not in df.columns:
        raise ValueError("Columns 'RANK_2025' and/or 'RANK_2026' not found in the file.")
    # Convert rank columns to numeric, coercing errors to NaN
    df['RANK_2025'] = pd.to_numeric(df['RANK_2025'], errors='coerce')
    df['RANK_2026'] = pd.to_numeric(df['RANK_2026'], errors='coerce')
    # Drop rows where either rank is missing
    df = df.dropna(subset=['RANK_2025', 'RANK_2026'])
    # Calculate jump (positive means improved rank, i.e., moved up)
    df['Rank_Jump'] = df['RANK_2025'] - df['RANK_2026']
    max_jump = df['Rank_Jump'].max()
    avg_jump = df['Rank_Jump'].mean()
    top_jumpers = df[df['Rank_Jump'] == max_jump]

    print(f"Highest rank jump from 2025 to 2026: {int(max_jump)}")
    print(f"Average rank jump from 2025 to 2026: {avg_jump:.2f}")
    print("University/universities with the highest jump:")
    if 'University' in df.columns:
        for _, row in top_jumpers.iterrows():
            print(f"  {row['University']} (2025: {int(row['RANK_2025'])}, 2026: {int(row['RANK_2026'])})")
    else:
        print(top_jumpers[['RANK_2025', 'RANK_2026']])
    return max_jump, avg_jump, top_jumpers

def compare_khalifa_ranks(file_2025_2026, file_2024_2025):
    """
    Compares the indicator ranks of Khalifa University between two files (2025_2026 and 2024_2025)
    and prints the difference (growth or decrease) for each indicator.
    """
    # Columns to analyze
    columns = [
        'SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF',
        'SCORE_IF', 'SCORE_IS', 'SCORE_ISD', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST'
    ]
    rank_columns = [col.replace('SCORE', 'RANK') for col in columns]

    # Read both files
    df_25 = pd.read_csv(file_2025_2026)
    df_24 = pd.read_csv(file_2024_2025)

    # Find Khalifa University row in both files
    def get_ku_row(df):
        if 'Institution Name' in df.columns:
            return df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        return None

    ku_25 = get_ku_row(df_25)
    ku_24 = get_ku_row(df_24)

    # Build rank dicts for both years
    rank_25 = {}
    rank_24 = {}
    for rank_col in rank_columns:
        rank_25[rank_col] = ku_25.iloc[0][rank_col] if ku_25 is not None and not ku_25.empty and rank_col in ku_25.columns else None
        rank_24[rank_col] = ku_24.iloc[0][rank_col] if ku_24 is not None and not ku_24.empty and rank_col in ku_24.columns else None

    # Compute difference (2025_2026 - 2024_2025)
    diff = {}
    for rank_col in rank_columns:
        try:
            val_25 = float(rank_25[rank_col]) if rank_25[rank_col] is not None else None
            val_24 = float(rank_24[rank_col]) if rank_24[rank_col] is not None else None
            if val_25 is not None and val_24 is not None:
                diff[rank_col] = val_25 - val_24
            else:
                diff[rank_col] = None
        except Exception:
            diff[rank_col] = None

    # Print as DataFrame
    table = pd.DataFrame({
        '2025_2026': rank_25,
        '2024_2025': rank_24,
        'Difference (2025_2026 - 2024_2025)': diff
    })
    print("\nKhalifa University Indicator Ranks Comparison:")
    print(table)
    return table

def create_spider_chart(filepath):
    """
    Creates a beautiful spider chart showing Khalifa University's indicator scores from qs_2025_2026.
    Uses unique colors for low indicators and removes title for cleaner look.
    """
    df = pd.read_csv(filepath)
    
    # Find Khalifa University
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
        
        # Get indicator scores
        score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_ISD', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
        scores = []
        labels = []
        
        for col in score_columns:
            if col in ku_row.columns:
                score = ku_row.iloc[0][col]
                if pd.notna(score):
                    scores.append(float(score))
                    # Create readable labels
                    label = col.replace('SCORE_', '').replace('_', ' ')
                    labels.append(label)
        
        if not scores:
            print("No valid scores found for Khalifa University.")
            return
        
        # Create spider chart
        angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Define color scheme based on score values
        colors = []
        for i, score in enumerate(scores[:-1]):  # Exclude the repeated first value
            if score < 50:
                colors.append('#FF6B6B')  # Red for very low scores
            elif score < 60:
                colors.append('#FFA500')  # Orange for low scores
            elif score < 70:
                colors.append('#FFD93D')  # Yellow for medium-low scores
            elif score < 80:
                colors.append('#6BCF7F')  # Light green for medium scores
            else:
                colors.append('#4ECDC4')  # Teal for high scores
        
        # Plot with custom colors
        ax.plot(angles, scores, 'o-', linewidth=3, label='Khalifa University', color='#2C3E50', alpha=0.8)
        
        # Fill the entire area inside the lines with a solid color
        ax.fill(angles, scores, alpha=0.3, color='#3498DB')
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add grid with custom styling
        ax.grid(True, alpha=0.3, color='#34495E')
        
        # Customize radial lines
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.set_rlabel_position(0)
        ax.tick_params(axis='y', colors='#2C3E50', labelsize=10)
        
        # Add score annotations
        for i, (angle, score) in enumerate(zip(angles[:-1], scores[:-1])):
            if not np.isnan(score):
                # Position text outside the chart
                text_angle = angle
                if text_angle > np.pi/2 and text_angle < 3*np.pi/2:
                    text_angle += np.pi
                
                ax.annotate(f'{score:.1f}', 
                           xy=(angle, score + 5),
                           xytext=(0, 0),
                           textcoords='offset points',
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color=colors[i],
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=colors[i]))
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig('khalifa_spider_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Spider chart saved as 'khalifa_spider_chart.png'")
    else:
        print("Institution Name column not found in the data.")

def create_scatter_plots(filepath):
    """
    Creates scatter plots showing correlation between each indicator score and overall score.
    """
    df = pd.read_csv(filepath)
    
    # Get score columns
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_ISD', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    if 'Overall_Score' not in df.columns:
        print("Overall_Score column not found in the data.")
        return
    
    # Convert scores to numeric
    for col in score_columns + ['Overall_Score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df_clean = df.dropna(subset=score_columns + ['Overall_Score'])
    
    # Create subplots
    n_cols = 3
    n_rows = (len(score_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(score_columns):
        ax = axes[i]
        
        # Calculate correlation
        correlation = df_clean[col].corr(df_clean['Overall_Score'])
        
        # Create scatter plot
        ax.scatter(df_clean[col], df_clean['Overall_Score'], alpha=0.6)
        ax.set_xlabel(col.replace('SCORE_', '').replace('_', ' '))
        ax.set_ylabel('Overall Score')
        ax.set_title(f'Correlation: {correlation:.3f}')
        
        # Add trend line
        z = np.polyfit(df_clean[col], df_clean['Overall_Score'], 1)
        p = np.poly1d(z)
        ax.plot(df_clean[col], p(df_clean[col]), "r--", alpha=0.8)
    
    # Hide empty subplots
    for i in range(len(score_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('indicator_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Scatter plots saved as 'indicator_correlations.png'")

def calculate_indicator_coefficients(filepath):
    """
    Calculates the coefficients for each indicator that best predict the overall score.
    Returns the formula: Overall_Score = a*SCORE_AR + b*SCORE_ER + ...
    (SCORE_ISD is excluded from the calculation.)
    """
    df = pd.read_csv(filepath)
    
    # Get score columns (exclude SCORE_ISD)
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    if 'Overall_Score' not in df.columns:
        print("Overall_Score column not found in the data.")
        return None
    
    # Convert scores to numeric
    for col in score_columns + ['Overall_Score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df_clean = df.dropna(subset=score_columns + ['Overall_Score'])
    
    if len(df_clean) < len(score_columns):
        print("Not enough data points for regression.")
        return None
    
    # Prepare data
    X = df_clean[score_columns]
    y = df_clean['Overall_Score']
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Print results
    print(f"\nLinear Regression Results:")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"\nFormula:")
    print(f"Overall_Score = {intercept:.4f}")
    
    for i, col in enumerate(score_columns):
        coef = coefficients[i]
        if coef >= 0:
            print(f" + {coef:.4f}*{col}")
        else:
            print(f" {coef:.4f}*{col}")
    
    # Create coefficient table
    coef_df = pd.DataFrame({
        'Indicator': score_columns,
        'Coefficient': coefficients
    })
    print(f"\nCoefficient Table:")
    print(coef_df.round(4))
    
    # Test the formula on a few examples
    print(f"\nFormula Validation (first 3 universities):")
    for i in range(min(3, len(df_clean))):
        actual = df_clean.iloc[i]['Overall_Score']
        predicted = intercept + sum(coefficients[j] * df_clean.iloc[i][score_columns[j]] for j in range(len(score_columns)))
        print(f"University {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Difference={abs(actual-predicted):.2f}")
    
    return model, coefficients, intercept, r_squared

def create_what_if_scenarios(filepath, prev_filepath=None, predicted_median=None):
    """
    Creates what-if scenarios for Khalifa University to reach the target median overall score.
    Uses the regression formula to calculate required score improvements for each indicator.
    If prev_filepath is provided, adds a scenario projecting the same improvement as from prev_filepath to filepath.
    If predicted_median is provided, uses it as the target instead of current year's median.
    Adds Scenario 5: increases each indicator by the average historical improvement (from all available years where the indicator exists).
    """
    # First, get the regression coefficients
    model, coefficients, intercept, r_squared = calculate_indicator_coefficients(filepath)
    if model is None:
        return
    
    # Get the target median overall score
    if predicted_median is not None:
        target_median = predicted_median
        target_label = "predicted 2026-2027"
    else:
        # Get the median overall score for universities ranked 100-150 from current year
        df = pd.read_csv(filepath)
        df['RANK_2026'] = pd.to_numeric(df['RANK_2026'], errors='coerce')
        df['Overall_Score'] = pd.to_numeric(df['Overall_Score'], errors='coerce')
        df_sorted = df.sort_values('RANK_2026')
        top_150 = df_sorted.head(150)
        between_100_150 = top_150[(top_150['RANK_2026'] >= 100) & (top_150['RANK_2026'] <= 150)]
        target_median = between_100_150['Overall_Score'].median()
        target_label = "current year (2025-2026)"
    
    # Get Khalifa University's current scores
    df = pd.read_csv(filepath)
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
    else:
        print("Institution Name column not found.")
        return
    
    # Get current scores for indicators used in regression
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    current_scores = {}
    for col in score_columns:
        current_scores[col] = pd.to_numeric(ku_row.iloc[0][col], errors='coerce')
    
    # Calculate current predicted overall score
    current_predicted = intercept + sum(coefficients[i] * current_scores[score_columns[i]] for i in range(len(score_columns)))
    
    print(f"\n=== WHAT-IF SCENARIOS FOR KHALIFA UNIVERSITY (Target: {target_label}) ===")
    print(f"Target median overall score: {target_median:.2f}")
    print(f"Current predicted overall score: {current_predicted:.2f}")
    print(f"Gap to target: {target_median - current_predicted:.2f}")
    
    # Scenario 1: Equal improvement across all indicators
    print(f"\n--- Scenario 1: Equal Improvement Across All Indicators ---")
    gap = target_median - current_predicted
    total_coefficient = sum(coefficients)
    equal_improvement = gap / total_coefficient if total_coefficient != 0 else 0
    
    scenario1_scores = {}
    for i, col in enumerate(score_columns):
        scenario1_scores[col] = current_scores[col] + equal_improvement
        print(f"{col}: {current_scores[col]:.2f} → {scenario1_scores[col]:.2f} (+{equal_improvement:.2f})")
    
    # Scenario 2: Focus on highest coefficient indicators
    print(f"\n--- Scenario 2: Focus on Highest Impact Indicators ---")
    coef_indicator_pairs = list(zip(coefficients, score_columns))
    coef_indicator_pairs.sort(reverse=True)  # Sort by coefficient (highest first)
    
    # Improve top 3 indicators by 50% of the gap each
    scenario2_scores = current_scores.copy()
    for i in range(min(3, len(coef_indicator_pairs))):
        coef, indicator = coef_indicator_pairs[i]
        improvement = (gap * 0.5) / coef if coef != 0 else 0
        scenario2_scores[indicator] += improvement
        print(f"{indicator}: {current_scores[indicator]:.2f} → {scenario2_scores[indicator]:.2f} (+{improvement:.2f})")
    
    # Scenario 3: Focus on lowest current scores
    print(f"\n--- Scenario 3: Focus on Lowest Current Scores ---")
    current_score_pairs = [(current_scores[col], col) for col in score_columns]
    current_score_pairs.sort()  # Sort by current score (lowest first)
    
    scenario3_scores = current_scores.copy()
    for i in range(min(3, len(current_score_pairs))):
        current_score, indicator = current_score_pairs[i]
        coef_idx = score_columns.index(indicator)
        coef = coefficients[coef_idx]
        improvement = (gap * 0.4) / coef if coef != 0 else 0
        scenario3_scores[indicator] += improvement
        print(f"{indicator}: {current_scores[indicator]:.2f} → {scenario3_scores[indicator]:.2f} (+{improvement:.2f})")
    
    # Scenario 4: Projected improvement based on previous year
    if prev_filepath is not None:
        print(f"\n--- Scenario 4: Projected Improvement Based on Previous Year ---")
        df_prev = pd.read_csv(prev_filepath)
        if 'Institution Name' in df_prev.columns:
            ku_prev = df_prev[df_prev['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
            if not ku_prev.empty:
                prev_scores = {col: pd.to_numeric(ku_prev.iloc[0][col], errors='coerce') for col in score_columns}
                projected_scores = {}
                for col in score_columns:
                    increment = current_scores[col] - prev_scores.get(col, current_scores[col])
                    projected_scores[col] = current_scores[col] + increment
                    print(f"{col}: {current_scores[col]:.2f} → {projected_scores[col]:.2f} (increment: {increment:.2f})")
                projected_predicted = intercept + sum(coefficients[i] * projected_scores[score_columns[i]] for i in range(len(score_columns)))
                print(f"Projected overall score = {projected_predicted:.2f} (Target: {target_median:.2f})")
                if projected_predicted >= target_median:
                    print("Khalifa University would be in the 100-150 rank group with this improvement.")
                else:
                    print("Khalifa University would NOT be in the 100-150 rank group with this improvement.")
            else:
                print("Khalifa University not found in previous year data.")
        else:
            print("Institution Name column not found in previous year data.")
    
    # Scenario 5: Average historical improvement for each indicator
    print(f"\n--- Scenario 5: Average Historical Improvement for Each Indicator ---")
    import os
    scenario5_scores = current_scores.copy()
    years_files = [
        ('2022', 'cleaned_dataset/qs_2022_2023.csv'),
        ('2023', 'cleaned_dataset/qs_2023_2024.csv'),
        ('2024', 'cleaned_dataset/qs_2024_2025.csv'),
        ('2025', 'cleaned_dataset/qs_2025_2026.csv'),
    ]
    indicator_history = {col: [] for col in score_columns}
    for year, file in years_files:
        if os.path.exists(file):
            df_year = pd.read_csv(file)
            if 'Institution Name' in df_year.columns:
                ku_year = df_year[df_year['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
                if not ku_year.empty:
                    for col in score_columns:
                        if col in ku_year.columns:
                            indicator_history[col].append(pd.to_numeric(ku_year.iloc[0][col], errors='coerce'))
                        else:
                            indicator_history[col].append(np.nan)
                else:
                    for col in score_columns:
                        indicator_history[col].append(np.nan)
            else:
                for col in score_columns:
                    indicator_history[col].append(np.nan)
        else:
            for col in score_columns:
                indicator_history[col].append(np.nan)
    # Calculate average improvement for each indicator
    for col in score_columns:
        history = [v for v in indicator_history[col] if not pd.isna(v)]
        if len(history) >= 2:
            improvements = [history[i+1] - history[i] for i in range(len(history)-1)]
            avg_improvement = np.nanmean(improvements)
            scenario5_scores[col] = current_scores[col] + avg_improvement
            print(f"{col}: {current_scores[col]:.2f} → {scenario5_scores[col]:.2f} (avg historical increment: {avg_improvement:.2f})")
        else:
            print(f"{col}: Not enough data for historical improvement.")
    predicted5 = intercept + sum(coefficients[i] * scenario5_scores[score_columns[i]] for i in range(len(score_columns)))
    print(f"Scenario 5: Predicted overall score = {predicted5:.2f} (Target: {target_median:.2f})")
    
    # Validate scenarios
    print(f"\n--- Scenario Validation ---")
    scenarios = {
        "Scenario 1 (Equal)": scenario1_scores,
        "Scenario 2 (High Impact)": scenario2_scores,
        "Scenario 3 (Low Scores)": scenario3_scores,
        "Scenario 5 (Avg Historical)": scenario5_scores
    }
    
    for name, scores in scenarios.items():
        predicted = intercept + sum(coefficients[i] * scores[score_columns[i]] for i in range(len(score_columns)))
        print(f"{name}: Predicted overall score = {predicted:.2f} (Target: {target_median:.2f})")
    
    return scenarios, target_median, current_predicted

def predict_future_median(historical_medians):
    """
    Predicts the 2026-2027 median overall score using linear regression on historical data.
    """
    if len(historical_medians) < 2:
        print("Need at least 2 data points for prediction.")
        return None
    
    # Create year labels and corresponding median values
    years = list(historical_medians.keys())
    medians = list(historical_medians.values())
    
    # Convert years to numeric for regression (2022=1, 2023=2, etc.)
    year_nums = [int(year.split('-')[0]) - 2021 for year in years]
    
    # Fit linear regression
    model = LinearRegression()
    X = np.array(year_nums).reshape(-1, 1)
    y = np.array(medians)
    model.fit(X, y)
    
    # Predict 2026-2027 (year 6)
    next_year = 6
    predicted_median = model.predict([[next_year]])[0]
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    print(f"\n=== PREDICTION FOR 2026-2027 ===")
    print(f"Historical trend R-squared: {r_squared:.4f}")
    print(f"Predicted median overall score (ranks 100-150) for 2026-2027: {predicted_median:.2f}")
    
    # Show the trend
    print(f"\nHistorical trend:")
    for i, year in enumerate(years):
        print(f"{year}: {medians[i]:.2f}")
    print(f"2026-2027: {predicted_median:.2f} (predicted)")
    
    return predicted_median, model

def calculate_median_100_150(filepath, year_label):
    """
    Calculates and prints the median overall score for universities ranked 100-150 from the given file.
    """
    df = pd.read_csv(filepath)
    
    # Determine the correct rank column based on the year
    if '2022_2023' in filepath:
        rank_col = 'RANK_2023'
    elif '2023_2024' in filepath:
        rank_col = 'RANK_2024'
    elif '2024_2025' in filepath:
        rank_col = 'RANK_2025'
    elif '2025_2026' in filepath:
        rank_col = 'RANK_2026'
    else:
        print(f"Could not determine rank column for {filepath}")
        return None, None
    
    # Clean and convert rank column
    df = clean_rank_columns(df, [rank_col])
    df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
    df['Overall_Score'] = pd.to_numeric(df['Overall_Score'], errors='coerce')
    
    # Sort and get top 150
    df_sorted = df.sort_values(rank_col)
    top_150 = df_sorted.head(150)
    between_100_150 = top_150[(top_150[rank_col] >= 100) & (top_150[rank_col] <= 150)]
    
    median_score = between_100_150['Overall_Score'].median()
    mean_score = between_100_150['Overall_Score'].mean()
    
    print(f"\n=== {year_label} STATISTICS ===")
    print(f"Median Overall Score (ranks 100-150): {median_score:.2f}")
    print(f"Mean Overall Score (ranks 100-150): {mean_score:.2f}")
    print(f"Number of universities in 100-150 range: {len(between_100_150)}")
    
    return median_score, mean_score

def peer_benchmarking_analysis(filepath, benchmark_peers):
    """
    Performs peer and aspiring benchmarking analysis of Khalifa University against specified peer universities.
    Compares indicator scores, overall performance, and identifies areas for improvement.
    """
    df = pd.read_csv(filepath)
    
    # Get score columns
    score_columns = ['Overall_Score', 'SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    # Convert scores to numeric
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get rank column
    rank_col = 'RANK_2026' if 'RANK_2026' in df.columns else 'RANK_2025' if 'RANK_2025' in df.columns else 'RANK_2024'
    df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
    
    # Find Khalifa University
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
    else:
        print("Institution Name column not found.")
        return
    
    # Get Khalifa University data
    ku_data = {}
    for col in score_columns:
        ku_data[col] = ku_row.iloc[0][col]
    ku_rank = ku_row.iloc[0][rank_col]
    
    print(f"\n=== PEER BENCHMARKING ANALYSIS ===")
    print(f"Khalifa University of Science and Technology")
    print(f"Current Rank: {ku_rank}")
    print(f"Overall Score: {ku_data['Overall_Score']:.2f}")
    
    # Find peer universities
    peer_data = {}
    for peer_name, country in benchmark_peers:
        # Try exact match first
        peer_row = df[df['Institution Name'].str.strip().str.lower() == peer_name.lower()]
        if peer_row.empty:
            # Try partial match
            peer_row = df[df['Institution Name'].str.contains(peer_name, case=False, na=False)]
        
        if not peer_row.empty:
            peer_data[peer_name] = {
                'country': country,
                'rank': peer_row.iloc[0][rank_col],
                'scores': {}
            }
            for col in score_columns:
                peer_data[peer_name]['scores'][col] = peer_row.iloc[0][col]
            print(f"\nFound: {peer_name} ({country}) - Rank: {peer_data[peer_name]['rank']}")
        else:
            print(f"\nNot found: {peer_name} ({country})")
    
    if not peer_data:
        print("No peer universities found in the data.")
        return
    
    # Create comparison table
    print(f"\n=== INDICATOR COMPARISON ===")
    comparison_data = []
    
    for indicator in score_columns:
        row_data = {'Indicator': indicator.replace('SCORE_', '').replace('_', ' ')}
        row_data['Khalifa Univ'] = ku_data[indicator]
        
        for peer_name in peer_data.keys():
            row_data[peer_name] = peer_data[peer_name]['scores'][indicator]
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(2))
    
    # Calculate gaps and identify opportunities
    print(f"\n=== GAP ANALYSIS ===")
    print("Gaps (Peer Score - Khalifa Score):")
    gap_data = []
    
    for indicator in score_columns:
        if indicator == 'Overall_Score':
            continue
            
        row_data = {'Indicator': indicator.replace('SCORE_', '').replace('_', ' ')}
        ku_score = ku_data[indicator]
        
        for peer_name in peer_data.keys():
            peer_score = peer_data[peer_name]['scores'][indicator]
            gap = peer_score - ku_score
            row_data[peer_name] = gap
            
            if gap > 0:
                print(f"{indicator}: {peer_name} leads by {gap:.2f} points")
        
        gap_data.append(row_data)
    
    gap_df = pd.DataFrame(gap_data)
    print(f"\nGap Summary Table:")
    print(gap_df.round(2))
    
    # Identify best performers for each indicator
    print(f"\n=== BEST PERFORMERS BY INDICATOR ===")
    for indicator in score_columns:
        if indicator == 'Overall_Score':
            continue
            
        scores = {peer_name: peer_data[peer_name]['scores'][indicator] for peer_name in peer_data.keys()}
        scores['Khalifa Univ'] = ku_data[indicator]
        
        best_performer = max(scores, key=scores.get)
        best_score = scores[best_performer]
        
        print(f"{indicator.replace('SCORE_', '').replace('_', ' ')}: {best_performer} ({best_score:.2f})")
    
    # Calculate average peer performance
    print(f"\n=== AVERAGE PEER PERFORMANCE ===")
    avg_peer_scores = {}
    for indicator in score_columns:
        peer_scores = [peer_data[peer_name]['scores'][indicator] for peer_name in peer_data.keys()]
        avg_peer_scores[indicator] = np.nanmean(peer_scores)
        
        gap_to_avg = avg_peer_scores[indicator] - ku_data[indicator]
        print(f"{indicator}: Average peer = {avg_peer_scores[indicator]:.2f}, Khalifa = {ku_data[indicator]:.2f}, Gap = {gap_to_avg:.2f}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("Priority areas for improvement (based on largest gaps to peer average):")
    
    gaps_to_avg = []
    for indicator in score_columns:
        if indicator == 'Overall_Score':
            continue
        gap = avg_peer_scores[indicator] - ku_data[indicator]
        gaps_to_avg.append((indicator, gap))
    
    gaps_to_avg.sort(key=lambda x: x[1], reverse=True)
    
    for i, (indicator, gap) in enumerate(gaps_to_avg[:3]):
        if gap > 0:
            print(f"{i+1}. {indicator.replace('SCORE_', '').replace('_', ' ')}: {gap:.2f} points below peer average")
    
    return comparison_df, gap_df, peer_data, avg_peer_scores

def create_peer_comparison_chart(filepath, benchmark_peers):
    """
    Creates a bar chart comparing Khalifa University's indicator scores against peer average.
    """
    # Get the peer benchmarking data
    comparison_df, gap_df, peer_data, avg_peer_scores = peer_benchmarking_analysis(filepath, benchmark_peers)
    
    if avg_peer_scores is None:
        print("No peer data available for chart creation.")
        return
    
    # Get Khalifa University data
    df = pd.read_csv(filepath)
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
    else:
        print("Institution Name column not found.")
        return
    
    # Get score columns (exclude Overall_Score for cleaner chart)
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    # Get Khalifa University scores
    ku_scores = {}
    for col in score_columns:
        ku_scores[col] = pd.to_numeric(ku_row.iloc[0][col], errors='coerce')
    
    # Prepare data for plotting
    indicators = [col.replace('SCORE_', '').replace('_', ' ') for col in score_columns]
    khalifa_values = [ku_scores[col] for col in score_columns]
    peer_avg_values = [avg_peer_scores[col] for col in score_columns]
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(indicators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, khalifa_values, width, label='Khalifa University', color='#9C27B0', alpha=0.8)
    bars2 = ax.bar(x + width/2, peer_avg_values, width, label='Peer Average', color='#ff7f0e', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel('Indicators', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Khalifa University vs Peer Average - Indicator Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(indicators, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Add gap annotations
    for i, (khalifa_val, peer_val) in enumerate(zip(khalifa_values, peer_avg_values)):
        if not (np.isnan(khalifa_val) or np.isnan(peer_val)):
            gap = peer_val - khalifa_val
            if abs(gap) > 1:  # Only show significant gaps
                color = 'red' if gap > 0 else 'green'
                ax.annotate(f'Gap: {gap:+.1f}',
                           xy=(i, max(khalifa_val, peer_val) + 2),
                           ha='center', va='bottom',
                           color=color, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('khalifa_vs_peer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Peer comparison chart saved as 'khalifa_vs_peer_comparison.png'")
    
    return fig

def recommend_peer_examples(filepath, benchmark_peers):
    """
    Based on indicator gaps between Khalifa University and peers, recommends which specific peer universities
    Khalifa University should look into as examples for improvement in each area.
    """
    # Get the peer benchmarking data
    comparison_df, gap_df, peer_data, avg_peer_scores = peer_benchmarking_analysis(filepath, benchmark_peers)
    
    if peer_data is None or not peer_data:
        print("No peer data available for recommendations.")
        return
    
    # Get Khalifa University data
    df = pd.read_csv(filepath)
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
    else:
        print("Institution Name column not found.")
        return
    
    # Get score columns (exclude Overall_Score)
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    # Get Khalifa University scores
    ku_scores = {}
    for col in score_columns:
        ku_scores[col] = pd.to_numeric(ku_row.iloc[0][col], errors='coerce')
    
    print(f"\n=== PEER UNIVERSITY RECOMMENDATIONS FOR KHALIFA UNIVERSITY ===")
    print("Based on indicator gaps, here are the recommended peer universities to study:")
    
    # Analyze each indicator and recommend the best peer
    recommendations = {}
    
    for indicator in score_columns:
        indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
        ku_score = ku_scores[indicator]
        
        if pd.isna(ku_score):
            continue
        
        # Find the best performing peer for this indicator
        best_peer = None
        best_score = -1
        peer_scores = {}
        
        for peer_name in peer_data.keys():
            peer_score = peer_data[peer_name]['scores'][indicator]
            if not pd.isna(peer_score):
                peer_scores[peer_name] = peer_score
                if peer_score > best_score:
                    best_score = peer_score
                    best_peer = peer_name
        
        if best_peer and best_score > ku_score:
            gap = best_score - ku_score
            recommendations[indicator] = {
                'best_peer': best_peer,
                'best_score': best_score,
                'gap': gap,
                'all_peers': peer_scores
            }
            
            print(f"\n📊 {indicator_name}:")
            print(f"   Khalifa University: {ku_score:.2f}")
            print(f"   Best Peer: {best_peer} ({best_score:.2f})")
            print(f"   Gap: {gap:.2f} points")
            print(f"   Recommendation: Study {best_peer}'s approach to {indicator_name}")
            
            # Show all peer scores for context
            print(f"   All peer scores:")
            for peer, score in sorted(peer_scores.items(), key=lambda x: x[1], reverse=True):
                peer_gap = score - ku_score
                status = "🔴" if peer_gap > 0 else "🟢"
                print(f"     {status} {peer}: {score:.2f} (gap: {peer_gap:+.2f})")
    
    # Create summary table
    print(f"\n=== SUMMARY OF RECOMMENDATIONS ===")
    if recommendations:
        summary_data = []
        for indicator, data in recommendations.items():
            indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
            summary_data.append({
                'Indicator': indicator_name,
                'Khalifa Score': ku_scores[indicator],
                'Best Peer': data['best_peer'],
                'Best Peer Score': data['best_score'],
                'Gap': data['gap'],
                'Recommendation': f"Study {data['best_peer']}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.round(2))
        
        # Identify most critical areas
        print(f"\n=== MOST CRITICAL AREAS (Largest Gaps) ===")
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1]['gap'], reverse=True)
        
        for i, (indicator, data) in enumerate(sorted_recommendations[:3]):
            indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
            print(f"{i+1}. {indicator_name}: {data['gap']:.2f} points behind {data['best_peer']}")
            print(f"   Priority: HIGH - Focus on studying {data['best_peer']}'s {indicator_name} strategies")
        
        # Identify which peer universities are most relevant overall
        print(f"\n=== MOST RELEVANT PEER UNIVERSITIES ===")
        peer_relevance = {}
        for indicator, data in recommendations.items():
            peer = data['best_peer']
            if peer not in peer_relevance:
                peer_relevance[peer] = {'count': 0, 'total_gap': 0, 'indicators': []}
            peer_relevance[peer]['count'] += 1
            peer_relevance[peer]['total_gap'] += data['gap']
            peer_relevance[peer]['indicators'].append(indicator.replace('SCORE_', '').replace('_', ' '))
        
        sorted_peers = sorted(peer_relevance.items(), key=lambda x: (x[1]['count'], x[1]['total_gap']), reverse=True)
        
        for peer, data in sorted_peers:
            print(f"\n🏆 {peer}:")
            print(f"   Leading in {data['count']} indicators")
            print(f"   Total gap: {data['total_gap']:.2f} points")
            print(f"   Indicators: {', '.join(data['indicators'])}")
            print(f"   Recommendation: Primary focus for benchmarking")
    else:
        print("No significant gaps found. Khalifa University is performing well compared to peers!")
    
    return recommendations, summary_df if 'summary_df' in locals() else None

def create_peer_learning_table(filepath, benchmark_peers):
    """
    Creates a formatted table showing peer universities vs top 3 improvement indicators,
    with gaps and color-coded recommendations for which peer Khalifa University should learn from.
    """
    # Get the peer benchmarking data
    comparison_df, gap_df, peer_data, avg_peer_scores = peer_benchmarking_analysis(filepath, benchmark_peers)
    
    if peer_data is None or not peer_data:
        print("No peer data available for table creation.")
        return
    
    # Get Khalifa University data
    df = pd.read_csv(filepath)
    if 'Institution Name' in df.columns:
        ku_row = df[df['Institution Name'].str.strip().str.lower() == 'khalifa university of science and technology']
        if ku_row.empty:
            print("Khalifa University not found in the data.")
            return
    else:
        print("Institution Name column not found.")
        return
    
    # Get score columns (exclude Overall_Score)
    score_columns = ['SCORE_AR', 'SCORE_ER', 'SCORE_FS', 'SCORE_CPF', 'SCORE_IF', 'SCORE_IS', 'SCORE_IRN', 'SCORE_EO', 'SCORE_ST']
    score_columns = [col for col in score_columns if col in df.columns]
    
    # Get Khalifa University scores
    ku_scores = {}
    for col in score_columns:
        ku_scores[col] = pd.to_numeric(ku_row.iloc[0][col], errors='coerce')
    
    # Calculate gaps to peer average and find top 3 improvement areas
    gaps_to_avg = []
    for indicator in score_columns:
        if indicator in avg_peer_scores:
            gap = avg_peer_scores[indicator] - ku_scores[indicator]
            gaps_to_avg.append((indicator, gap))
    
    gaps_to_avg.sort(key=lambda x: x[1], reverse=True)
    top_3_indicators = [indicator for indicator, gap in gaps_to_avg[:3]]
    
    print(f"\n=== PEER LEARNING TABLE ===")
    print("Top 3 Improvement Areas for Khalifa University:")
    for i, indicator in enumerate(top_3_indicators):
        indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
        gap = gaps_to_avg[i][1]
        print(f"{i+1}. {indicator_name} (Gap to peer average: {gap:.2f})")
    
    # Create the table
    print(f"\n{'Peer University':<30} {'Gap to Avg':<12}", end="")
    for indicator in top_3_indicators:
        indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
        print(f" {indicator_name:<15}", end="")
    print()
    
    print("-" * (42 + len(top_3_indicators) * 15))
    
    # Add Khalifa University row first
    print(f"{'Khalifa University':<30} {'N/A':<12}", end="")
    for indicator in top_3_indicators:
        score = ku_scores[indicator]
        print(f" {score:<15.2f}", end="")
    print(" (Baseline)")
    
    print("-" * (42 + len(top_3_indicators) * 15))
    
    # Find best performer for each indicator
    best_performers = {}
    for indicator in top_3_indicators:
        best_peer = None
        best_score = -1
        for peer_name in peer_data.keys():
            peer_score = peer_data[peer_name]['scores'][indicator]
            if not pd.isna(peer_score) and peer_score > best_score:
                best_score = peer_score
                best_peer = peer_name
        best_performers[indicator] = best_peer
    
    # Add peer university rows
    for peer_name in peer_data.keys():
        # Calculate gap to peer average
        peer_scores = [peer_data[peer_name]['scores'][col] for col in score_columns if col in avg_peer_scores]
        peer_avg = np.nanmean(peer_scores)
        gap_to_avg = peer_avg - ku_scores.get('Overall_Score', 0)
        
        print(f"{peer_name:<30} {gap_to_avg:<12.2f}", end="")
        
        for indicator in top_3_indicators:
            score = peer_data[peer_name]['scores'][indicator]
            if not pd.isna(score):
                # Check if this peer is the best performer for this indicator
                if peer_name == best_performers[indicator]:
                    print(f" **{score:<13.2f}**", end="")  # Bold for best performer
                else:
                    print(f" {score:<15.2f}", end="")
            else:
                print(f" {'N/A':<15}", end="")
        
        # Add recommendation if this peer is best in any indicator
        best_in = [ind for ind, best in best_performers.items() if best == peer_name]
        if best_in:
            indicators_str = ", ".join([ind.replace('SCORE_', '').replace('_', ' ') for ind in best_in])
            print(f" ← Learn from ({indicators_str})")
        else:
            print()
    
    print("-" * (42 + len(top_3_indicators) * 15))
    
    # Add summary row
    print(f"{'Peer Average':<30} {'N/A':<12}", end="")
    for indicator in top_3_indicators:
        avg_score = avg_peer_scores[indicator]
        print(f" {avg_score:<15.2f}", end="")
    print(" (Target)")
    
    # Add legend
    print(f"\nLegend:")
    print(f"**Bold** = Best performer in that indicator")
    print(f"← Learn from = Recommended peer to study for specific indicators")
    print(f"Gap to Avg = Difference between peer's overall score and Khalifa's overall score")
    
    # Create recommendations summary
    print(f"\n=== STRATEGIC RECOMMENDATIONS ===")
    for indicator in top_3_indicators:
        indicator_name = indicator.replace('SCORE_', '').replace('_', ' ')
        best_peer = best_performers[indicator]
        gap = gaps_to_avg[top_3_indicators.index(indicator)][1]
        
        print(f"🎯 {indicator_name}:")
        print(f"   • Study {best_peer} (best performer: {peer_data[best_peer]['scores'][indicator]:.2f})")
        print(f"   • Gap to peer average: {gap:.2f} points")
        print(f"   • Priority: {'HIGH' if gap > 5 else 'MEDIUM' if gap > 2 else 'LOW'}")
    
    return top_3_indicators, best_performers, gaps_to_avg

def create_historical_rank_chart():
    """
    Creates a line chart showing historical QS rankings for selected universities from 2022 to 2026.
    Reads actual data from CSV files for accuracy.
    """
    # Define peer universities to track
    peer_universities = [
        'Universiti Putra Malaysia',
        'Qatar University', 
        'United Arab Emirates University',
        'King Saud University',
        'Khalifa University of Science and Technology'
    ]
    
    # Define years and corresponding file paths with correct rank column names
    year_files = {
        2022: ('cleaned_dataset/qs_2022_2023.csv', 'RANK_2023'),
        2023: ('cleaned_dataset/qs_2023_2024.csv', 'RANK_2024'), 
        2024: ('cleaned_dataset/qs_2024_2025.csv', 'RANK_2025'),
        2025: ('cleaned_dataset/qs_2025_2026.csv', 'RANK_2026')
    }
    
    # Initialize data structure
    universities_data = {university: [] for university in peer_universities}
    years = []
    
    # Read data from each file
    for year, (filepath, rank_col) in year_files.items():
        try:
            df = pd.read_csv(filepath)
            years.append(year)
            
            # Extract ranks for each university
            for university in peer_universities:
                # Try different name variations
                university_variations = [
                    university,
                    university.replace(' of Science and Technology', ''),
                    university.replace('University', 'Univ.'),
                    university.replace('Universiti', 'University')
                ]
                
                rank_found = False
                for variation in university_variations:
                    mask = df['Institution Name'].str.contains(variation, case=False, na=False)
                    if mask.any():
                        rank_value = df.loc[mask, rank_col].iloc[0]
                        universities_data[university].append(rank_value)
                        rank_found = True
                        break
                
                if not rank_found:
                    print(f"Warning: {university} not found in {filepath}")
                    universities_data[university].append(None)
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            for university in peer_universities:
                universities_data[university].append(None)
    
    # Create the chart
    plt.figure(figsize=(16, 10))
    
    # Define colors for each university (Khalifa University in blue, Putra Malaysia in purple)
    colors = ['#9C27B0', '#A23B72', '#F18F01', '#C73E1D', '#1E88E5']  # Purple for Putra Malaysia, Blue for Khalifa
    
    # Plot lines for each university
    for i, (university, ranks) in enumerate(universities_data.items()):
        # Convert string ranges to numeric values for plotting
        numeric_ranks = []
        for rank in ranks:
            if pd.isna(rank) or rank is None:
                numeric_ranks.append(None)
            elif isinstance(rank, str):
                if '-' in rank:
                    # For ranges like "251-300", use the midpoint
                    start, end = map(int, rank.split('-'))
                    numeric_ranks.append((start + end) / 2)
                else:
                    # Remove any non-numeric characters and convert
                    rank_clean = str(rank).replace('=', '').replace('"', '').strip()
                    try:
                        numeric_ranks.append(float(rank_clean))
                    except:
                        numeric_ranks.append(None)
            else:
                numeric_ranks.append(float(rank))
        
        # Filter out None values for plotting
        valid_years = []
        valid_ranks = []
        for year, rank in zip(years, numeric_ranks):
            if rank is not None:
                valid_years.append(year)
                valid_ranks.append(rank)
        
        if valid_ranks:
            # Make Khalifa University line thicker
            if university == 'Khalifa University of Science and Technology':
                linewidth = 8
                markersize = 16
            else:
                linewidth = 5
                markersize = 12
            
            plt.plot(valid_years, valid_ranks, marker='o', linewidth=linewidth, markersize=markersize, 
                    label=university, color=colors[i], alpha=0.8)
    
    # Customize the chart
    plt.xlabel('', fontsize=14, fontweight='bold')
    plt.ylabel('', fontsize=14, fontweight='bold')
    
    # Invert y-axis so lower numbers (better rankings) appear at the top
    plt.gca().invert_yaxis()
    
    # Set y-axis limits and ticks
    plt.ylim(350, 100)
    plt.yticks(range(100, 351, 50))
    
    # Format x-axis to show years as integers
    plt.xticks(years, [int(year) for year in years])
    
    # Remove grid
    plt.grid(False)
    
    # Remove borders/spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Add legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, framealpha=0.9)
    
    # Add value labels on points
    for university, ranks in universities_data.items():
        numeric_ranks = []
        for rank in ranks:
            if pd.isna(rank) or rank is None:
                numeric_ranks.append(None)
            elif isinstance(rank, str):
                if '-' in rank:
                    start, end = map(int, rank.split('-'))
                    numeric_ranks.append((start + end) / 2)
                else:
                    rank_clean = str(rank).replace('=', '').replace('"', '').strip()
                    try:
                        numeric_ranks.append(float(rank_clean))
                    except:
                        numeric_ranks.append(None)
            else:
                numeric_ranks.append(float(rank))
        
        for year, rank in zip(years, numeric_ranks):
            if rank is not None:
                plt.annotate(f'{rank:.0f}', (year, rank), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center', 
                           fontsize=10,
                           fontweight='bold')
    
    # Set background color
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('historical_rankings_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Historical rankings chart saved as 'historical_rankings_chart.png'")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HISTORICAL RANKING ANALYSIS SUMMARY (2022-2026)")
    print("="*60)
    
    for university, ranks in universities_data.items():
        numeric_ranks = []
        for rank in ranks:
            if pd.isna(rank) or rank is None:
                numeric_ranks.append(None)
            elif isinstance(rank, str):
                if '-' in rank:
                    start, end = map(int, rank.split('-'))
                    numeric_ranks.append((start + end) / 2)
                else:
                    rank_clean = str(rank).replace('=', '').replace('"', '').strip()
                    try:
                        numeric_ranks.append(float(rank_clean))
                    except:
                        numeric_ranks.append(None)
            else:
                numeric_ranks.append(float(rank))
        
        # Filter out None values
        valid_ranks = [r for r in numeric_ranks if r is not None]
        valid_years = [y for y, r in zip(years, numeric_ranks) if r is not None]
        
        if valid_ranks:
            start_rank = valid_ranks[0]
            end_rank = valid_ranks[-1]
            improvement = start_rank - end_rank
            
            print(f"\n{university}:")
            print(f"  {valid_years[0]} Rank: {ranks[0]}")
            print(f"  {valid_years[-1]} Rank: {ranks[-1]}")
            print(f"  Improvement: {improvement:.0f} positions")
            
            # Find best and worst years
            best_year_idx = np.argmin(valid_ranks)
            worst_year_idx = np.argmax(valid_ranks)
            print(f"  Best Year: {valid_years[best_year_idx]} (Rank: {ranks[valid_years.index(valid_years[best_year_idx])]})")
            print(f"  Worst Year: {valid_years[worst_year_idx]} (Rank: {ranks[valid_years.index(valid_years[worst_year_idx])]})")
        else:
            print(f"\n{university}: No valid data found")

def create_ranking_employment_correlation():
    """
    Creates a vertical bar chart showing employment ratios and university ranks 
    for Qatar University and Khalifa University only.
    """
    # Read the 2025-2026 QS rankings data
    qs_data = pd.read_csv('cleaned_dataset/qs_2025_2026.csv')
    
    # Read the employment data
    employment_data = pd.read_csv('cleaned_dataset/world_bank/employed_to_population_ratio.csv')
    
    # Clean up employment data - get the latest available year (2024)
    employment_data = employment_data[['Country Name', '2024']].copy()
    employment_data.columns = ['Country', 'Employment_Ratio_2024']
    employment_data = employment_data.dropna()
    
    # Convert employment ratio to numeric
    employment_data['Employment_Ratio_2024'] = pd.to_numeric(employment_data['Employment_Ratio_2024'], errors='coerce')
    employment_data = employment_data.dropna()
    
    # Define universities to analyze with their countries (only Qatar and Khalifa)
    universities_countries = {
        'Qatar University': 'Qatar',
        'Khalifa University of Science and Technology': 'United Arab Emirates'
    }
    
    # Extract data for these universities
    chart_data = []
    
    for university, country in universities_countries.items():
        # Find university in QS data
        uni_row = qs_data[qs_data['Institution Name'].str.strip().str.lower() == university.lower()]
        
        if not uni_row.empty:
            rank = uni_row['RANK_2026'].iloc[0]
            
            # Clean rank data
            if isinstance(rank, str):
                if '=' in rank:
                    rank = rank.replace('=', '')
                if '-' in rank:
                    # For ranges, use the midpoint
                    start, end = map(int, rank.split('-'))
                    rank = (start + end) / 2
            
            rank = pd.to_numeric(rank, errors='coerce')
            
            # Find country employment data
            country_emp = employment_data[employment_data['Country'] == country]
            
            if not country_emp.empty and not pd.isna(rank):
                emp_ratio = country_emp['Employment_Ratio_2024'].iloc[0]
                chart_data.append({
                    'University': university,
                    'Country': country,
                    'Rank': rank,
                    'Employment_Ratio': emp_ratio
                })
    
    if not chart_data:
        print("No data found for analysis.")
        return
    
    # Create DataFrame
    df_chart = pd.DataFrame(chart_data)
    
    # Create the vertical bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Define colors
    colors = ['#1E88E5', '#9C27B0']  # Blue for Qatar, Purple for Khalifa
    
    # Create bars for employment ratio (top subplot)
    x_pos = np.arange(len(df_chart))
    bars1 = ax1.bar(x_pos, df_chart['Employment_Ratio'], 
                    color=colors[:len(df_chart)], alpha=0.7, 
                    edgecolor='black', linewidth=2)
    
    # Customize employment ratio subplot
    ax1.set_ylabel('Employment to Population Ratio (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Employment Ratios by Country (2024)', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels for employment
    ax1.set_xticks(x_pos)
    x_labels = []
    for _, row in df_chart.iterrows():
        uni_short = row['University'].split()[0]  # First word of university name
        country_short = row['Country'].split()[-1]  # Last word of country name
        x_labels.append(f"{uni_short}\n({country_short})")
    
    ax1.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    
    # Add value labels on employment bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Create bars for university rank (bottom subplot)
    bars2 = ax2.bar(x_pos, df_chart['Rank'], 
                    color=colors[:len(df_chart)], alpha=0.7, 
                    edgecolor='black', linewidth=2)
    
    # Customize university rank subplot
    ax2.set_ylabel('QS World University Ranking', fontsize=14, fontweight='bold')
    ax2.set_title('University Rankings (2025-2026)', fontsize=16, fontweight='bold', pad=20)
    ax2.invert_yaxis()  # Invert so lower rank (better) appears higher
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels for rankings
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    
    # Add value labels on ranking bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('qatar_khalifa_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print analysis
    print("\n" + "="*70)
    print("QATAR UNIVERSITY vs KHALIFA UNIVERSITY COMPARISON")
    print("="*70)
    print(f"{'University':<35} {'Country':<20} {'Rank':<8} {'Emp. Ratio':<12}")
    print("-" * 70)
    
    for _, row in df_chart.iterrows():
        print(f"{row['University']:<35} {row['Country']:<20} {row['Rank']:<8.0f} {row['Employment_Ratio']:<12.2f}%")
    
    print("\nKey Insights:")
    print("-" * 40)
    
    # Find best and worst performers
    best_rank = df_chart.loc[df_chart['Rank'].idxmin()]
    highest_emp = df_chart.loc[df_chart['Employment_Ratio'].idxmax()]
    
    print(f"• Better Ranked University: {best_rank['University']} (Rank {best_rank['Rank']:.0f})")
    print(f"• Higher Employment Ratio: {highest_emp['Country']} ({highest_emp['Employment_Ratio']:.1f}%)")
    
    # Calculate correlation
    correlation = df_chart['Employment_Ratio'].corr(df_chart['Rank'])
    print(f"\n• Correlation between Employment Ratio and University Rank: {correlation:.3f}")
    
    if abs(correlation) < 0.3:
        print("  → Weak correlation: No clear relationship between employment and rankings")
    elif abs(correlation) < 0.7:
        print("  → Moderate correlation: Some relationship exists")
    else:
        print("  → Strong correlation: Clear relationship between employment and rankings")
    
    print("\nNote: Employment ratio represents female employment-to-population ratio (15+)")
    print("Lower university rank numbers indicate better performance")
    print("="*70)

def create_malaysia_uae_trends():
    """
    Creates two line charts:
    1. Female research ratio trends for Malaysia and UAE
    2. QS ranking trends for Khalifa University and Universiti Putra Malaysia
    """
    # Read the female research data
    research_data = pd.read_csv('cleaned_dataset/Unisco/femal_research_perecentage_higher_education.csv')
    
    # Filter data for Malaysia (MYS) and UAE (ARE)
    countries_data = research_data[research_data['geoUnit'].isin(['MYS', 'ARE'])]
    
    # Convert year to numeric and sort
    countries_data['year'] = pd.to_numeric(countries_data['year'], errors='coerce')
    countries_data = countries_data.dropna(subset=['year', 'value'])
    countries_data = countries_data.sort_values(['geoUnit', 'year'])
    
    # Create separate figures for each chart
    # Figure 1: Female Research Ratio Trends
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Define colors and country names
    colors = {'MYS': '#9C27B0', 'ARE': '#1E88E5'}  # Purple for Malaysia, Blue for UAE
    country_names = {'MYS': 'Malaysia', 'ARE': 'United Arab Emirates'}
    
    # Plot 1: Female Research Ratio Trends
    for country_code in ['MYS', 'ARE']:
        country_data = countries_data[countries_data['geoUnit'] == country_code]
        # Filter data from 2015 onwards
        country_data = country_data[country_data['year'] >= 2015]
        if not country_data.empty:
            ax1.plot(country_data['year'], country_data['value'], 
                    marker='o', linewidth=3, markersize=8, 
                    color=colors[country_code])
            
            # Add value annotations
            for _, row in country_data.iterrows():
                ax1.annotate(f'{row["value"]:.1f}%', 
                           (row['year'], row['value']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=14, fontweight='bold')
    
    # Customize research ratio subplot (minimal styling)
    ax1.set_xlabel('Year', fontsize=16, fontweight='bold')
    
    # Remove borders and grid
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(False)
    
    # Set x-axis to show only years with data (from 2015 onwards)
    all_years = sorted(countries_data[countries_data['year'] >= 2015]['year'].unique())
    ax1.set_xticks(all_years)
    ax1.set_xticklabels([int(year) for year in all_years], rotation=45, fontsize=14, fontweight='bold')
    
    # Make y-axis numbers bigger
    ax1.tick_params(axis='y', labelsize=14)
    
    # Adjust layout and save first chart
    plt.tight_layout()
    plt.savefig('female_research_trends.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Figure 2: QS Ranking Trends
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Define specific universities to track
    universities = {
        'MYS': 'Universiti Putra Malaysia (UPM)',
        'ARE': 'Khalifa University of Science and Technology'
    }
    
    # Read QS data for multiple years (same approach as create_historical_rank_chart)
    qs_files = {
        2022: ('cleaned_dataset/qs_2022_2023.csv', 'RANK_2023'),
        2023: ('cleaned_dataset/qs_2023_2024.csv', 'RANK_2024'),
        2024: ('cleaned_dataset/qs_2024_2025.csv', 'RANK_2025'),
        2025: ('cleaned_dataset/qs_2025_2026.csv', 'RANK_2026')
    }
    
    ranking_data = {}
    
    for year, (filepath, rank_col) in qs_files.items():
        try:
            qs_data = pd.read_csv(filepath)
            for country_code, university in universities.items():
                if country_code not in ranking_data:
                    ranking_data[country_code] = {}
                
                # Find university in QS data (same approach as historical chart)
                uni_row = qs_data[qs_data['Institution Name'].str.strip().str.lower() == university.lower()]
                
                if not uni_row.empty:
                    rank = uni_row[rank_col].iloc[0]
                    
                    # Clean rank data (same approach as historical chart)
                    if isinstance(rank, str):
                        if '=' in rank:
                            rank = rank.replace('=', '')
                        if '-' in rank:
                            # For ranges like "251-300", use the midpoint
                            start, end = map(int, rank.split('-'))
                            rank = (start + end) / 2
                    
                    rank = pd.to_numeric(rank, errors='coerce')
                    
                    if not pd.isna(rank):
                        ranking_data[country_code][year] = rank
                        print(f"Found {university} in {year}: Rank {rank}")
                    else:
                        print(f"No valid rank found for {university} in {year}")
                else:
                    print(f"University not found: {university} in {year}")
                    
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    # Plot QS ranking trends (same styling as historical chart)
    for country_code in ['MYS', 'ARE']:
        if country_code in ranking_data and ranking_data[country_code]:
            years = sorted(ranking_data[country_code].keys())
            ranks = [ranking_data[country_code][year] for year in years]
            
            # Use same line styling as historical chart
            ax2.plot(years, ranks, marker='o', linewidth=5, markersize=12,
                    color=colors[country_code])
            
            # Add value annotations
            for year, rank in zip(years, ranks):
                ax2.annotate(f'{rank:.0f}', 
                           (year, rank),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=14, fontweight='bold')
    
    # Customize ranking subplot (minimal styling)
    ax2.set_xlabel('Year', fontsize=16, fontweight='bold')
    ax2.invert_yaxis()  # Invert so lower rank (better) appears higher
    
    # Remove borders and grid
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(False)
    
    # Set x-axis for rankings (same as historical chart)
    ranking_years = sorted(set([year for country_data in ranking_data.values() for year in country_data.keys()]))
    if ranking_years:
        ax2.set_xticks(ranking_years)
        ax2.set_xticklabels([int(year) for year in ranking_years], rotation=45, fontsize=14, fontweight='bold')
    
    # Make y-axis numbers bigger
    ax2.tick_params(axis='y', labelsize=14)
    
    # Adjust layout and save second chart
    plt.tight_layout()
    plt.savefig('qs_ranking_trends.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("\n" + "="*80)
    print("TREND ANALYSIS COMPLETED")
    print("="*80)
    print("✓ Female research ratio trends saved as 'female_research_trends.png'")
    print("✓ QS ranking trends saved as 'qs_ranking_trends.png'")
    print("✓ Both charts created with minimal styling (no borders, grids, or titles)")
    print("✓ Value annotations added for better readability")
    print("✓ Years and numbers displayed in larger, bold font")

def predict_optimal_scores_for_top_150_100():
    """
    Predict optimal score improvements for Khalifa University to reach top 150-100 ranks
    for 2026-2027 QS World University Rankings.
    """
    print("\n" + "="*80)
    print("KHALIFA UNIVERSITY: OPTIMAL SCORE PREDICTION FOR TOP 150-100 RANKS")
    print("="*80)
    
    # Read the latest QS data (2025-2026)
    try:
        qs_data = pd.read_csv('cleaned_dataset/qs_2025_2026.csv')
        print("✓ Loaded QS 2025-2026 data")
    except Exception as e:
        print(f"❌ Error loading QS data: {e}")
        return
    
    # Find Khalifa University data
    khalifa_data = qs_data[qs_data['Institution Name'].str.contains('Khalifa', case=False, na=False)]
    
    if khalifa_data.empty:
        print("❌ Khalifa University not found in 2025-2026 dataset")
        return
    
    # Get current Khalifa University scores
    current_scores = {}
    score_columns = ['SCORE_AR', 'SCORE_CPF', 'SCORE_ER', 'SCORE_FS', 'SCORE_EO', 'SCORE_IF', 'SCORE_IRN', 'SCORE_ISD', 'SCORE_IS', 'SCORE_ST']
    
    for col in score_columns:
        if col in khalifa_data.columns:
            current_scores[col] = pd.to_numeric(khalifa_data[col].iloc[0], errors='coerce')
    
    current_overall = pd.to_numeric(khalifa_data['Overall_Score'].iloc[0], errors='coerce') if 'Overall_Score' in khalifa_data.columns else None
    current_rank = pd.to_numeric(khalifa_data['RANK_2026'].iloc[0], errors='coerce') if 'RANK_2026' in khalifa_data.columns else None
    
    print(f"\n📊 CURRENT KHALIFA UNIVERSITY PERFORMANCE (2025-2026):")
    print(f"   • Current Rank: {current_rank}")
    print(f"   • Overall Score: {current_overall:.2f}" if current_overall is not None and not pd.isna(current_overall) else "   • Overall Score: Not available")
    
    # Debug: Check if we have the required columns
    print(f"\n🔍 DEBUG: Available columns in dataset:")
    print(f"   • Score columns found: {[col for col in score_columns if col in khalifa_data.columns]}")
    print(f"   • Missing score columns: {[col for col in score_columns if col not in khalifa_data.columns]}")
    print(f"   • Current scores: {current_scores}")
    
    # Convert rank column to numeric for comparison
    qs_data['RANK_2026_numeric'] = pd.to_numeric(qs_data['RANK_2026'], errors='coerce')
    
    # Analyze top 150-100 universities to understand target scores
    top_150_100 = qs_data[(qs_data['RANK_2026_numeric'] >= 100) & (qs_data['RANK_2026_numeric'] <= 150)].copy()
    
    if top_150_100.empty:
        print("❌ No universities found in rank range 100-150")
        return
    
    # Convert Overall_Score to numeric for analysis
    top_150_100['Overall_Score_numeric'] = pd.to_numeric(top_150_100['Overall_Score'], errors='coerce')
    
    print(f"\n🎯 TARGET RANGE ANALYSIS (Rank 100-150):")
    print(f"   • Number of universities in range: {len(top_150_100)}")
    print(f"   • Average overall score: {top_150_100['Overall_Score_numeric'].mean():.2f}")
    print(f"   • Score range: {top_150_100['Overall_Score_numeric'].min():.2f} - {top_150_100['Overall_Score_numeric'].max():.2f}")
    
    # Calculate target scores for different rank positions
    target_scores = {
        'rank_100': top_150_100['Overall_Score_numeric'].quantile(0.95),  # Top 5% of 100-150 range
        'rank_125': top_150_100['Overall_Score_numeric'].quantile(0.5),   # Median of 100-150 range
        'rank_150': top_150_100['Overall_Score_numeric'].quantile(0.05)   # Bottom 5% of 100-150 range
    }
    
    print(f"\n📈 TARGET SCORES FOR DIFFERENT RANK POSITIONS:")
    for rank_pos, score in target_scores.items():
        rank_num = rank_pos.split('_')[1]
        score_gap = score - current_overall if current_overall else 0
        print(f"   • Rank {rank_num}: {score:.2f} (gap: {score_gap:+.2f})")
    
    # Analyze improvement difficulty across the entire dataset
    print(f"\n🔍 ANALYZING IMPROVEMENT DIFFICULTY ACROSS DATASET:")
    print("-" * 60)
    
    # Convert all score columns to numeric for analysis
    for col in score_columns:
        if col in qs_data.columns:
            qs_data[f'{col}_numeric'] = pd.to_numeric(qs_data[col], errors='coerce')
    
    # Calculate improvement difficulty metrics for each indicator
    difficulty_metrics = {}
    
    for indicator in score_columns:
        if f'{indicator}_numeric' in qs_data.columns:
            scores = qs_data[f'{indicator}_numeric'].dropna()
            
            if len(scores) > 0:
                # Calculate various difficulty metrics
                mean_score = scores.mean()
                std_score = scores.std()
                median_score = scores.median()
                q75 = scores.quantile(0.75)
                q25 = scores.quantile(0.25)
                
                # Difficulty factors:
                # 1. Standard deviation (higher = more variable = potentially easier to improve)
                # 2. Distance from top performers (closer to top = harder to improve)
                # 3. Current score relative to distribution (lower = more room for improvement)
                # 4. Score distribution skewness (how many universities are already high)
                
                current_score = current_scores.get(indicator, 0)
                if pd.isna(current_score):
                    current_score = 0
                
                # Calculate difficulty score (0-100, higher = harder to improve)
                difficulty_score = 0
                
                # Factor 1: Current score relative to distribution (30% weight)
                if current_score >= q75:
                    difficulty_score += 30  # Already in top 25%
                elif current_score >= median_score:
                    difficulty_score += 20  # Above median
                elif current_score >= q25:
                    difficulty_score += 10  # Below median but above bottom 25%
                else:
                    difficulty_score += 0   # In bottom 25% - easier to improve
                
                # Factor 2: Distance from top performers (25% weight)
                top_10_score = scores.quantile(0.90)
                distance_from_top = top_10_score - current_score
                if distance_from_top <= 10:
                    difficulty_score += 25  # Very close to top
                elif distance_from_top <= 20:
                    difficulty_score += 15  # Moderately close
                elif distance_from_top <= 30:
                    difficulty_score += 10  # Some distance
                else:
                    difficulty_score += 5   # Far from top - easier to improve
                
                # Factor 3: Score variability (20% weight)
                cv = std_score / mean_score if mean_score > 0 else 0  # Coefficient of variation
                if cv <= 0.2:
                    difficulty_score += 20  # Low variability - harder to stand out
                elif cv <= 0.4:
                    difficulty_score += 15  # Moderate variability
                elif cv <= 0.6:
                    difficulty_score += 10  # High variability
                else:
                    difficulty_score += 5   # Very high variability - easier to improve
                
                # Factor 4: Current score level (25% weight)
                if current_score >= 80:
                    difficulty_score += 25  # Already very high
                elif current_score >= 60:
                    difficulty_score += 15  # Moderately high
                elif current_score >= 40:
                    difficulty_score += 10  # Moderate
                else:
                    difficulty_score += 5   # Low - easier to improve
                
                difficulty_metrics[indicator] = {
                    'difficulty_score': difficulty_score,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'current_score': current_score,
                    'distance_from_top': distance_from_top,
                    'cv': cv,
                    'q75': q75,
                    'q25': q25
                }
    
    # 2026 QS World University Rankings weighting system
    indicator_weights = {
        'SCORE_AR': 0.30,  # Academic Reputation (30%)
        'SCORE_CPF': 0.20, # Citations per Faculty (20%)
        'SCORE_ER': 0.15,  # Employer Reputation (15%)
        'SCORE_FS': 0.10,  # Faculty Student Ratio (10%)
        'SCORE_EO': 0.05,  # Employment Outcomes (5%)
        'SCORE_IF': 0.05,  # International Faculty Ratio (5%)
        'SCORE_IRN': 0.05, # International Research Network (5%)
        'SCORE_ISD': 0.00, # International Student Diversity (0%)
        'SCORE_IS': 0.05,  # International Student Ratio (5%)
        'SCORE_ST': 0.05   # Sustainability (5%)
    }
    
    # Calculate target improvements for rank 125 (median of 100-150)
    target_overall = target_scores['rank_125']
    score_gap = target_overall - current_overall if current_overall else 0
    
    print(f"Target overall score improvement: {score_gap:+.2f}")
    print(f"Current overall score: {current_overall:.2f}")
    print(f"Target overall score: {target_overall:.2f}")
    
    # Create readable indicator names
    indicator_names = {
        'SCORE_AR': 'Academic Reputation',
        'SCORE_CPF': 'Citations per Faculty',
        'SCORE_ER': 'Employer Reputation', 
        'SCORE_FS': 'Faculty Student Ratio',
        'SCORE_EO': 'Employment Outcomes',
        'SCORE_IF': 'International Faculty',
        'SCORE_IRN': 'International Research Network',
        'SCORE_ISD': 'International Student Diversity',
        'SCORE_IS': 'International Students',
        'SCORE_ST': 'Sustainability'
    }
    
    print(f"\n📊 DIFFICULTY ANALYSIS BY INDICATOR:")
    print("-" * 80)
    print(f"{'Indicator':<25} {'Difficulty':<12} {'Current':<8} {'Mean':<8} {'Std':<8} {'CV':<8} {'Distance':<10}")
    print("-" * 80)
    
    for indicator in score_columns:
        if indicator in difficulty_metrics:
            metrics = difficulty_metrics[indicator]
            difficulty_level = "EASY" if metrics['difficulty_score'] <= 30 else "MEDIUM" if metrics['difficulty_score'] <= 60 else "HARD"
            print(f"{indicator_names.get(indicator, indicator):<25} {difficulty_level:<12} "
                  f"{metrics['current_score']:<8.1f} {metrics['mean_score']:<8.1f} "
                  f"{metrics['std_score']:<8.1f} {metrics['cv']:<8.2f} {metrics['distance_from_top']:<10.1f}")
    
    # Calculate optimal improvements based on difficulty and weights
    print(f"\n📊 OPTIMAL SCORE IMPROVEMENTS BY INDICATOR (Difficulty-Based):")
    print("-" * 80)
    
    # Calculate improvement allocation based on difficulty and weights
    total_improvement_needed = score_gap
    
    # Create improvement allocation factors
    allocation_factors = {}
    total_factor = 0
    
    for indicator, weight in indicator_weights.items():
        if indicator in difficulty_metrics:
            # Combine weight and difficulty (inverse relationship)
            # Higher weight = more important, lower difficulty = easier to improve
            difficulty = difficulty_metrics[indicator]['difficulty_score']
            ease_factor = (100 - difficulty) / 100  # Convert to ease (0-1)
            
            # Combined factor = weight * ease_factor
            combined_factor = weight * ease_factor
            allocation_factors[indicator] = combined_factor
            total_factor += combined_factor
    
    # Normalize allocation factors
    if total_factor > 0:
        for indicator in allocation_factors:
            allocation_factors[indicator] /= total_factor
    
    optimal_improvements = {}
    for indicator, weight in indicator_weights.items():
        if indicator in current_scores and indicator in allocation_factors:
            current_score = current_scores[indicator]
            
            # Calculate improvement based on allocation factor
            allocated_improvement = total_improvement_needed * allocation_factors[indicator]
            target_score = current_score + allocated_improvement
            
            # Ensure target score doesn't exceed 100
            target_score = min(target_score, 100.0)
            actual_improvement = target_score - current_score
            
            difficulty = difficulty_metrics[indicator]['difficulty_score']
            difficulty_level = "EASY" if difficulty <= 30 else "MEDIUM" if difficulty <= 60 else "HARD"
            
            optimal_improvements[indicator] = {
                'current': current_score,
                'target': target_score,
                'improvement': actual_improvement,
                'weight': weight,
                'difficulty': difficulty,
                'difficulty_level': difficulty_level,
                'allocation_factor': allocation_factors[indicator]
            }
            
            print(f"{indicator_names.get(indicator, indicator):<25} | "
                  f"Current: {current_score:>6.2f} | "
                  f"Target: {target_score:>6.2f} | "
                  f"Improvement: {actual_improvement:>+6.2f} | "
                  f"Weight: {weight*100:>3.0f}% | "
                  f"Difficulty: {difficulty_level:<6} | "
                  f"Allocation: {allocation_factors[indicator]*100:>5.1f}%")
    
    # Create visualization of current vs target scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Current vs Target Scores
    indicators = list(optimal_improvements.keys())
    current_values = [optimal_improvements[ind]['current'] for ind in indicators]
    target_values = [optimal_improvements[ind]['target'] for ind in indicators]
    
    x = np.arange(len(indicators))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, current_values, width, label='Current Score', 
                    color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, target_values, width, label='Target Score', 
                    color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('QS Indicators', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Current vs Target Scores for Top 150-100 Rank', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    # Create readable indicator names for charts
    chart_labels = {
        'SCORE_AR': 'Academic\nReputation',
        'SCORE_CPF': 'Citations per\nFaculty',
        'SCORE_ER': 'Employer\nReputation', 
        'SCORE_FS': 'Faculty Student\nRatio',
        'SCORE_EO': 'Employment\nOutcomes',
        'SCORE_IF': 'International\nFaculty',
        'SCORE_IRN': 'International\nResearch Network',
        'SCORE_ISD': 'International\nStudent Diversity',
        'SCORE_IS': 'International\nStudents',
        'SCORE_ST': 'Sustainability'
    }
    
    ax1.set_xticklabels([chart_labels.get(ind, ind) for ind in indicators], 
                        rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Chart 2: Required Improvements
    improvements = [optimal_improvements[ind]['improvement'] for ind in indicators]
    colors = ['#FF6B6B' if imp < 0 else '#4ECDC4' for imp in improvements]
    
    bars = ax2.bar(indicators, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('QS Indicators', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score Improvement Required', fontsize=12, fontweight='bold')
    ax2.set_title('Required Score Improvements for Top 150-100 Rank', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([chart_labels.get(ind, ind) for ind in indicators], 
                        rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                f'{height:+.1f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('khalifa_optimal_scores_prediction.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Priority recommendations based on difficulty and impact
    print(f"\n🎯 STRATEGIC RECOMMENDATIONS (Difficulty-Based):")
    print("-" * 80)
    
    # Sort improvements by a combination of difficulty, weight, and improvement magnitude
    def calculate_priority_score(data):
        # Priority factors:
        # 1. Ease of improvement (inverse of difficulty) - 40%
        # 2. QS weight importance - 30%
        # 3. Improvement magnitude - 20%
        # 4. Current score level - 10%
        
        ease_factor = (100 - data['difficulty']) / 100
        weight_factor = data['weight']
        improvement_factor = abs(data['improvement']) / 20  # Normalize to 0-1 range
        current_score_factor = data['current'] / 100
        
        priority_score = (ease_factor * 0.4 + 
                         weight_factor * 0.3 + 
                         improvement_factor * 0.2 + 
                         current_score_factor * 0.1)
        
        return priority_score
    
    sorted_improvements = sorted(optimal_improvements.items(), 
                                key=lambda x: calculate_priority_score(x[1]), reverse=True)
    
    print("Priority order for improvement (difficulty + impact based):")
    print(f"{'Rank':<4} {'Indicator':<25} {'Priority Score':<15} {'Difficulty':<10} {'Improvement':<12} {'Strategy'}")
    print("-" * 80)
    
    for i, (indicator, data) in enumerate(sorted_improvements, 1):
        priority_score = calculate_priority_score(data)
        
        # Determine strategy based on difficulty and improvement
        if data['difficulty'] <= 30 and data['improvement'] > 2:
            strategy = "🚀 QUICK WIN"
        elif data['difficulty'] <= 50 and data['weight'] > 0.1:
            strategy = "🎯 HIGH IMPACT"
        elif data['difficulty'] <= 60:
            strategy = "⚡ MODERATE EFFORT"
        else:
            strategy = "🏗️ LONG-TERM"
        
        print(f"{i:<4} {indicator_names.get(indicator, indicator):<25} "
              f"{priority_score:<15.3f} {data['difficulty_level']:<10} "
              f"{data['improvement']:>+8.2f} {strategy}")
    
    # Additional insights based on difficulty analysis
    print(f"\n📊 DIFFICULTY-BASED INSIGHTS:")
    print("-" * 60)
    
    easy_indicators = [ind for ind, data in optimal_improvements.items() if data['difficulty'] <= 30]
    hard_indicators = [ind for ind, data in optimal_improvements.items() if data['difficulty'] >= 70]
    
    if easy_indicators:
        print(f"🎯 EASY WINS (Difficulty ≤ 30):")
        for indicator in easy_indicators:
            data = optimal_improvements[indicator]
            print(f"   • {indicator_names.get(indicator, indicator)}: "
                  f"Current {data['current']:.1f}, Target {data['target']:.1f} "
                  f"(+{data['improvement']:.1f})")
    
    if hard_indicators:
        print(f"🏗️ CHALLENGING AREAS (Difficulty ≥ 70):")
        for indicator in hard_indicators:
            data = optimal_improvements[indicator]
            print(f"   • {indicator_names.get(indicator, indicator)}: "
                  f"Current {data['current']:.1f}, Target {data['target']:.1f} "
                  f"(+{data['improvement']:.1f}) - Requires long-term strategy")
    
    # Resource allocation recommendations
    print(f"\n💰 RESOURCE ALLOCATION RECOMMENDATIONS:")
    print("-" * 60)
    
    total_allocation = sum(data['allocation_factor'] for data in optimal_improvements.values())
    
    for indicator, data in optimal_improvements.items():
        allocation_percentage = (data['allocation_factor'] / total_allocation) * 100
        print(f"   • {indicator_names.get(indicator, indicator)}: "
              f"{allocation_percentage:.1f}% of improvement effort "
              f"({data['difficulty_level']} difficulty)")
    
    # Calculate feasibility score
    total_improvement_needed = sum(abs(data['improvement']) for data in optimal_improvements.values())
    feasibility_score = max(0, 100 - total_improvement_needed)
    
    print(f"\n📊 FEASIBILITY ASSESSMENT:")
    print("-" * 60)
    print(f"Total improvement needed: {total_improvement_needed:.2f} points")
    print(f"Feasibility score: {feasibility_score:.1f}/100")
    
    if feasibility_score >= 70:
        print("🎉 HIGH FEASIBILITY: Target rank 100-150 is achievable with focused efforts")
    elif feasibility_score >= 40:
        print("⚠️  MODERATE FEASIBILITY: Target requires significant improvements but is possible")
    else:
        print("🚨 LOW FEASIBILITY: Target rank may be too ambitious for 2026-2027")
    
    print(f"\n" + "="*80)
    print("PREDICTION COMPLETED - Chart saved as 'khalifa_optimal_scores_prediction.png'")
    print("="*80)

# Example usage:
if __name__ == "__main__":
    # analyze_top_150_qs('cleaned_dataset/qs_2025_2026.csv')
    # highest_rank_jump('cleaned_dataset/qs_2025_2026.csv')
    # compare_khalifa_ranks('cleaned_dataset/qs_2025_2026.csv', 'cleaned_dataset/qs_2024_2025.csv')
    # create_spider_chart('cleaned_dataset/qs_2025_2026.csv')
    
    # # Calculate median for all four years and store results
    # historical_medians = {}
    # median_2022, _ = calculate_median_100_150('cleaned_dataset/qs_2022_2023.csv', '2022-2023')
    # historical_medians['2022-2023'] = median_2022
    
    # median_2023, _ = calculate_median_100_150('cleaned_dataset/qs_2023_2024.csv', '2023-2024')
    # historical_medians['2023-2024'] = median_2023
    
    # median_2024, _ = calculate_median_100_150('cleaned_dataset/qs_2024_2025.csv', '2024-2025')
    # historical_medians['2024-2025'] = median_2024
    
    # median_2025, _ = calculate_median_100_150('cleaned_dataset/qs_2025_2026.csv', '2025-2026')
    # historical_medians['2025-2026'] = median_2025
    
    # # Predict 2026-2027 median
    # predicted_2026_2027, _ = predict_future_median(historical_medians)
    
    # # Calculate indicator coefficients
    # calculate_indicator_coefficients('cleaned_dataset/qs_2025_2026.csv')
    
    # # Create what-if scenarios using predicted 2026-2027 median
    # create_what_if_scenarios('cleaned_dataset/qs_2025_2026.csv', 
    #                        prev_filepath='cleaned_dataset/qs_2024_2025.csv',
    #                        predicted_median=predicted_2026_2027)
    
    # # Peer benchmarking analysis and visualization
    # benchmark_peers = [
    #     ("Universiti Putra Malaysia", "Malaysia"),
    #     ("Qatar University", "Qatar"),
    #     ("United Arab Emirates University", "United Arab Emirates"),
    #     ("King Saud University", "Saudi Arabia")
    # ]
    # create_peer_comparison_chart('cleaned_dataset/qs_2025_2026.csv', benchmark_peers)
    
    # # Recommend specific peer universities to study
    # recommend_peer_examples('cleaned_dataset/qs_2025_2026.csv', benchmark_peers)
    
    # # Create peer learning table
    # create_peer_learning_table('cleaned_dataset/qs_2025_2026.csv', benchmark_peers)
    
    # # Create historical rank chart
    # create_historical_rank_chart()
    
    # Create ranking employment correlation
    #create_ranking_employment_correlation()
    
    # Create Malaysia UAE trends
    # create_malaysia_uae_trends()
    
    # Predict optimal scores for top 150-100 ranks
    predict_optimal_scores_for_top_150_100()