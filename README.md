# QS World University Rankings Analysis - Code Repository

## 📊 Project Overview

This repository contains the Python analysis code for comprehensive QS World University Rankings analysis, specifically designed for Khalifa University strategic planning. The code provides advanced analytics, predictive modeling, and strategic recommendations for achieving top 150-100 rankings.

## 🎯 Key Features

### 1. **Advanced Data Analysis**
- Historical QS rankings analysis (2017-2026)
- Peer university benchmarking
- Indicator performance analysis
- Trend identification and forecasting

### 2. **Strategic Planning Tools**
- **Difficulty-based improvement recommendations** - Analyzes improvement feasibility across the entire dataset
- What-if scenario analysis
- Optimal score predictions for target rankings
- Resource allocation strategies

### 3. **Predictive Analytics**
- Linear regression modeling
- Future ranking predictions
- Indicator coefficient analysis
- Feasibility assessments

### 4. **Visualization Suite**
- Spider charts for indicator performance
- Historical ranking trends
- Peer comparison charts
- Employment correlation analysis
- Female research trends

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Basic Usage
```python
# Import the analysis module
from qs_analysis import predict_optimal_scores_for_top_150_100

# Run comprehensive difficulty-based analysis
predict_optimal_scores_for_top_150_100()
```

## 📈 Core Analysis Functions

### 1. **Difficulty-Based Optimal Score Prediction**
```python
predict_optimal_scores_for_top_150_100()
```
**Features:**
- Analyzes improvement difficulty across the entire dataset
- Uses sophisticated difficulty assessment algorithm
- Provides strategic recommendations based on feasibility
- Generates visualizations of current vs target scores

**Difficulty Assessment Factors:**
- Current performance level (30% weight)
- Score variability across universities (20% weight)
- Distance from top performers (25% weight)
- Current score level (25% weight)

### 2. **Top 150 Analysis**
```python
analyze_top_150_qs('path_to_qs_data.csv')
```
- Analyzes top 150 universities
- Compares target university performance
- Identifies gaps and opportunities

### 3. **Peer Benchmarking**
```python
benchmark_peers = [
    ("Universiti Putra Malaysia", "Malaysia"),
    ("Qatar University", "Qatar"),
    ("United Arab Emirates University", "United Arab Emirates"),
    ("King Saud University", "Saudi Arabia")
]
create_peer_comparison_chart('path_to_qs_data.csv', benchmark_peers)
```

### 4. **Historical Trends Analysis**
```python
create_historical_rank_chart()  # 2022-2026 ranking trends
create_malaysia_uae_trends()    # Regional comparison
```

### 5. **Strategic Scenario Analysis**
```python
create_what_if_scenarios('current_data.csv', 'previous_data.csv')
```

## 🎯 Strategic Insights

### Difficulty-Based Improvement Strategy
The analysis uses a sophisticated algorithm that considers:

1. **Current Performance Level** (30% weight)
   - Position relative to global distribution
   - Distance from top performers

2. **Score Variability** (20% weight)
   - Coefficient of variation across universities
   - Ease of standing out in the indicator

3. **Distance from Top** (25% weight)
   - Gap to top 10% performers
   - Realistic improvement potential

4. **Current Score Level** (25% weight)
   - Absolute performance level
   - Room for improvement

### Priority Categories
- **🚀 QUICK WINS**: Easy improvements (Difficulty ≤ 30)
- **🎯 HIGH IMPACT**: Important indicators with moderate difficulty
- **⚡ MODERATE EFFORT**: Balanced difficulty and impact
- **🏗️ LONG-TERM**: Challenging areas requiring sustained effort

## 📊 QS 2026 Methodology Support

The code supports the official 2026 QS World University Rankings methodology:

- **Academic Reputation (30%)**: Global academic survey
- **Citations per Faculty (20%)**: Research impact measurement
- **Employer Reputation (15%)**: Graduate employability survey
- **Faculty Student Ratio (10%)**: Teaching quality indicator
- **Employment Outcomes (5%)**: Graduate employment rates
- **International Faculty (5%)**: International diversity
- **International Research Network (5%)**: Research collaboration
- **International Student Diversity (0%)**: Student diversity (not weighted in 2026)
- **International Students (5%)**: International student ratio
- **Sustainability (5%)**: Environmental and social impact

## 🔧 Technical Details

### Data Processing
- Automatic cleaning of rank columns (removing '=' symbols)
- Handling of rank ranges (using midpoints)
- Missing data imputation and validation
- Cross-year data consistency checks

### Statistical Methods
- Linear regression for trend analysis
- Correlation analysis for indicator relationships
- Quantile-based difficulty assessment
- Weighted improvement allocation

### Visualization Features
- Interactive spider charts
- Multi-year trend analysis
- Peer comparison visualizations
- Employment correlation charts

## 📋 Usage Examples

### Basic Analysis
```python
# Run comprehensive analysis
predict_optimal_scores_for_top_150_100()

# Generate specific visualizations
create_spider_chart('path_to_qs_data.csv')
create_historical_rank_chart()
```

### Custom Peer Analysis
```python
# Define custom peer universities
custom_peers = [
    ("Your University", "Country"),
    ("Another University", "Country")
]

# Run peer benchmarking
create_peer_comparison_chart('path_to_qs_data.csv', custom_peers)
```

### Trend Analysis
```python
# Analyze specific trends
create_malaysia_uae_trends()
create_ranking_employment_correlation()
```

## 🎓 Academic Applications

This analysis is particularly useful for:
- **University Strategic Planning**: Data-driven decision making
- **Performance Benchmarking**: Peer comparison and gap analysis
- **Resource Allocation**: Evidence-based investment decisions
- **Goal Setting**: Realistic target setting with feasibility assessment
- **Research**: Academic studies on university rankings and performance

## 📈 Key Outputs

### 1. **Difficulty Analysis Table**
Shows difficulty level, current scores, and improvement potential for each indicator.

### 2. **Optimal Score Predictions**
Provides target scores and required improvements for each indicator.

### 3. **Strategic Recommendations**
Prioritized list of improvement areas with feasibility assessment.

### 4. **Visualizations**
- Current vs target score comparisons
- Required improvement charts
- Historical trend analysis
- Peer benchmarking charts

## 🔮 Future Enhancements

### Planned Features
- Machine learning-based prediction models
- Real-time data integration
- Interactive dashboard
- Multi-year scenario planning
- Regional benchmarking tools

### Data Integration
- Additional ranking systems (THE, ARWU)
- Financial performance indicators
- Research output metrics
- Student satisfaction data

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Improve documentation
- Add new analysis methods

## 📄 License

This project is for educational and research purposes. Please ensure proper attribution when using the analysis or code.

## 📞 Contact

For questions or collaboration opportunities, please contact the project maintainers.

---

**Note**: This code requires QS World University Rankings data in CSV format. The analysis is based on publicly available QS data and supplementary datasets. Results should be interpreted in the context of the specific methodology and data limitations. 