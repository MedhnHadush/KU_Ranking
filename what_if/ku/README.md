# Khalifa University Ranking Analyzer

A Django web application that provides interactive what-if scenario analysis for Khalifa University's QS World University Rankings performance.

## Features

- **Dual Dataset Analysis**: Switch between current QS rankings and projected scores
- **Interactive Sliders**: Adjust 10 different performance indicators using intuitive sliders
- **Real-time Analysis**: See how changes in scores affect KU's ranking position
- **Beautiful Tabbed UI**: Modern, responsive design with Bootstrap 5 and custom styling
- **Comparative View**: Shows surrounding universities in the new ranking
- **Ranking Impact**: Visualizes whether the rank improves, declines, or stays the same

## Dataset Tabs

### 📊 Current Rankings Tab
- **Data Source**: Official QS World University Rankings 2025-2026
- **Universities**: 1,501 institutions worldwide
- **KU Position**: #177 globally
- **Features**: Complete ranking analysis with rank change indicators

### 🔮 Projected Scores Tab
- **Data Source**: Projected performance scores based on modeling
- **Universities**: 599 institutions with projected indicators
- **Status**: KU data not available in this dataset
- **Features**: Shows informative message and redirects to current rankings

## Performance Indicators

The application includes sliders for all key QS ranking indicators:

1. **Academic Reputation** (30% weight)
2. **Employer Reputation** (15% weight)
3. **Faculty Student Ratio** (10% weight)
4. **Citations per Faculty** (20% weight)
5. **International Faculty** (5% weight)
6. **International Students** (5% weight)
7. **International Student Diversity** (0% weight)
8. **International Research Network** (5% weight)
9. **Employment Outcomes** (5% weight)
10. **Sustainability** (5% weight)

## Current Status

- **KU Current Rank**: #177 (2026 QS Rankings)
- **Overall Score**: 58.8
- **Country**: United Arab Emirates
- **Region**: Asia

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ku
   ```

2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run migrations**:
   ```bash
   python3 manage.py makemigrations
   python3 manage.py migrate
   ```

4. **Start the development server**:
   ```bash
   python3 manage.py runserver
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

## Data Sources

The application uses two datasets:

### Primary Dataset: QS Rankings 2025-2026 (`qs_2025_2026.csv`)
- 1,501 universities worldwide
- Complete ranking data with all performance indicators
- Khalifa University's current performance metrics
- Full country and regional information

### Secondary Dataset: Projected Scores (`projected_scores (1).csv`)
- 599 universities with projected performance indicators
- Score projections based on performance modeling
- Note: Khalifa University is not included in this dataset

## How It Works

### Current Rankings Analysis
1. **Load Current Data**: The application loads KU's current scores from the QS CSV file
2. **Interactive Adjustment**: Users can modify scores using sliders (0-100 range)
3. **Recalculation**: The system recalculates overall scores using updated QS weighting methodology
4. **Ranking Comparison**: All universities are re-ranked based on the new calculated scores
5. **Results Display**: Shows KU's new position and surrounding universities with rank change indicators

### Projected Scores Analysis
1. **Dataset Check**: Verifies if KU data is available in projected scores
2. **Graceful Handling**: Shows informative message when data is not available
3. **User Guidance**: Provides clear direction to use current rankings for analysis

## Technical Implementation

- **Backend**: Django 5.2.4 with pandas for data processing
- **Frontend**: Bootstrap 5 with custom CSS and JavaScript
- **Database**: SQLite for development (models included for future enhancements)
- **Data Processing**: Pandas DataFrame operations for efficient ranking calculations
- **UI/UX**: Tabbed interface for seamless dataset switching

## Key Files

- `ranking_analyzer/views.py`: Main application logic and ranking calculations
- `ranking_analyzer/templates/ranking_analyzer/home.html`: Interactive web interface with tabs
- `ranking_analyzer/models.py`: Database models for university data
- `qs_2025_2026.csv`: Source data with all university rankings
- `projected_scores (1).csv`: Projected performance scores dataset

## Updated Ranking Calculation

The application uses an updated weighting system based on QS methodology:

```python
score_weights = {
    'SCORE_AR': 0.3,   # Academic Reputation - 30%
    'SCORE_ER': 0.15,  # Employer Reputation - 15%
    'SCORE_FS': 0.10,  # Faculty Student Ratio - 10%
    'SCORE_CPF': 0.2,  # Citations per Faculty - 20%
    'SCORE_IF': 0.05,  # International Faculty - 5%
    'SCORE_IS': 0.05,  # International Students - 5%
    'SCORE_ISD': 0.00, # International Student Diversity - 0%
    'SCORE_IRN': 0.05, # International Research Network - 5%
    'SCORE_EO': 0.05,  # Employment Outcomes - 5%
    'SCORE_ST': 0.05,  # Sustainability - 5%
}
```

## Usage Examples

### Current Rankings Analysis
- **Scenario 1**: Improving Academic Reputation from 34.3 to 50.0
- **Scenario 2**: Enhancing International Profile across multiple indicators
- **Scenario 3**: Research Excellence through Citations per Faculty improvements

### Projected Scores Analysis
- Currently not available for KU (dataset limitation)
- Interface gracefully handles missing data
- Provides clear guidance to users

## New Features Added

### 🆕 Tabbed Interface
- Clean separation between current and projected data
- Visual indicators for data availability
- Smooth transitions between datasets

### 🆕 Enhanced Error Handling
- Graceful handling of missing data
- Informative messages for unavailable datasets
- User guidance for optimal experience

### 🆕 Improved Data Management
- Dual dataset support in backend
- Flexible data loading based on user selection
- Robust JSON handling for frontend

## Future Enhancements

- **Extended Projected Data**: Include KU in projected scores dataset
- **Historical Analysis**: Track ranking changes over multiple years
- **Peer Comparison**: Compare with similar institutions
- **Target Setting**: Set specific ranking goals and required improvements
- **Export Functionality**: Save scenarios and results
- **Advanced Analytics**: Statistical analysis of ranking factors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and analytical purposes. QS ranking data is used under fair use for academic analysis.

## Support

For technical issues or questions about the ranking analyzer, please create an issue in the repository.

---

**Note**: This tool provides estimated rankings based on simplified calculations. Actual QS rankings involve complex methodologies and additional factors not captured in this simulation. 