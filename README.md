# QS World University Rankings Analysis (Code Only)

This repository contains Python code for analyzing QS World University Rankings data, with a focus on Khalifa University and peer benchmarking. The code provides tools for indicator analysis, peer comparison, and strategic recommendations.

## Features
- Analyze QS ranking data (2017-2026)
- Benchmark Khalifa University against peers
- Predict optimal indicator improvements for target ranks
- Visualize trends and gaps

## Quick Start
1. Install requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place your QS CSV data in the same directory (not included here).
3. Run the main analysis:
   ```python
   from qs_analysis import predict_optimal_scores_for_top_150_100
   predict_optimal_scores_for_top_150_100()
   ```

## Main File
- `qs_analysis.py` — All analysis and visualization functions

## Note
- No datasets are included in this repository.
- You must supply your own QS data in CSV format.

## License
For academic and research use. Please cite appropriately. 
