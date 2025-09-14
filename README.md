# College Football Team Success Predictor
A machine learning application that predicts college football success levels based on statistical performance metrics using K-Nearest Neighbors classification.

## Overview
This project analyzes college football team statistics from a given season to predict team success in three categories.
- **Poor** (<50% win rate)
- **Good** (50-75% win rate)
- **Excellent** (>75% win rate)
The model uses 10 key performance indicators including offensive and defensive rankings, yards per game, points scored/allowed, turnover margins, and third-down conversion rates

## Features
- **Data processing**: Automated cleaning and feature engineering
- **Model Training**: K-Nearest Neighbors classification with hyperparameter optimization
- **Predictions**: Predict success for teams in the dataset or custom team statistics
- **Visualizations**: Distribution chart and confusion matrix heatmap
- **Interactive Interface**: Command line interface for team predictions

## Requirements
### Python Version
- Python 3.8 - 3.12
- Tested on Python 3.10 - 3.12

## Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Installation
1. Clone the repository
```bash
git clone https://github.com/notwilliam1/FootballML.git
cd FootballML
```
2. Create a venv (recommended):
```bash
python -m venv .venv
```
3. Activate the venv
    * Windows: `.venv\Scripts\activate`
    * macOS/Linux: `source .venv/bin/activate`
4. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Requirements
The application expects a CSV file located at `/FootballML/Data` with the following columns. You can adjust the filepath to fit your data if you decide to use a different CSV file.
### Required Columns:
- `Team`-Team name
- `Win-Loss` - Season record(format:"W-L", e.g., "12-1")
- `Off Rank` - Offensive ranking
- `Def Rank` - Defensive ranking
- `Off Yards Per Game`
- `Yards Per Game Allowed`
- `Points Per Game`
- `Avg Points Per Game Allowed`
- `Turnover Margin`
- `Total Tackle For Loss`
- `3rd Percent` - Third down conversion %
- `Opponent 3rd Percent`

## Troubleshooting
### Common Issues
1. Import Errors: Ensure all dependencies are installed and the venv is activated
2. File Not Found: verify the data file exists at `/FootballML/Data`
3. Python Version: Make sure you are **NOT** using Python 3.13 (recommended 3.12)

### Data Path Config
To change the data file location, modify the `path` variable in `main.py`:
```python
path = 'your custom path to data'
```
## Future Enhancements
- Support for multiple seasons of data
- Additional machine learning algorithms
- Web-based interface
- Real-time data integration
- Model performance tracking over time



