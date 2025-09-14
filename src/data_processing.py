import pandas as pd
import numpy as np

def load_and_clean_data(path):
    
    data = pd.read_csv(path + "/cfb23.csv")

    print("Dataset shape:", data.shape)
    print("First few rows:")
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum().sum(), "total missing values")

    print("\n" + "=" * 40)
    print("Cleaning the Data")
    print("=" * 40)

    data = data.drop(data.columns[0], axis=1)

    return data

def get_success_lvl(win_loss_record):

    try: 
        wins, losses = win_loss_record.split('-')
        wins = int(wins)
        losses = int(losses)

        win_percentage = wins / (wins + losses)

        if win_percentage >= 0.75:
            return 2
        elif win_percentage >= 0.5:
            return 1
        else:
            return 0
    except:
        return 1
    
def create_target_variable(data):

    data['Team_Success'] = data['Win-Loss'].apply(get_success_lvl)

    print("Success levels:")
    print("0: Poor (<50% wins)")
    print("1: Good (50-75% wins)")
    print("2: Excellent (>75% wins)")
    print("\nDistribution:")
    print(data['Team_Success'].value_counts().sort_index())

    return data

def prepare_features(data):

    important_features = [
        'Off Rank', 'Def Rank', 'Off Yards per Game', 'Yards Per Game Allowed',
        'Points Per Game', 'Avg Points per Game Allowed', 'Turnover Margin',
        'Total Tackle For Loss', '3rd Percent', 'Opponent 3rd Percent'       
    ]
    
    available_features = [col for col in important_features if col in data.columns]
    print(f"\nUsing {len(available_features)} features:", available_features)

    X = data[available_features].copy()
    y = data['Team_Success']

    print("Cleaning feature data...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(X.median())

    print(f"Final data shape: {X.shape}")

    return X, y, available_features
    