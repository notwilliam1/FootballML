#Imports
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

path = 'E:/FootballML/Data'
data = pd.read_csv(path + "/cfb23.csv")

print("Dataset shape:", data.shape)
print("First few rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum().sum(), "total missing values")

print("\n" + "=" * 40)
print("Cleaning the Data")
print("=" * 40)

data = data.drop(data.columns[0], axis = 1)

# Create target variable
def get_success_lvl(win_loss_record):
    """Convert W-L record into success level"""
    try:
        wins, losses = win_loss_record.split('-')
        wins = int(wins)
        losses = int(losses)

        win_percentage = wins / (wins + losses)

        if win_percentage > 0.75:
            return 2 # Excellent team
        elif win_percentage > 0.50:
            return 1 # Good team
        else:
            return 0 # Poor team
    except:
        return 1 # Default
    
data['Team_Success'] = data['Win-Loss'].apply(get_success_lvl)

print("Success levels:")
print("0 = Poor (< 50% wins)")
print("1 = Good (50-75% wins)")
print("2 = Excellent (> 75% wins)")
print("\nDistribution:")
print(data['Team_Success'].value_counts().sort_index())

# Select features for model
important_features = [
    'Off Rank', 'Def Rank', 'Off Yards per Game', 'Yards Per Game Allowed',
    'Points Per Game', 'Avg Points per Game Allowed', 'Turnover Margin',
    'Total Tackle For Loss', '3rd Percent', 'Opponent 3rd Percent'
]

available_features = [col for col in important_features if col in data.columns]
print(f"\nUsing {len(available_features)} features:", available_features)

# Prepare data
X = data[available_features].copy() # Features
y = data['Team_Success'] # Target

# Clean data
print("Cleaning feature data...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors = 'coerce')

# Handle missing values
X = X.fillna(X.median())

print(f"\nFinal data shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

print(f"Training set: {X_train.shape[0]} teams")
print(f"Test set: {X_test.shape[0]} teams")

# Scale features to similar ranges
print("\n" + "=" * 40)
print("Scaling Features")
print("=" * 40)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

print("Features scaled to 0-1 range")

print("\n" + "=" * 40)
print("Training the Model")
print("=" * 40)

# Try values for k
k_values = [3, 5, 7, 9]
best_k = 3
best_accuracy = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train_scaled, y_train)

    # Test on training data to pick best k
    train_pred = knn.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)

    print(f"k={k}: Training accuracy = {train_accuracy:.3f}")

    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        best_k = k

print(f"\nBest k value: {best_k}")

# Train final model with best k
final_model = KNeighborsClassifier(n_neighbors = best_k)
final_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = final_model.predict(x_test_scaled)

print("\n" + "=" * 40)
print("Model Results")
print("=" * 40)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("Rows =  Actual, Columns = Predicted")
print("     Poor  Good  Excellent")
for i, row in enumerate(cm):
    labels = ["Poor", "Good", "Excellent"]
    print(f"{labels[i]:>8} {row[0]:4d} {row[1]:4d} {row[2]:8d}")

# Calculate accuracy
for i in range(3):
    category_accuracy = cm[i][i] / sum(cm[i]) if sum(cm[i]) > 0 else 0
    labels = ["Poor", "Good", "Excellent"]
    print(f"{labels[i]} teams predicted correctly: {category_accuracy:.3f}")

print("\n" + "=" * 40)
print("Creating Visuals")
print("=" * 40)

# Create plots
fig, axes = plt.subplots(1, 2, figsize = (12, 5))

# Plot 1: Distribution of success lvls
success_counts = data['Team_Success'].value_counts().sort_index()
bars = axes[0].bar(['Poor', 'Good', 'Excellent'], success_counts.values,
                   color=['red', 'orange', 'green'], alpha = 0.7)
axes[0].set_title("Distribution of Team Success Levels")
axes[0].set_ylabel("Number of Teams")

# Add numbers on top of bars
for bar, count in zip(bars, success_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              str(count), ha = 'center', va = 'bottom')
    
# Plot 2: Confusion matrix heatmap
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', ax = axes[1],
            xticklabels = ['Poor', 'Good', 'Excellent'],
            yticklabels = ['Poor', 'Good', 'Excellent'])
axes[1].set_title(f"Confusion Matrix (Accuracy: {accuracy:.3f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

def predict_team_success(off_rank, def_rank, off_ypg, def_ypg_allowed,
                         ppg, ppg_allowed, turnover_margin, tfl,
                         third_down_pct, opp_third_down_pct):
    """
    Predict how successful a team will be based on specific stats
    """

    # Create data point with input stats
    new_team = [[off_rank, def_rank, off_ypg, def_ypg_allowed, ppg,
                 ppg_allowed, turnover_margin, tfl, third_down_pct, opp_third_down_pct]]
    
    new_team_scaled = scaler.transform(new_team)

    prediction = final_model.predict(new_team_scaled)[0]
    probabilities = final_model.predict_proba(new_team_scaled)[0]

    labels = ['Poor', 'Good', 'Excellent']
    confidence = probabilities[prediction]

    print(f"Predicition: {labels[prediction]} team")
    print(f"Confidence: {confidence:.3f}")

    return prediction, confidence

# Predict specific teams from dataset
print("\n" + "=" * 40)
print("Predicting specific teams for 2025")
print("=" * 40)

def predict_existing_team(team_name):
    """Predict succes for team in dataset"""
    try:
        # Find team
        escaped_team_name = re.escape(team_name)
        team_row = data[data['Team'].str.contains(escaped_team_name, case = False, na = False)]

        if team_row.empty:
            print(f"Team '{team_name}' not found in dataset")
            return None
        
        # Get 2023 stats
        team_stats = team_row[available_features].iloc[0]
        actual_success = team_row['Team_Success'].iloc[0]
        actual_record = team_row['Win-Loss'].iloc[0]

        print(f"\n{team_row['Team'].iloc[0]}:")
        print(f"2023 Record: {actual_record}")
        print(f"2023 Actual Success Level: {['Poor', 'Good', 'Excellent'][actual_success]}")

        team_stats_numeric = pd.to_numeric(team_stats, errors = 'coerce')

        clean_stats = team_stats_numeric.fillna(team_stats_numeric.median()).astype(float).values
        clean_stats_df = pd.DataFrame([clean_stats], columns = available_features)

        team_scaled = scaler.transform(clean_stats_df)

        prediction = final_model.predict(team_scaled)[0]
        probabilities = final_model.predict_proba(team_scaled)[0]

        labels = ['Poor', 'Good', 'Excellent']
        print(f"2025 Prediction: {labels[prediction]}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print("All probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  {labels[i]}: {prob:.3f}")

        return prediction, probabilities
    
    except Exception as e:
        print(f"Error predicting {team_name}: {e}")
        return None

# Enter teams to predict
teams_to_predict = ['Georgia Tech', 'Georgia', 'LSU', 'Ohio State', 'Clemson']

for team in teams_to_predict:
    team_exists = data['Team'].str.contains(team, case = False, na = False).any()
    if team_exists:
        predict_existing_team(team)
    else:
        print(f"\n'{team}' not found in dataset")

print("\n" + "=" * 40)
print("Manual Prediction Example")
print("=" * 40)
print("You can also predict any team manually using their stats:")
print("predict_team_success(off_rank=10, def_rank=15, off_ypg=450, def_ypg_allowed=300,")
print("ppg=35, ppg_allowed=20, turnover_margin=1.5, tfl=80,") 
print("third_down_pct=0.45, opp_third_down_pct=0.30)")

# Show available teams
print(f"\nAvailable teams in dataset ({len(data)} total):")
for i, team in enumerate(data['Team'].head(10)):
    print(f"{i+1}. {team}")
print("... and more")

print("\n" + "=" * 40)
print("Model Ready!")
print("=" * 40)

# User input 
def user_input_prediction(user_team_name):
    """Get user input to predict a team"""
    print(f"Searching for teams containing '{user_team_name}'...")

    matching_teams = data[data['Team'].str.contains(user_team_name, case = False, na = False)]

    if matching_teams.empty:
        print(f"No teams found matching '{user_team_name}'")
        return
    
    print(f"Found {len(matching_teams)} matching teams:")
    for i, team in enumerate(matching_teams['Team'].values, 1):
        print(f"{i}. {team}")

    selection = input("Enter the number of the team to predict (or 'cancel' to skip): ")
    if selection.lower() == 'cancel':
        print("Prediction cancelled.")
        return
    try:
        selection = int(selection)
        if 1 <= selection <= len(matching_teams):
            team_name = matching_teams['Team'].values[selection - 1]
            predict_existing_team(team_name)
        else:
            print("Invalid selection number.")
    except ValueError:
        print("Invalid input. Please enter a number or 'cancel'.")

    return

print()

user_team_name = input("Enter a team name (case sensitive) to predict (or 'exit' to quit): ")

while user_team_name != 'exit':
    user_input_prediction(user_team_name)
    user_team_name = input("\nEnter another team name (or 'exit' to quit): ")

