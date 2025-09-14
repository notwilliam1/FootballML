import pandas as pd
import re

def predict_team_success(final_model, scaler, off_rank, def_rank, off_ypg, def_ypg_allowed,
                         ppg, ppg_allowed, turnover_margin, tfl,
                         third_down_pct, opp_third_down_pct):

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

def predict_existing_team(data, available_features, final_model, scaler, team_name):

    try:
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
        print("2023 Stats:")
        print(team_stats.to_string())

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

def predict_specific_teams(data, available_features, final_model, scaler):

    print("\n" + "=" * 40)
    print("Predicting specific teams for 2025")
    print("=" * 40)

    teams_to_predict = ['Georgia Tech', 'Georgia', 'LSU', 'Ohio St.', 'Clemson']

    for team in teams_to_predict:
        team_exists = data['Team'].str.contains(team, case = False, na = False).any()
        if team_exists:
            predict_existing_team(data, available_features, final_model, scaler, team)
        else:
            print(f"\n'{team}' not found in dataset")

def user_input_prediction(data, available_features, final_model, scaler, user_team_name):

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
            predict_existing_team(data, available_features, final_model, scaler, team_name)
        else:
            print("Invalid selection number.")
    except ValueError:
        print("Invalid input. Please enter a number or 'cancel'.")

    return

def show_manual_prediction_example():

    print("\n" + "=" * 40)
    print("Manual Prediction Example")
    print("=" * 40)
    print("You can also predict any team manually using their stats:")
    print("predict_team_success(off_rank=10, def_rank=15, off_ypg=450, def_ypg_allowed=300,")
    print("ppg=35, ppg_allowed=20, turnover_margin=1.5, tfl=80,") 
    print("third_down_pct=0.45, opp_third_down_pct=0.30)")

def show_available_teams(data):

    print(f"\nAvailable teams in dataset ({len(data)} total):")
    for i, team in enumerate(data['Team'].head(10)):
        print(f"{i+1}. {team}")
    print("... and more")
