import src.data_processing as dp
import src.model_training as mt
import src.predictions as pred
import src.visualizations as viz

def main():
    # Data processing
    path = 'E:/FootballML/Data'
    data = dp.load_and_clean_data(path)
    data = dp.create_target_variable(data)
    X, y, available_features = dp.prepare_features(data)
    
    # Model training
    X_train, X_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler = mt.split_and_scale_data(X, y)
    final_model, accuracy, cm = mt.train_model(x_train_scaled, y_train, x_test_scaled, y_test)
    
    # Visualizations
    viz.create_visuals(data, cm, accuracy)
    
    # Predictions
    pred.predict_specific_teams(data, available_features, final_model, scaler)
    pred.show_manual_prediction_example()
    pred.show_available_teams(data)
    
    print("\n" + "=" * 40)
    print("Model Ready!")
    print("=" * 40)

    # User input 
    print()

    user_team_name = input("Enter a team name (case sensitive) to predict (or 'exit' to quit): ")

    while user_team_name != 'exit':
        pred.user_input_prediction(data, available_features, final_model, scaler, user_team_name)
        user_team_name = input("\nEnter another team name (or 'exit' to quit): ")

if __name__ == "__main__":
    main()

