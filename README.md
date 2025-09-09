# College Football Success Predictor 🏈

A machine learning model that predicts college football team success based on 2023 statistical data.

## 🎯 Features

- Predicts team success categories: Poor (<50% wins), Good (50-75% wins), Excellent (>75% wins)
- Uses K-Nearest Neighbors algorithm with optimized hyperparameters
- Includes data visualization and model performance metrics
- Can predict both existing teams from dataset and new teams with manual stats
- The model uses these 10 key statistical features: Offensive Rank, Defensive Rank, Offensive Yards per game, Yards per Game Allowed, Points per Game,
    Average Points per Game Allowed, Turnover Margin, Total Tackle for Loss, 3rd Down Conversion %, Opponent 3rd Down Conversion %

## 📊 Model Performance

- **Test Accuracy**: 61%
- **Best for**: Identifying poor and good teams (73% accuracy each)
- **Challenge**: Distinguishing between good and excellent teams

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/notwilliam1/FootballML.git
cd FootballML

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.