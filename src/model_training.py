from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def split_and_scale_data(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Training set: {X_train.shape[0]} teams")
    print(f"Test set: {X_test.shape[0]} teams")

    print("\n" + "=" * 40)
    print("Scaling Features")
    print("=" * 40)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    print("Features scaled to 0-1 range")
    
    return X_train, X_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler   

def train_model(x_train_scaled, y_train, x_test_scaled, y_test):
    
    print("\n" + "=" * 40)
    print("Training the Model")
    print("=" * 40)

    k_values = [3, 5, 7, 9]
    best_k = 3
    best_accuracy = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train_scaled, y_train)

        train_pred = knn.predict(x_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)

        print(f"k={k}: Training accuracy = {train_accuracy:.3f}")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_k = k

    print(f"\nBest k value: {best_k}")

    final_model = KNeighborsClassifier(n_neighbors = best_k)
    final_model.fit(x_train_scaled, y_train)

    y_pred = final_model.predict(x_test_scaled)

    print("\n" + "=" * 40)
    print("Model Results")
    print("=" * 40)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    for i in range(3):
        category_accuracy = cm[i][i] / sum(cm[i]) if sum(cm[i]) > 0 else 0
        labels = ["Poor", "Good", "Excellent"]
        print(f"{labels[i]} teams predicted correctly: {category_accuracy:.3f}")
    
    return final_model, accuracy, cm
