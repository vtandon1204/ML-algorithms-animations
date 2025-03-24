from algorithms.linear_regression import LinearRegression
from algorithms.logistic_regression import LogisticRegression
from algorithms.knn import KNN
from algorithms.k_means import KMeans
from algorithms.dbscan import DBSCAN  # Assuming DBSCAN is implemented here
from utils.data_generation import generate_classification_data

def plot_linear_regression():
    print("Running Linear Regression Animation...")
    model = LinearRegression(learning_rate=0.01, no_of_iterations=100)
    X, y = model.generate_data()
    model.fit(X, y)
    # Code to plot or save animation

def plot_logistic_regression():
    print("Running Logistic Regression Animation...")
    model = LogisticRegression(learning_rate=0.01, no_of_iterations=100)
    X, y = generate_classification_data()
    model.fit(X, y)
    # Code to plot or save animation

def plot_knn():
    print("Running KNN Animation...")
    model = KNN(k=3)
    X, y = generate_classification_data()
    model.fit(X, y)
    # Code to plot or save animation

def plot_k_means():
    print("Running K-Means Animation...")
    model = KMeans(n_clusters=3)
    X, _ = generate_classification_data(n_features=2)
    model.fit(X)
    # Code to plot or save animation

def plot_dbscan():
    print("Running DBSCAN Animation...")
    model = DBSCAN(eps=0.5, min_samples=5)
    X, _ = generate_classification_data(n_features=2)
    model.fit(X)
    # Code to plot or save animation

def main():
    # You can call the algorithm functions sequentially
    plot_linear_regression()  # For Linear Regression animation
    plot_logistic_regression()  # For Logistic Regression animation
    plot_knn()  # For K-Nearest Neighbors animation
    plot_k_means()  # For K-Means animation
    plot_dbscan()  # For DBSCAN animation

if __name__ == "__main__":
    main()
