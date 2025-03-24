import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_classification

class LogisticRegression:
    """
    Logistic Regression model implemented using stochastic gradient descent.
    Parameters:
        learning_rate (float): The learning rate for weight updates.
        iterations (int): The number of iterations for training.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100):
        self.lr = learning_rate
        self.iter = iterations
        self.w = None  # Initialize weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on the provided data and generates frames for animation.
        """
        X = np.insert(X, 0, 1, axis=1)  # Add bias term to features
        self.w = np.ones(X.shape[1])  # Initialize weights
        frames = []

        # Determine axis limits
        x_min, x_max = min(X[:, 1]), max(X[:, 1])
        y_min, y_max = min(X[:, 2]), max(X[:, 2])

        # Gradient descent iterations
        for i in range(self.iter):
            j = np.random.randint(0, X.shape[0])  # Randomly select a point
            y_pred = self.sigmoid(np.dot(self.w, X[j]))  # Calculate prediction

            # Update weights
            self.w += self.lr * (y[j] - y_pred) * X[j]

            # Capture frame every 10 iterations
            if i % 10 == 0:
                frames.append(self._create_frame(X, y, x_min, x_max, i))

        # Generate initial boundary line
        x_initial = np.linspace(x_min, x_max, 100)
        y_initial = -(self.w[1] / self.w[2]) * x_initial - (self.w[0] / self.w[2])
        
        # Plot the animation with Streamlit
        plot_animation(frames, X, y, x_initial, y_initial, x_min, x_max, y_min, y_max)

    def sigmoid(self, z: float) -> float:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def _create_frame(self, X: np.ndarray, y: np.ndarray, x_min: float, x_max: float, iteration: int) -> go.Frame:
        """
        Creates a frame for the animation showing the decision boundary.
        """
        x_values = np.linspace(x_min, x_max, 100)
        y_line = -(self.w[1] / self.w[2]) * x_values - (self.w[0] / self.w[2])

        return go.Frame(
            data=[
                go.Scatter(x=X[:, 1], y=X[:, 2], mode='markers', marker=dict(color=y, colorscale=['blue', 'orange']), name='Data Points'),
                go.Scatter(x=x_values, y=y_line, mode='lines', name=f'Iteration {iteration + 1}')
            ],
            layout=go.Layout(title_text=f"Iteration {iteration + 1}")
        )

def plot_animation(frames: list, X: np.ndarray, Y: np.ndarray, x_initial: np.ndarray, y_initial: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    Creates an animated plot for the logistic regression training process.
    """
    fig = go.Figure(
        data=[
            go.Scatter(x=X[:, 1], y=X[:, 2], mode='markers', marker=dict(color=Y, colorscale=['blue', 'orange']), name='Data Points'),
            go.Scatter(x=x_initial, y=y_initial, mode='lines', name='Initial Decision Boundary')
        ],
        layout=go.Layout(
            title="Logistic Regression Animation",
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
            updatemenus=[{
                "buttons": [{
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                }],
                "type": "buttons",
                "showactive": False,
            }]
        ),
        frames=frames
    )
    st.plotly_chart(fig)

def plot_model(n_samples: int = 100, separation: float = 1.0, learning_rate: float = 0.01, iterations: int = 100):
    """
    Generates a synthetic dataset and fits the LogisticRegression model, then plots the animation.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
        class_sep=separation, random_state=41
    )
    model = LogisticRegression(learning_rate=learning_rate, iterations=iterations)
    model.fit(X, y)

# Streamlit UI
if __name__ == "__main__":
    st.title("Logistic Regression Animation")

    # Set up sliders for parameters
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
    separation = st.sidebar.slider("Class Separation", 0.1, 2.0, 1.0)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
    iterations = st.sidebar.slider("Iterations", 50, 500, 100)

    # Plot the model with specified parameters
    plot_model(n_samples=n_samples, separation=separation, learning_rate=learning_rate, iterations=iterations)
