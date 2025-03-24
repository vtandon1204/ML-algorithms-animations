import plotly.express as px
import numpy as np
from sklearn import datasets
import plotly.graph_objects as go
import streamlit as st

class LinearRegression:
    """
    A simple linear regression model using gradient descent.
    Parameters:
        learning_rate: float - The learning rate for gradient descent.
        no_of_iterations: int - The number of iterations for gradient descent.
        samples: int - Number of data points.
        noise: float - Noise level in the data.
    """

    def __init__(self, learning_rate=0.01, no_of_iterations=100, samples=100, noise=15):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.samples = samples
        self.noise = noise
        self.w = None  # Weight
        self.b = 0  # Bias

    def generate_data(self):
        """
        Generates synthetic linear data with specified noise and sample size.
        """
        X, y = datasets.make_regression(
            n_samples=self.samples, n_features=1, n_informative=1, noise=self.noise, random_state=42
        )
        return X, y

    def fit(self, X, y):
        """
        Fits the model to the data using gradient descent and stores animation frames.
        """
        m, n = X.shape
        self.w = np.zeros(n)  # Initialize weight
        frames = []

        for i in range(self.no_of_iterations):
            y_pred = self.predict(X)
            self._update_weights(X, y, y_pred, m)

            # Only capture frames every 10th iteration to reduce memory usage
            if i % 10 == 0:
                frames.append(self._create_frame(X, y, y_pred, i))

        initial_y_pred = self.predict(X)
        self.plot_animation(frames, X, y, initial_y_pred)

    def _update_weights(self, X, y, y_pred, m):
        """
        Updates weights using the computed gradient.
        """
        dw = -(2 / m) * X.T.dot(y - y_pred)
        db = -(2 / m) * np.sum(y - y_pred)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts target values for the given feature matrix X.
        """
        return X.dot(self.w) + self.b

    def _create_frame(self, X, y, y_pred, iteration):
        """
        Creates a frame for animation with the current model state.
        """
        return go.Frame(
            data=[ 
                go.Scatter(x=X[:, 0], y=y, mode="markers", name="Data Points"),
                go.Scatter(x=X[:, 0], y=y_pred, mode="lines", name=f"Iteration {iteration}")
            ],
            layout=go.Layout(title_text=f"Iteration {iteration}")
        )

    def plot_animation(self, frames, X, y, initial_y_pred):
        """
        Plots the animation of the linear regression fitting process.
        """
        fig = go.Figure(
            data=[
                go.Scatter(x=X[:, 0], y=y, mode="markers", name="Data Points"),
                go.Scatter(x=X[:, 0], y=initial_y_pred, mode="lines", name="Initial Line")
            ],
            layout=go.Layout(
                title="Linear Regression Animation",
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

# Streamlit UI
if __name__ == "__main__":
    st.title("Linear Regression Animation")
    
    # Model parameters
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    no_of_iterations = st.sidebar.slider("Iterations", 50, 500, 100)
    noise = st.sidebar.slider("Noise Level", 0, 50, 15)
    samples = st.sidebar.slider("Sample Size", 50, 200, 100)
    
    # Model and data
    model = LinearRegression(learning_rate=learning_rate, no_of_iterations=no_of_iterations, samples=samples, noise=noise)
    X, y = model.generate_data()
    model.fit(X, y)
