import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs

class K_Means:
    def __init__(self, n_clusters=2, max_iter=200, tolerance=1e-4):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum iterations
        self.tolerance = tolerance  # Convergence tolerance

    def fit(self, X):
        # Randomly select initial centroids from the dataset
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        frames = []
        frames.append(self.create_frame(X, centroids, np.zeros(len(X)), 0))  # Initial frame

        # Iterate and update centroids
        for i in range(self.max_iter):
            clusters = self._get_clusters(X, centroids)

            # Move centroids to the mean of assigned data points
            new_centroids = self._move_centroids(X, clusters)

            # Check for convergence (if centroids change by less than tolerance)
            if np.all(np.abs(new_centroids - centroids) < self.tolerance):
                break

            centroids = new_centroids

            # Add a frame for the current iteration (with clusters and centroids)
            frames.append(self.create_frame(X, centroids, clusters, i + 1))

        # Plot the final frame
        frames.append(self.create_frame(X, centroids, clusters, "Final"))

        # Display the animation
        self.plot_animation(frames, X)

    def _move_centroids(self, X, clusters):
        # Move centroids to the mean of assigned data points
        return np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)])

    def _get_clusters(self, X, centroids):
        # Efficient cluster assignment using vectorized distance calculation
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def create_frame(self, X, centroids, clusters, iteration):
        # Scatter plot for data points colored by cluster
        trace_data_points = go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=clusters, size=8, colorscale='Viridis', line=dict(width=0.2)),
            name=f"Iteration {iteration}"
        )

        # Scatter plot for centroids
        trace_centroids = go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x', line=dict(width=0.2)),
            name='Centroids'
        )

        return go.Frame(data=[trace_data_points, trace_centroids], layout=go.Layout(title_text=f"Iteration {iteration}"))

    def plot_animation(self, frames, X):
        # Create a figure for the animation
        fig = go.Figure(
            data=[frames[0].data[0], frames[0].data[1]],  # Initial data points and centroids
            layout=go.Layout(
                title="K-Means Clustering Animation",
                xaxis=dict(range=[X[:, 0].min() - 1, X[:, 0].max() + 1]),
                yaxis=dict(range=[X[:, 1].min() - 1, X[:, 1].max() + 1]),
                updatemenus=[{
                    "buttons": [{
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    }],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            ),
            frames=frames
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

# Generate data and apply K-Means
def plot_kmeans_model(n_samples, n_cluster, iter, cluster_std):
    # Generate synthetic 2D data using sklearn's make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=n_cluster, n_features=2, random_state=42, cluster_std=cluster_std)

    # Initialize and fit KMeans
    model = K_Means(n_clusters=n_cluster, max_iter=iter)
    model.fit(X)
