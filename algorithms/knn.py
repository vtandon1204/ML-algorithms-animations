import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        # Efficient vectorized distance computation
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Determine the most common class label
        return Counter(k_nearest_labels).most_common(1)[0][0]

def animate_knn(X, y, new_point, k_max):
    frames = []

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    for k in range(1, k_max + 1):
        knn_model = KNN(k)
        knn_model.fit(X, y)
        neighbors = [knn_model._predict(new_point)]  # Ensure it's a list

        trace_neighbors = go.Scatter(
            title="K-Nearest Neighbours Algorithm Animation",
            x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=y, colorscale=['blue', 'orange']),
            name='Data Points'
        )
        trace_nearest = go.Scatter(
            x=[X[i][0] for i in range(len(X)) if y[i] in neighbors],
            y=[X[i][1] for i in range(len(X)) if y[i] in neighbors],
            mode='markers',
            marker=dict(color='black', size=10, symbol='circle-x-open'),
            name=f'{k}-Nearest Neighbors'
        )
        trace_new_point = go.Scatter(
            x=[new_point[0]], y=[new_point[1]],
            mode='markers',
            marker=dict(color='Red', size=12, symbol='x'),
            name=f'New Point (k={k})'
        )

        frames.append(go.Frame(data=[trace_neighbors, trace_nearest, trace_new_point],
                               layout=go.Layout(title_text=f'K = {k}')))

    fig = go.Figure(
        data=[trace_neighbors, trace_new_point],
        layout=go.Layout(
            title="K-Nearest Neighbors Animation",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
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
    st.plotly_chart(fig)


def plot_knn_model(n_samples, sep, k, no_classes):
    # Generate synthetic dataset
    X, y = make_classification(n_samples=n_samples, n_features=2, n_classes=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, hypercube=False, random_state=41, class_sep=sep)
    
    # Choose a random new point to classify
    new_point = X[np.random.randint(0, X.shape[0])]
    animate_knn(X, y, new_point, k_max=k)
