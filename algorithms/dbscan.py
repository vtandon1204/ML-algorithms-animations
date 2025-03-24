from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import numpy as np
import streamlit as st

class Point:
    def __init__(self, coordinates, position, neighbor_points, cluster):
        self.coordinates = coordinates
        self.position = position
        self.neighbor_points = neighbor_points
        self.cluster = cluster

class DBSCAN:

    def __init__(self, eps, min_points):
        self.eps = eps
        self.min_points = min_points

    def _assign_position(self, X, point_idx, distance_matrix):
        """ Assign point position based on neighbors within eps distance """
        neighbor_points = np.where(distance_matrix[point_idx] <= self.eps)[0]
        if len(neighbor_points) >= self.min_points:
            return neighbor_points, 1  # Core point
        elif len(neighbor_points) > 1:
            return neighbor_points, 2  # Border point
        return neighbor_points, 3  # Noise point

    def fit(self, X):
        current_cluster = 0
        points = [None] * len(X)  # Initialize list to avoid index errors
        frames = []
        colors = [-2] * len(X)  # -2: Noise, -1: Unprocessed

        # Compute pairwise distance matrix once
        distance_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

        for i, point in enumerate(X):
            # If point is already processed, skip it
            if colors[i] != -2:
                continue
            
            # Assign position based on neighbors
            neighbor_points, point_pos = self._assign_position(X, i, distance_matrix)
            points[i] = Point(point, point_pos, neighbor_points, -1)  # Default cluster = -1
            colors[i] = 1 - point_pos  # Mark point as core or border
            frames.append(self._create_frame(X, colors, 'Selecting Core and Border Points', i))

            # If it's a core point, start cluster assignment
            if point_pos == 1:
                current_cluster += 1
                colors[i] = current_cluster
                points[i].cluster = current_cluster
                frames.append(self._create_frame(X, colors, f'Assign Cluster {current_cluster}', i))
                self._expand_cluster(X, i, current_cluster, points, colors, frames, distance_matrix)

        self.plot_animation(frames, X)
        return 

    def _expand_cluster(self, X, point_idx, current_cluster, points, colors, frames, distance_matrix):
        """ Expand cluster by processing neighbor points """
        if points[point_idx] is None:
            return  # Safety check

        cluster_members = points[point_idx].neighbor_points
        j = 0
        while j < len(cluster_members):
            expansion_point_idx = cluster_members[j]

            if 0 <= expansion_point_idx < len(points) and points[expansion_point_idx] is not None:
                # Skip already processed points
                if points[expansion_point_idx].cluster == -1:  # Correctly handle unprocessed points (Noise)
                    colors[expansion_point_idx] = current_cluster
                    points[expansion_point_idx].cluster = current_cluster
                    frames.append(self._create_frame(X, colors, f'Assign Cluster {current_cluster}', expansion_point_idx))

                elif points[expansion_point_idx].cluster == 0:
                    colors[expansion_point_idx] = current_cluster
                    points[expansion_point_idx].cluster = current_cluster
                    cluster_members = np.append(cluster_members, points[expansion_point_idx].neighbor_points)

            j += 1

    def _create_frame(self, X, colors, process, index):
        """ Create the frame for animation at each step """
        trace_data_points = go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=colors, size=10, colorscale='Viridis', line=dict(width=0.2)),
            name=process
        )
        trace_point = go.Scatter(
            x=[X[index][0]], y=[X[index][1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=0.2)),
            name='Processing Point'
        ) 
        return go.Frame(data=[trace_data_points, trace_point], layout=go.Layout(title_text=process))

    def plot_animation(self, frames, X):
        """ Display the animation in Streamlit """
        fig = go.Figure(
            data=[frames[0].data[0], frames[0].data[1]],
            layout=go.Layout(
                title="DBSCAN Algorithm Animation",
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
        st.plotly_chart(fig)

# Streamlit UI for running DBSCAN animation
def DBSCAN_Animation(eps, min_points, n_samples, n_cluster, cluster_std):
    X, y = make_blobs(n_samples=n_samples, centers=n_cluster, n_features=2, random_state=42, cluster_std=cluster_std)
    model = DBSCAN(eps=eps, min_points=min_points)
    model.fit(X)

    
