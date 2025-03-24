import streamlit as st
from sklearn.datasets import make_regression
from algorithms.linear_regression import LinearRegression  # Ensure this imports the class properly
from algorithms.logistic_regression import plot_model  # Ensure this is properly imported
from algorithms.knn import plot_knn_model  # Ensure this is properly imported
from algorithms.k_means import plot_kmeans_model  # Ensure this is properly imported
from algorithms.dbscan import DBSCAN_Animation  # Ensure this is properly imported

# Title and sidebar
st.title(':red[ML Algorithm] :blue[Animations]')
st.sidebar.title(':red[ML Algorithm] :blue[Animations]')

st.write('''Welcome to :red[ML Algorithm] :blue[Animations]! ''')
st.write("Discover machine learning like never before. This platform brings complex algorithms to life with interactive, easy-to-understand animations. Whether you are a student, data enthusiast, or a seasoned professional, you can explore and visualize the inner workings of various machine learning models, from simple linear regression to advanced neural networks.")

st.write("About the Project:") 
st.write("The ML Animations project is designed to help users grasp the core concepts of machine learning through dynamic, visual representations. Each animation breaks down the steps of the algorithms, offering an intuitive learning experience. Whether you are looking to reinforce your understanding or gain a fresh perspective, this platform provides an engaging way to interact with machine learning models.")

# Sidebar for choosing supervised or unsupervised learning
Ml = st.sidebar.selectbox('Select ML type:', options=['Supervised', 'Un-Supervised'])
algorithms = []

if Ml == 'Supervised':
    algorithms = ['Linear Regression', 'Logistic Regression', 'K-Nearest Neighbors']
elif Ml == 'Un-Supervised':
    algorithms = ['K-Means Clustering', 'DBSCAN']

Model = st.sidebar.selectbox('Select Machine Learning Algorithm:', options=algorithms)

from sklearn.datasets import make_regression

if Model == 'Linear Regression':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    noise = st.sidebar.number_input('Noise in data', value=50, min_value=0)
    lr = st.sidebar.number_input('Enter Learning rate', value=0.03, step=0.01)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show Animation'):
        # Generate dataset
        X, y = make_regression(n_samples=samples, n_features=1, noise=noise, random_state=42)

        # Initialize and fit the model
        model = LinearRegression(learning_rate=lr, no_of_iterations=max_iter)
        model.fit(X, y)  # Corrected this line


elif Model == 'Logistic Regression':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    sep = st.sidebar.number_input('Separation between classes', value=5)
    lr = st.sidebar.number_input('Enter Learning rate', value=0.03, step=0.01)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show Animation'):
        plot_model(n_samples=samples, separation=sep, learning_rate=lr, iterations=max_iter)

elif Model == 'K-Nearest Neighbors':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    no_classes = st.sidebar.number_input('Enter no of classes', value=2, step=1)
    sep = st.sidebar.number_input('Separation between classes', value=5)
    k = st.sidebar.number_input('Enter value of k', value=5, step=1)

    if st.sidebar.button('Show Animation'):
        plot_knn_model(n_samples=samples, sep=sep, k=k, no_classes=no_classes)

# Unsupervised learning model selection
elif Model == 'K-Means Clustering':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    n_clusters = st.sidebar.number_input('Enter no of clusters', value=2, step=1)
    cluster_std = st.sidebar.number_input('Enter cluster std', value=2)
    max_iter = st.sidebar.number_input('Enter number of iterations', value=100, step=1)

    if st.sidebar.button('Show Animation'):
        plot_kmeans_model(n_samples=samples, n_cluster=n_clusters, iter=max_iter, cluster_std=cluster_std)

elif Model == 'DBSCAN':
    samples = st.sidebar.number_input('Enter Number of sample data points', value=100)
    n_clusters = st.sidebar.number_input('Enter no of clusters', value=4, step=1)
    cluster_std = st.sidebar.number_input('Enter cluster std', value=2)
    eps = st.sidebar.number_input('Enter eps value', value=3)
    min_point = st.sidebar.number_input('Enter value of min neighbor points', value=5)

    if st.sidebar.button('Show Animation'):
        DBSCAN_Animation(n_samples=samples, n_cluster=n_clusters, cluster_std=cluster_std, eps=eps, min_points=min_point)