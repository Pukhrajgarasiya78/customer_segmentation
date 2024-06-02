from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('process_file', filename=file.filename))
    return redirect(request.url)

@app.route('/process/<filename>')
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(file_path)
    
    # Handle missing values if any
    data.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category', 'promotion_usage']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Separate numerical and categorical columns
    numeric_columns = ['age', 'income', 'purchase_amount', 'satisfaction_score']
    categorical_columns = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category', 'promotion_usage']

    # Standardize the numerical features
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Convert the processed data to a NumPy array for clustering
    data_for_clustering = data.drop(columns=['id']).values

    # Determine the optimal number of clusters using the Elbow Method
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_for_clustering)
        sse.append(kmeans.inertia_)

    # Plot the SSE for different values of k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Distances (SSE)')
    elbow_plot_path = os.path.join('static', 'elbow_plot.png')
    plt.savefig(elbow_plot_path)
    plt.close()

    # Apply K-Means clustering with the optimal number of clusters (e.g., k=4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(data_for_clustering)

    # Add the cluster labels to the original dataset
    data['Cluster'] = clusters

    # Analyze cluster characteristics
    cluster_summary = data.groupby('Cluster').mean()
    cluster_summary_path = os.path.join('static', 'cluster_summary.html')
    cluster_summary.to_html(cluster_summary_path)

    # Visualize the clusters
    sns.pairplot(data, hue='Cluster', vars=['age', 'income', 'purchase_amount', 'satisfaction_score'])
    cluster_plot_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(cluster_plot_path)
    plt.close()

    return render_template('result.html', elbow_plot_path=elbow_plot_path, cluster_summary_path=cluster_summary_path, cluster_plot_path=cluster_plot_path)

if __name__ == '__main__':
    app.run(debug=True)
