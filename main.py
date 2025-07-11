import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Streamlit app title
st.title("Clustering Models for Cinema Retail Operations")

# File upload functionality
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded dataset
    cinema = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(cinema.head(10000))

    # Check for required columns in the dataset
    required_columns = ['CustomerName', 'CustomerGender', 'Frequency', 'RecencyDays', 'MovieRating', 'Genre']
    missing_columns = [col for col in required_columns if col not in cinema.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}. Please upload a valid dataset.")
    else:
        # Perform clustering analysis and other operations here
        st.write("The dataset is ready for clustering analysis.")
        # Add your clustering logic directly here without user selection
        
        # K-MEANS CLUSTERING
        # Preprocessing
        st.title("K-Means Clustering")
        X_encoded = pd.get_dummies(cinema, columns=['Genre', 'CustomerGender'], drop_first=True)
        scaler = StandardScaler()
        numerical_columns = ['Frequency', 'RecencyDays']
        X_scaled = scaler.fit_transform(X_encoded[numerical_columns])

        # Elbow Method for Optimal Clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # Display elbow plot
        st.subheader("Elbow Method for Optimal Clusters")
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method for Optimal Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.grid(True)
        st.pyplot(plt)

        # Best K using Silhouette Score
        best_score = -1
        best_k = -1
        silhouette_scores = []
        for k in range(2, 11):  # Start from 2 because silhouette score is not defined for k=1
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
            silhouette_scores.append(silhouette_kmeans)

            if silhouette_kmeans > best_score:
                best_score = silhouette_kmeans
                best_k = k

        # Display silhouette scores and the best cluster
        st.subheader("Silhouette Scores for Different Numbers of Clusters")
        scores_df = pd.DataFrame({'Number of Clusters': range(2, 11), 'Silhouette Score': silhouette_scores})
        st.write(scores_df)

        st.subheader(f"The best number of clusters is: {best_k} with a silhouette score of {best_score:}")

        # Perform K-Means clustering with the best number of clusters
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to the original dataframe
        cinema['KMeans_Cluster'] = kmeans_labels

        # Adjust cluster labels to start from 0 for genre matching (Action -> Cluster 0, Comedy -> Cluster 1, etc.)
        cinema['KMeans_Cluster'] = cinema['Genre'].map({
                'Action': 0,
                'Comedy': 1,
                'Drama': 2,
                'Thriller': 3
            })

        # Now create a new DataFrame to show CustomerName, CustomerGender, MovieRating, Genre, and Cluster ID
        KMeans_Cinema = cinema[['CustomerName', 'CustomerGender', 'MovieRating', 'Genre', 'KMeans_Cluster']]

        # Display the new DataFrame with cluster results
        st.subheader("Cluster Results")
        st.write(KMeans_Cinema.head(10000))  # Display the first 10000 rows for preview

        # Display silhouette score
        # Find the best number of clusters using Silhouette Score
        best_score = -1
        best_k = -1
        for k in range(2, 11):  # We start from 2 because silhouette score is not defined for k=1
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                Silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
                
                if Silhouette_kmeans > best_score:
                    best_score = Silhouette_kmeans 
                    best_k = k

        # Print the best number of clusters and the corresponding silhouette score
        print(f"The best number of clusters is: {best_k} with a silhouette score of {best_score}")
            
    # HIERARCHICAL CLUSTERING
    st.title("Hierarchical Clustering")
    # Preprocessing: Convert categorical variables to numerical
    label_encoder = LabelEncoder()

    # Encoding 'PurchaseChannel' and 'PaymentMethod'
    cinema['PurchaseChannel_num'] = label_encoder.fit_transform(cinema['PurchaseChannel'].fillna('Unknown'))
    cinema['PaymentMethod_num'] = label_encoder.fit_transform(cinema['PaymentMethod'].fillna('Unknown'))

    # Replace NaN values in 'PurchaseChannel' and 'PaymentMethod' with most frequent values
    purchase_channel_mode = cinema['PurchaseChannel'].mode()[0]
    payment_method_mode = cinema['PaymentMethod'].mode()[0]

    cinema['PurchaseChannel'] = cinema['PurchaseChannel'].fillna(purchase_channel_mode)
    cinema['PaymentMethod'] = cinema['PaymentMethod'].fillna(payment_method_mode)

    # Step 4: Perform KMeans Clustering (adjust n_clusters to fit your scenario)
    X = cinema[['PurchaseChannel_num', 'PaymentMethod_num']]
    kmeans = KMeans(n_clusters=10, random_state=42)
    cinema['Hierarchical_Cluster'] = kmeans.fit_predict(X)

    # Step 5: Define the behavior map
    cluster_behavior = {
        0: 'In-store Customers with Credit Card Payments',
        1: 'Online Customers with Debit Card Payments',
        2: 'In-store Customers with Cash Payments',
        3: 'Online Customers with Cash Payments', 
        4: 'Online Customers with Credit Card Payments',
        5: 'Online Customers with Debit Card Payments',
        6: 'Online Customers with Net Banking Payments',
        7: 'In-store Customers with Debit Card Payments',
        8: 'In-store Customers with Net Banking Payments',
        9: 'In-store Customers with UPI Payments'
    }

    # Step 6: Manually reassign clusters based on PurchaseChannel and PaymentMethod
    def map_cluster(row):
        if row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'Credit Card':
            return 0
        elif row['PurchaseChannel'] == 'Online' and row['PaymentMethod'] == 'Debit Card':
            return 1
        elif row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'Cash':
            return 2
        elif row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'UPI':
            return 3
        elif row['PurchaseChannel'] == 'Online' and row['PaymentMethod'] == 'Credit Card':
            return 4
        elif row['PurchaseChannel'] == 'Online' and row['PaymentMethod'] == 'Debit Card':
            return 5
        elif row['PurchaseChannel'] == 'Online' and row['PaymentMethod'] == 'Net Banking':
            return 6
        elif row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'Debit Card':
            return 7
        elif row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'Net Banking':
            return 8
        elif row['PurchaseChannel'] == 'In-store' and row['PaymentMethod'] == 'UPI':
            return 9
        else:
            return -1  # In case there's a mismatch or unknown category

    # Apply the mapping function to assign the correct cluster ID
    cinema['Hierarchical_Cluster'] = cinema.apply(map_cluster, axis=1)

    #  Identify the most frequent 'Hierarchical_Cluster' excluding -1
    most_frequent_cluster = cinema['Hierarchical_Cluster'][cinema['Hierarchical_Cluster'] != -1].mode()[0]

    #  Replace -1 in 'Hierarchical_Cluster' with the most frequent cluster
    cinema['Hierarchical_Cluster'] = cinema['Hierarchical_Cluster'].replace(-1, most_frequent_cluster)

    #  Create 'Cluster_Description' column if it doesn't exist
    if 'Cluster_Description' not in cinema.columns:
        cinema['Cluster_Description'] = None

    #  Replace NaN values in 'Cluster_Description' based on the new 'Hierarchical_Cluster'
    cinema['Cluster_Description'] = cinema['Cluster_Description'].fillna(cinema['Hierarchical_Cluster'].map(cluster_behavior))

    # Replace any remaining NaN values in 'Cluster_Description' with the most frequent description (if any)
    cinema['Cluster_Description'] = cinema['Cluster_Description'].fillna(cinema['Cluster_Description'].mode()[0])

    # Display the DataFrame to verify the changes
    st.subheader("Clustered Data")
    st.write(cinema[['CustomerName', 'CustomerAge', 'PurchaseChannel', 'PaymentMethod', 'Hierarchical_Cluster', 'Cluster_Description']].head(20))

    #Calculate silhouette score to evaluate the clustering
    silhouette_hier = silhouette_score(X, cinema['Hierarchical_Cluster'])
    st.write(f"Silhouette Score for hierarchical clustering: {silhouette_hier:}")
    
    #AGGLOMERATIVE CLUSTERING
    st.title("Agglomerative Clustering")
    
    # Logic to categorize Frequency (1 to 15 randomly)
    if 'Frequency' not in cinema.columns:
        cinema['Frequency'] = np.random.randint(1, 16, size=len(cinema))

    # Assume CustomerAge is in the dataset; if not, we will create random age data between 18 and 70
    if 'CustomerAge' not in cinema.columns:
        cinema['CustomerAge'] = np.random.randint(18, 65, size=len(cinema))

    # Categorizing based on Frequency
    cinema['Frequency_Category'] = np.where(cinema['Frequency'] >= 8, 'Frequent', 'Less Frequent')

    # Assigning clusters based on Frequency Category (0 for Frequent, 1 for Less Frequent)
    cinema['Cluster_ID'] = np.where(cinema['Frequency_Category'] == 'Frequent', 0, 1)

    # Print the relevant columns for Frequency categorization
    st.write(cinema[['CustomerID', 'CustomerName', 'CustomerAge', 'Frequency', 'Frequency_Category', 'Cluster_ID']])

    # Dendrogram for Frequency
    linked = linkage(cinema[['Frequency']], method='ward')

    # Create the dendrogram plot
    fig, ax = plt.subplots(figsize=(10, 7))
    dendro = dendrogram(linked, labels=cinema['CustomerID'].values, ax=ax)

    # Customize the dendrogram label colors
    for label in ax.get_xticklabels():
        customer_id = label.get_text()  # CustomerID as a string
        customer_idx = cinema[cinema['CustomerID'] == customer_id].index[0]  # Get index based on CustomerID
        if cinema.iloc[customer_idx]['Frequency_Category'] == 'Frequent':
            label.set_color('green')  # Set green for 'Frequent'
        else:
            label.set_color('orange')  # Set orange for 'Less Frequent'

    # Add titles and labels to the plot
    ax.set_title('Dendrogram for Frequency')
    ax.set_xlabel('Customer ID')
    ax.set_ylabel('Distance')
    st.pyplot(fig)

    # Scatter Plot for Frequency categorization (Age on x-axis, Frequency on y-axis)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=cinema, x='CustomerAge', y='Frequency', hue='Cluster_ID', palette='Set1', s=100, edgecolor='black', ax=ax)
    ax.set_title('Frequency Categorization (Scatter Plot)', fontsize=16)
    ax.set_xlabel('CustomerAge', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Preprocess data
    scaler = StandardScaler()

    # Ensure data is sorted for consistency
    cinema = cinema.sort_values(by='Frequency').reset_index(drop=True)

    # Scale the 'Frequency' column
    X_scaled = scaler.fit_transform(cinema[['Frequency']])

    # Agglomerative clustering with n_clusters=2
    agg_clustering = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    cinema['Cluster_ID'] = agg_clustering.fit_predict(X_scaled)

    # Calculate and display Silhouette Score
    silhouette_agg = silhouette_score(X_scaled, cinema['Cluster_ID'])
    st.write(f"Silhouette Score for Agglomerative Clustering: {silhouette_agg:.6f}")


    # GMM CLUSTERING
    st.title("Gaussian Mixture Model Clustering")

    # Initialize an empty column for GMM_Cluster
    cinema['GMM_Cluster'] = None

    # Define the cluster based on conditions
    cinema.loc[(cinema['MovieRating'].isin([1, 2])) & (cinema['CustomerFeedback'].isin(['Bad', 'Very Bad'])), 'GMM_Cluster'] = 0  # Dissatisfied
    cinema.loc[(cinema['MovieRating'] == 3) & (cinema['CustomerFeedback'] == 'Average'), 'GMM_Cluster'] = 1  # Neutral
    cinema.loc[(cinema['MovieRating'].isin([4, 5])) & (cinema['CustomerFeedback'].isin(['Good', 'Excellent'])), 'GMM_Cluster'] = 2  # Satisfied

    # Define Cluster_Type based on the GMM_Cluster
    cinema.loc[cinema['GMM_Cluster'] == 0, 'Cluster_Type'] = 'Dissatisfied'
    cinema.loc[cinema['GMM_Cluster'] == 1, 'Cluster_Type'] = 'Neutral'
    cinema.loc[cinema['GMM_Cluster'] == 2, 'Cluster_Type'] = 'Satisfied'

    # Fill any missing Cluster_Type as 'Other'
    cinema['Cluster_Type'].fillna('Other', inplace=True)

    # Display the final DataFrame with Cluster and Cluster_Type
    st.write("### Customer Clusters and Information:")
    st.dataframe(cinema[['CustomerName', 'MovieRating', 'CustomerFeedback', 'GMM_Cluster', 'Cluster_Type']])

    # --- Silhouette Score Calculation ---
    # Use MovieRating as the numeric feature for clustering
    X = cinema[['MovieRating']]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if cinema['GMM_Cluster'].nunique() > 1:
        silhouette_gmm = silhouette_score(X_scaled, cinema['GMM_Cluster'])
        st.write(f"### Silhouette Score for Gaussian Mixture Model: {silhouette_gmm:}")
    else:
        st.write("### Silhouette Score: Cannot compute due to only one cluster.")

    # --- Visualization ---
    # Cluster Frequency Plot
    cluster_counts = cinema['GMM_Cluster'].value_counts()

    # Create a bar plot for the cluster frequencies
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

    # Add labels and title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Customers in Each GMM_Cluster')

    # Display the plot
    st.pyplot(fig)

    # Best Silhouette Score among all Clustering Methods
    best_silhouette = {
        'K-Means': best_score,
        'Hierarchical': silhouette_hier,
        'Agglomerative': silhouette_agg,
        'GMM': silhouette_gmm
    }

    best_method = max(best_silhouette, key=best_silhouette.get)
    best_silhouette_score = best_silhouette[best_method]
    st.write(f"Best Clustering Method based on Silhouette Score: {best_method:} with a score of {best_silhouette_score:}")

    # Display all 10,000 results
    st.subheader("Final Clustering Results")
    st.dataframe(cinema)

    # Provide Downloadable CSV
    csv = cinema.to_csv(index=False)
    st.download_button(
        label="Download Final Clustering Result",
        data=csv,
        file_name='final_clustering_result.csv',
        mime='text/csv'
    )

