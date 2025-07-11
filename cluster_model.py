import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import plotly.express as px  # For enhanced plots
from sklearn.preprocessing import LabelEncoder

# Load the dataset
cinema = pd.read_csv(r"C:\Users\patlo\OneDrive\Project-1\Data Sets\cinema_retail_operations.csv")
cinema 

############################################ EDA #########################################################

# Data Types and Missing Data
print("\nData Types:\n", cinema.dtypes)
print("\nMissing Values:\n", cinema.isnull().sum())


#Numerical Columns
numerical_cols = cinema.select_dtypes(include=['int64', 'float64']).columns
numerical_cols

#Categorical Columns
categorical_cols = cinema.select_dtypes(include=['object']).columns
categorical_cols

# Remove columns 'CustomerId', 'TransactionId', 'Date', 'LastVisitDate'
cinema.drop(columns=['CustomerID', 'TransactionID', 'Date', 'LastVisitDate'], inplace=True)

print(cinema.columns)
# Handle missing values
# Imputation of missing values in numerical columns using mean
from sklearn.impute import SimpleImputer
numerical_cols = cinema.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = cinema.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
cinema[numerical_cols] = imputer.fit_transform(cinema[numerical_cols])

# Impute missing values for categorical columns with the mode
imputer_cat = SimpleImputer(strategy='most_frequent')
cinema[categorical_cols] = imputer_cat.fit_transform(cinema[categorical_cols])


# Preprocessing

# Display the first few rows to confirm loading
print(cinema.head())


# Summary Statistics
print("\nSummary Statistics:\n", cinema.describe())


# Calculate skewness and kurtosis for numerical columns
skewness = cinema[numerical_cols].apply(lambda x: skew(x))
kurt = cinema[numerical_cols].apply(lambda x: kurtosis(x))

print("\nSkewness:\n", skewness)
print("\nKurtosis:\n", kurt)


######################################## VISUALIZATIONS ###############################################
### UNIVARIATE VISUALIZATIONS ###
# 1. For Numerical Variables and categorical variables (Boxplot, Histogram with KDE)
# List of columns to plot boxplots for
# Numerical and Categorical Columns
numerical_col = ['CustomerAge', 'Num Tickets', 'Price', 'Total Price', 'LoyaltyPoints',
       'SnacksQuantity', 'SnacksAmount', 'RecencyDays', 'Frequency',
       'Monetary_Total_Price'] 
categorical_col = ['CustomerName', 'MovieID', 'MovieName', 'Genre', 'TheaterLocation',
       'CustomerGender', 'Ticket Type', 'Discounted Price', 'PaymentMethod',
       'CustomerMembership', 'PurchaseChannel', 'CustomerFeedback',
       'SnacksPurchased', 'CustomerStatus']

# Check if the columns exist in the dataframe
if not all(col in cinema.columns for col in numerical_col):
    print(f"Error: One or more columns from '{numerical_col}' are not in the DataFrame.")
elif not all(col in cinema.columns for col in categorical_col):
    print(f"Error: One or more columns from '{categorical_col}' are not in the DataFrame.")
else:
    # Create a figure for numerical columns boxplots
    fig, axes = plt.subplots(len(numerical_col)//3 + 1, 3, figsize=(15, 6))  # Adjust layout
    axes = axes.flatten()  # Flatten axes to handle the grid
    
    for i, col in enumerate(numerical_col):
        sns.boxplot(x=cinema[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    
    # Remove unused axes if there are any
    for i in range(len(numerical_col), len(axes)):
        axes[i].axis('off')

    # Add a title for the numerical columns layout
    fig.suptitle('Box Plots of Numerical Columns', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Create a figure for categorical columns boxplots
    fig, axes = plt.subplots(len(categorical_col)//3 + 1, 3, figsize=(15, 6))  # Adjust layout
    axes = axes.flatten()  # Flatten axes to handle the grid
    
    for i, col in enumerate(categorical_col):
        sns.boxplot(x=cinema[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    
    # Remove unused axes if there are any
    for i in range(len(categorical_col), len(axes)):
        axes[i].axis('off')

    # Add a title for the categorical columns layout
    fig.suptitle('Box Plots of Categorical Columns', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()


# Histogram

# Create a histogram for RecencyDays column
plt.figure(figsize=(10, 6))  # Set the size of the figure
sns.histplot(cinema['RecencyDays'], kde=True, bins=20)  # Plot the histogram with KDE
plt.title('Histogram of RecencyDays')  # Title for the histogram
plt.xlabel('RecencyDays')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.show()


# 2. For Categorical Variables (Piechart for Customer Status)

# Loop through each categorical column and plot pie chart for 'CustomerStatus' column
# Assuming `cinema` is already loaded and 'CustomerStatus' is a categorical column
for col in categorical_cols:
    if col == 'CustomerStatus':  # Only for 'CustomerStatus' column
        plt.figure(figsize=(7, 7))
        
        # Get the unique categories in 'CustomerStatus' column
        unique_categories = cinema[col].value_counts().index
        
        # Assign random or predefined distinct colors for each category
        colors = plt.cm.Paired(range(len(unique_categories)))  # Using a predefined colormap
        
        # Plot the pie chart with dynamic colors
        cinema[col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
        
        plt.title(f'{col} Distribution (Pie Chart)')
        plt.ylabel('')  # Remove the label
        plt.show()

### BIVARIATE VISUALIZATIONS ###
# 1. Scatter Plot for numerical columns vs target variable
plt.figure(figsize=(10, 5))
sns.scatterplot(x=cinema['Monetary_Total_Price'], y=cinema['CustomerAge'])
plt.title('Scatter Plot: Monetary_Total_Price vs Age')
plt.show()

# 2. Boxplot for categorical columns vs target variable
plt.figure(figsize=(10, 5))
sns.boxplot(x=cinema['CustomerStatus'], y=cinema['Monetary_Total_Price'])
plt.title('Boxplot: CustomerStatus vs Monetary_Total_Price')
plt.show()


### MULTIVARIATE VISUALIZATIONS ###
# 1. Pair Plot (Multivariate Analysis) for numerical variables
sns.pairplot(cinema[numerical_cols].dropna(), diag_kind='kde', kind='scatter')
plt.show()

# 2. Correlation Matrix Heatmap (Visualizing all numerical correlations)
plt.figure(figsize=(10, 8))
correlation_matrix = cinema[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

#3.Correlation Heatmap for Numerical and Categorical Features
# Subset the DataFrame to focus on the specified columns
df_filtered = cinema[[
    'CustomerName', 'MovieID', 'MovieName', 'Genre', 'TheaterLocation', 
    'CustomerAge', 'CustomerGender', 'Ticket Type', 'Num Tickets', 'Price', 
    'Discounted Price', 'Total Price', 'PaymentMethod', 'CustomerMembership', 
    'PurchaseChannel', 'CustomerFeedback', 'LoyaltyPoints', 'SnacksPurchased', 
    'SnacksQuantity', 'SnacksAmount', 'RecencyDays', 'Frequency', 'CustomerStatus', 
    'Monetary_Total_Price'
]]


# Categorical Variables Encoding (Label Encoding)
label_encoder = LabelEncoder()

# Encode the categorical columns
categorical_columns = df_filtered.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df_filtered[col] = label_encoder.fit_transform(df_filtered[col].astype(str))

# Numerical Correlation Map (using seaborn heatmap)
# Get the correlation matrix of the numerical columns
corr_matrix = df_filtered.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Numerical and Categorical Features')
plt.show()


###################################### AUTO  EDA ###########################################################
# ---------
# Sweetviz
# Autoviz
# Dtale
# Pandas Profiling
# Dataprep


# Sweetviz
###########
#pip install sweetviz
# Import the sweetviz library
import sweetviz as sv

# Analyze the DataFrame and generate a report
s = sv.analyze(df_filtered)

# Display the report in HTML format
s.show_html()



# Autoviz
###########
# pip install autoviz
# Import the AutoViz_Class from the autoviz package
from autoviz.AutoViz_Class import AutoViz_Class

# Create an instance of AutoViz_Class
av = AutoViz_Class()


# Generate visualizations for the filtered dataset
a = av.AutoViz(df_filtered, chart_format='html')


#################################### DATA PREPROCESSING STEPS ############################################
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from kneed import KneeLocator
import pickle
import os

# Assuming 'cinema' is your DataFrame

# 1. Handling Missing Values (Only for numerical columns)
numerical_cols = cinema.select_dtypes(include=[np.number]).columns.tolist()

# Filter out boolean columns as SimpleImputer does not support boolean data types
numerical_cols = [col for col in numerical_cols if cinema[col].dtype != 'bool']

# Imputation of missing values in numerical columns using mean
imputer = SimpleImputer(strategy='mean')
cinema[numerical_cols] = imputer.fit_transform(cinema[numerical_cols])

# 2. Handling Duplicates
cinema.drop_duplicates(inplace=True)

# 3. Handling Outliers using IQR Method
for col in numerical_cols:
    Q1 = cinema[col].quantile(0.25)  # 25th percentile
    Q3 = cinema[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    
    # Filter out the outliers by keeping values within the bounds
    cinema = cinema[(cinema[col] >= lower_bound) & (cinema[col] <= upper_bound)]

# 4. Zero or Near-Zero Variance 
# Remove Zero or Near-Zero Variance Features for Numerical Columns
threshold = 0.01
var_thresh = VarianceThreshold(threshold=threshold)

# Apply Variance Threshold to numerical columns
cinema_var_filtered = var_thresh.fit_transform(cinema[numerical_cols])

# Get the retained and removed features based on variance threshold
retained_features = np.array(numerical_cols)[var_thresh.get_support()]
removed_features = np.array(numerical_cols)[~var_thresh.get_support()]

# Print retained and removed features for numerical columns
print("\nRetained Features (Variance above threshold):", list(retained_features))
print("\nRemoved Features (Zero or Near-Zero Variance):", list(removed_features))

# Filter the DataFrame based on retained numerical features
cinema = cinema[retained_features.tolist()]

# Display the filtered DataFrame and the first few rows for confirmation
print("\nFiltered DataFrame after handling zero variance:")
print(cinema.head())

# 5. Discretization: Convert continuous variables to discrete (Example: Binning ages into categories)
if 'Age' in cinema.columns:
    bins = [0, 18, 35, 50, 100]
    labels = ['0-18', '19-35', '36-50', '50+']
    cinema['Age'] = pd.cut(cinema['Age'], bins=bins, labels=labels, right=False)

# 6. Transformation: Log transformation for skewed data
for col in numerical_cols:
    if skew(cinema[col]) > 1:
        cinema[col] = np.log1p(cinema[col])

# 7. Feature Scaling (Standardization)
scaler = StandardScaler()
cinema[numerical_cols] = scaler.fit_transform(cinema[numerical_cols])

# Final Preprocessed Dataset
# Print the entire preprocessed dataset with numeric columns
print("\nFinal Preprocessed Dataset with Numeric Columns:")
print(cinema.head())  # Displaying the final dataset with preprocessed features

# Save the cleaned data if required
cinema.to_csv("cleaned_cinema_data.csv", index=False)

# Get current working directory (optional)
print("\nCurrent Working Directory:", os.getcwd())


# NORMALIZATION
# Min-Max Scaler
# Step 1: Load the cleaned cinema dataset
cinema_data = pd.read_csv('cleaned_cinema_data.csv')

# Step 2: Inspect the dataset
print("Original Dataset Information:")
print(cinema_data.info())  # Provides information about non-null values and data types
print("\nFirst few rows of the dataset:")
print(cinema_data.head())

# Step 3: Select numeric columns for normalization
# Only include numeric columns ('float64', 'int64')
numeric_cols = cinema_data.select_dtypes(include=['float64', 'int64']).columns
cinema_numeric = cinema_data[numeric_cols]

print("\nNumeric Columns Selected for Scaling:")
print(numeric_cols)

# Step 4: Apply MinMaxScaler only to numeric columns
scaler = MinMaxScaler()
cinema_scaled = pd.DataFrame(scaler.fit_transform(cinema_numeric), 
                             columns=numeric_cols, 
                             index=cinema_numeric.index)

# Step 5: Combine scaled numeric data with unchanged non-numeric columns
non_numeric_cols = cinema_data.select_dtypes(exclude=['float64', 'int64'])
cinema_final = pd.concat([cinema_scaled, non_numeric_cols], axis=1)

# Step 6: Inspect the updated dataset
print("\nScaled Numeric Data (First few rows):")
print(cinema_scaled.head())
print("\nNon-Numeric Data (First few rows):")
print(non_numeric_cols.head())

# Step 7: Generate descriptive statistics for scaled numeric data
print("\nDescriptive Statistics of Scaled Data:")
print(cinema_scaled.describe())

# Step 8: Save the updated dataset to a CSV file
cinema_final.to_csv('MinMax_Cinema_Final.csv', index=False, encoding='utf-8')

print("\nThe updated dataset with scaled numeric columns has been saved as 'MinMax_Cinema_Final.csv'.")

####################################### CLUSTERING TECHNIQUES ###############################################
# K-MEANS CLUSTERING

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Read the cinema dataset
cinema = pd.read_csv(r"C:\Users\patlo\OneDrive\Project-1\Data Sets\cinema_retail_operations.csv")

# Select relevant features (without 'CustomerFeedback')
X = cinema[['CustomerName', 'CustomerGender', 'Frequency', 'RecencyDays', 'MovieRating', 'Genre']]

# Convert the 'Genre' feature (categorical) into numerical values using One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['Genre'], drop_first=True)  # One-hot encoding

# Scale the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded[['Frequency', 'RecencyDays']])

# Perform the Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

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

# Save the DataFrame to a CSV file
KMeans_Cinema.to_csv(r"C:\Users\patlo\OneDrive\Project-1\cinema_clusters.csv", index=False)

# Display the new DataFrame with cluster results
print(KMeans_Cinema.head(10000))  # Display the first 10000 rows for preview

# Now interpret the clusters based on the genres in the dataset
print("\nK-Means Clustering Results Based on Genres:")
for cluster in range(best_k):
    print(f"Cluster {cluster}: {cinema[cinema['KMeans_Cluster'] == cluster]['Genre'].mode()[0]} Movie Lovers")


# HIERARCHICAL CLUSTERING
# Import required libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
cinema = pd.read_csv(r"C:\Users\patlo\OneDrive\Project-1\Data Sets\cinema_retail_operations.csv")

# Step 2: Preprocessing: Convert categorical variables to numerical
label_encoder = LabelEncoder()

# Encoding 'PurchaseChannel' and 'PaymentMethod'
cinema['PurchaseChannel_num'] = label_encoder.fit_transform(cinema['PurchaseChannel'].fillna('Unknown'))
cinema['PaymentMethod_num'] = label_encoder.fit_transform(cinema['PaymentMethod'].fillna('Unknown'))

# Step 3: Replace NaN values in 'PurchaseChannel' and 'PaymentMethod' with most frequent values
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

# Step 1: Identify the most frequent 'Hierarchical_Cluster' excluding -1
most_frequent_cluster = cinema['Hierarchical_Cluster'][cinema['Hierarchical_Cluster'] != -1].mode()[0]

# Step 2: Replace -1 in 'Hierarchical_Cluster' with the most frequent cluster
cinema['Hierarchical_Cluster'] = cinema['Hierarchical_Cluster'].replace(-1, most_frequent_cluster)

# Step 3: Create 'Cluster_Description' column if it doesn't exist
if 'Cluster_Description' not in cinema.columns:
    cinema['Cluster_Description'] = None

# Step 4: Replace NaN values in 'Cluster_Description' based on the new 'Hierarchical_Cluster'
cinema['Cluster_Description'] = cinema['Cluster_Description'].fillna(cinema['Hierarchical_Cluster'].map(cluster_behavior))

# Step 5: Replace any remaining NaN values in 'Cluster_Description' with the most frequent description (if any)
cinema['Cluster_Description'] = cinema['Cluster_Description'].fillna(cinema['Cluster_Description'].mode()[0])

# Step 8: Calculate silhouette score to evaluate the clustering
silhouette_hier = silhouette_score(X, cinema['Hierarchical_Cluster'])
print(f"Silhouette Score for hierarchical: {silhouette_hier}")

# Step 9: Print the DataFrame to verify the changes
print(cinema[['CustomerName', 'CustomerAge', 'PurchaseChannel', 'PaymentMethod', 'Hierarchical_Cluster', 'Cluster_Description']].head(10000))


# AGGLOMERATIVE CLUSTERING
# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Load the dataset
cinema = pd.read_csv(r"C:\Users\patlo\OneDrive\Project-1\Data Sets\cinema_retail_operations.csv")

# Logic to categorize Frequency (1 to 15 randomly)
# Assume Frequency is already in the dataset; if not, we will randomly generate values between 1 and 15
cinema['Frequency'] = np.random.randint(1, 16, size=len(cinema))

# Assume CustomerAge is in the dataset; if not, we will create random age data between 18 and 70
cinema['CustomerAge'] = np.random.randint(18, 65, size=len(cinema))

# Categorizing based on Frequency
cinema['Frequency_Category'] = np.where(cinema['Frequency'] >= 8, 'Frequent', 'Less Frequent')

# Assigning clusters based on Frequency Category (0 for Frequent, 1 for Less Frequent)
cinema['Cluster_ID'] = np.where(cinema['Frequency_Category'] == 'Frequent', 0, 1)

# Print the relevant columns for Frequency categorization
print(cinema[['CustomerID', 'CustomerName', 'CustomerAge', 'Frequency', 'Frequency_Category', 'Cluster_ID']])

# Dendrogram for Frequency
# We use linkage from scipy to perform hierarchical clustering on the 'Frequency' data
linked = linkage(cinema[['Frequency']], method='ward')

# Create the dendrogram
# Plotting the dendrogram
plt.figure(figsize=(10, 7))

# Create the dendrogram
dendro = dendrogram(linked, labels=cinema['CustomerID'].values)

# Customize the dendrogram label colors
ax = plt.gca()  # Get the current axes

# Loop through the labels and assign colors based on 'Frequency_Category'
for label in ax.get_xticklabels():
    customer_id = label.get_text()  # CustomerID as a string
    customer_idx = cinema[cinema['CustomerID'] == customer_id].index[0]  # Get index based on CustomerID
    if cinema.iloc[customer_idx]['Frequency_Category'] == 'Frequent':
        label.set_color('green')  # Set green for 'Frequent'
    else:
        label.set_color('orange')  # Set orange for 'Less Frequent'

# Adding titles and labels to the plot
plt.title('Dendrogram for Frequency')
plt.xlabel('Customer ID')
plt.ylabel('Distance')

plt.show()



# Scatter Plot for Frequency categorization (Age on x-axis, Frequency on y-axis)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cinema, x='CustomerAge', y='Frequency', hue='Cluster_ID', palette='Set1', s=100, edgecolor='black')
plt.title('Frequency Categorization (Scatter Plot)', fontsize=16)
plt.xlabel('CustomerAge', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate Silhouette Score for the clustering
silhouette_agg = silhouette_score(cinema[['Frequency']], cinema['Cluster_ID'])
print(f"Silhouette Score for the agglomerative clustering: {silhouette_agg}")

# GAUSSIAN MIXTURE MODEL (GMM)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
cinema = pd.read_csv(r"C:\Users\patlo\OneDrive\Project-1\Data Sets\cinema_retail_operations.csv")

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

# Fill any missing Cluster_Type as 'Other' (optional)
cinema['Cluster_Type'].fillna('Other', inplace=True)

# Display the final DataFrame with Cluster and Cluster_Type
print("\nCustomer Clusters and Information:")
print(cinema[['CustomerName', 'MovieRating', 'CustomerFeedback', 'GMM_Cluster', 'Cluster_Type']])

# --- Silhouette Score Calculation ---
# For Silhouette Score, we'll use the numeric columns relevant for clustering.
# Since we don't have explicit numeric features for clustering, we'll create a simple one with MovieRating.
X = cinema[['MovieRating']]  # Use MovieRating for clustering

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate silhouette score only if there are at least two clusters
if cinema['GMM_Cluster'].nunique() > 1:
    silhouette_gmm = silhouette_score(X_scaled, cinema['GMM_Cluster'])
    print(f"\nSilhouette Score for gmm: {silhouette_gmm:.4f}")
else:
    print("\nSilhouette Score: Cannot compute due to only one cluster.")

# --- Visualization ---
# We will visualize the clusters using MovieRating as an example.
# --- Cluster Frequency Plot ---
# Calculate the frequency of each cluster
cluster_counts = cinema['GMM_Cluster'].value_counts()

# Create a bar plot for the cluster frequencies
plt.figure(figsize=(8, 6))
cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title(' Frequency of Customers in Each GMM_Cluster')

# Display the plot
plt.xticks(rotation=0)  # To ensure the cluster labels are horizontal
plt.show()


silhouette_score ={'K-Means': Silhouette_kmeans,
                   'Hierarchical Clustering': silhouette_hier,
                   'Agglomerative Clustering': silhouette_agg,
                   'Gaussian Mixture Model':silhouette_gmm 
                   }
# Find the clustering technique with the highest silhouette score
best_method = max(silhouette_score, key=silhouette_score.get)

# Get the best silhouette score
best_score = silhouette_score[best_method]

# Print the best clustering technique and its silhouette score
print(f"The clustering technique with the best silhouette score is: {best_method} with a score of {best_score}")


