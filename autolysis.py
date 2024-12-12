# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "chardet",
#   "pandas",
#   "numpy",
#   "requests",
#   "scipy",
#   "matplotlib",
#   "seaborn",
#   "folium",
#   "networkx",
#   "scikit-learn",
#   "statsmodels"
# ]
# ///


import os
import chardet
import pandas as pd
import numpy as np
import requests
from scipy.stats import zscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose

# Set your API key for AI Proxy
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

# -------------------------------------------------------------------------------------------------------------------
# GETTING DATAFRAME

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read the first 10,000 bytes
        result = chardet.detect(raw_data)
    return result['encoding']

def read_csv_with_fallback(file_path, encodings):
    for encoding in encodings:
        try:
            print(f"Trying to read {file_path} with encoding {encoding}...")
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            print(f"Error reading file {file_path} with encoding {encoding}: {e}")
    raise Exception(f"Failed to read file {file_path} with the provided encodings.")

def load_dataframe(file_path):
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'ascii']
    detected_encoding = detect_encoding(file_path)
    print(f"Detected encoding: {detected_encoding}")
    try:
        return read_csv_with_fallback(file_path, encodings_to_try)
    except Exception as e:
        print(f"Error reading csv file: {e}")
        return None

# -------------------------------------------------------------------------------------------------------------------
# GENERIC ANALYSIS

def data_overview(df):
    return {
        'Number of Rows': df.shape[0],
        'Number of Columns': df.shape[1],
        'Data Types': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Column Names': df.columns.tolist()
    }

def data_structure(df):
    return {
        'Numerical Columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'Categorical Columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'Datetime Columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }

def data_size_and_scale(df):
    scale_info = {}
    outliers = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        scale_info[col] = {
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Mean': df[col].mean(),
            'Std Dev': df[col].std()
        }
        z_scores = np.abs(zscore(df[col].dropna()))
        outliers[col] = np.where(z_scores > 3)[0].tolist()

    return scale_info, outliers

def missing_data_patterns(df):
    return df.isnull().sum().loc[lambda x: x > 0]

def distribution_insights(df):
    return {
        col: {
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurtosis()
        } for col in df.select_dtypes(include=[np.number]).columns
    }

def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr() if not numeric_df.empty else "No numeric columns available."

def analyze_data(df):
    return {
        'Data Overview': data_overview(df),
        'Data Structure': data_structure(df),
        'Data Size and Scale': data_size_and_scale(df),
        'Missing Data': missing_data_patterns(df),
        'Distribution Insights': distribution_insights(df),
        'Correlation Matrix': correlation_analysis(df)
    }

# -------------------------------------------------------------------------------------------------------------------
# CHOOSE ANALYSIS

def generate_choice_prompt(df_head, analysis_results):
    return f"""This is a dataframe 'dataset.csv' whose df.head() = {df_head} and generic analysis = {analysis_results}. 
    Based on this information, select and return exactly one analysis from the following list: 
    ['Outlier and Anomaly Detection', 'Correlation analysis, regression analysis, feature importance', 'time series analysis', 'cluster analysis', 'geographic analysis', 'network analysis']. 
    Provide only the selected element as the output.

    If the selected analysis requires specific columns (e.g., datetime for time series, latitude/longitude for geographic, source/target for network), also specify the most appropriate column names based on df.head(). 
    If you cannot determine suitable columns, return 'Analysis not possible: Missing columns' as the output.
    """

def request_to_aiproxy_for_choice(prompt):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "gpt-4o-mini",  # Use GPT-4o-Mini model
        "messages": [
            {"role": "system", "content": "You are a data science assistant who provides concise, precise answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,  # Adjust as needed for creativity
        "max_tokens": 20,   # Adjust based on expected response length
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def choose_analysis(df_head, analysis_results):
    choice_prompt = generate_choice_prompt(df_head, analysis_results)
    analysis_response = request_to_aiproxy_for_choice(choice_prompt)
    analysis_choice = analysis_response['choices'][0]['message']['content']
    print("Analysis Choice by LLM : ", analysis_choice)
    return analysis_choice

def parse_analysis_choice(analysis_choice):
    if ':' in analysis_choice:
        analysis_type, column_info = analysis_choice.split(':', 1)
        column_info = dict(item.split('=') for item in column_info.strip().split(', '))
    else:
        analysis_type = analysis_choice.strip()
        column_info = {}
    return analysis_type, column_info

# -------------------------------------------------------------------------------------------------------------------
# ANALYSIS CODE 

# Function to handle Outlier Detection
def outlier_analysis(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols].dropna()

    threshold = 3
    z_scores = ((df_numeric - df_numeric.mean()) / df_numeric.std()).abs()
    outliers = (z_scores > threshold)

    outlier_plots = []
    for col in df_numeric.columns:
        plt.figure()
        plt.scatter(df_numeric.index, df_numeric[col], label="Data")
        plt.scatter(df_numeric.index[outliers[col]], df_numeric[col][outliers[col]], color='red', label="Outliers")
        plt.title(f"Outliers in {col}")
        plt.legend()
        fig_path = f"outliers_{col}.png"
        plt.savefig(fig_path)
        outlier_plots.append(fig_path)
        plt.close()

    return {'Outlier Analysis': {'outlier_plots': outlier_plots}}

# Function to handle Correlation Analysis
def correlation_analysis(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols].dropna()

    correlation_matrix = df_numeric.corr(method='pearson')

    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    correlation_plot_path = "correlation_matrix.png"
    plt.savefig(correlation_plot_path)
    plt.close()

    return {
        'Correlation Analysis': {
            'correlation_matrix': correlation_matrix,
            'correlation_plot': correlation_plot_path
        }
    }

# Function to handle Regression Analysis
def regression_analysis(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) >= 2:
        df_clean = df[numeric_columns].dropna()

        x = df_clean[numeric_columns[0]]
        y = df_clean[numeric_columns[1]]
        m, b = np.polyfit(x, y, 1)

        plt.figure()
        plt.scatter(x, y, label="Data")
        plt.plot(x, m * x + b, color="red", label="Regression Line")
        plt.title("Regression Analysis")
        plt.legend()
        regression_plot_path = "regression_plot.png"
        plt.savefig(regression_plot_path)
        plt.close()

        return {
            'Regression Analysis': {
                'slope': m,
                'intercept': b,
                'regression_plot': regression_plot_path
            }
        }
    else:
        return {'Regression Analysis': "Analysis not possible: Not enough numeric columns"}

# Function to handle Feature Extraction (PCA)
def feature_extraction(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("PCA - First Two Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    pca_plot_path = "pca_plot.png"
    plt.savefig(pca_plot_path)
    plt.close()

    return {
        'Feature Extraction': {
            'explained_variance': pca.explained_variance_ratio_,
            'pca_plot': pca_plot_path
        }
    }

# Function to handle Time Series Analysis
def time_series_analysis(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df.set_index(datetime_column, inplace=True)

    decomposed = seasonal_decompose(df['value'], model='additive', period=12)

    decomposition_plot_path = "time_series_decomposition.png"
    decomposed.plot()
    plt.savefig(decomposition_plot_path)
    plt.close()

    return {'Time Series Analysis': {'decomposition_plot': decomposition_plot_path}}

# Function to handle Clustering Analysis
def clustering_analysis(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    plt.figure()
    plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    clustering_plot_path = "clustering_plot.png"
    plt.savefig(clustering_plot_path)
    plt.close()

    return {
        'Clustering Analysis': {
            'cluster_centers': kmeans.cluster_centers_,
            'clustering_plot': clustering_plot_path
        }
    }

# Function to handle Geographic Analysis
def geographic_analysis(df, latitude_column, longitude_column):
    df_clean = df[[latitude_column, longitude_column]].dropna()

    m = folium.Map(location=[df_clean[latitude_column].mean(), df_clean[longitude_column].mean()], zoom_start=10)
    for _, row in df_clean.iterrows():
        folium.Marker(location=[row[latitude_column], row[longitude_column]]).add_to(m)

    map_path = "map.html"
    m.save(map_path)

    return {'Geographic Analysis': {'map': map_path}}

# Function to handle Network Analysis
def network_analysis(df, source_column, target_column):
    df_clean = df[[source_column, target_column]].dropna()

    G = nx.from_pandas_edgelist(df_clean, source=source_column, target=target_column)

    network_plot_path = "network_graph.png"
    plt.figure()
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Network Graph")
    plt.savefig(network_plot_path)
    plt.close()

    return {'Network Analysis': {'graph': G, 'network_plot': network_plot_path}}

# Main function to execute based on analysis type
def execute_analysis(df, analysis_type, column_info=None):
    analysis_results = {}

    if "OUTLIER" in str(analysis_type).upper():
        analysis_results.update(outlier_analysis(df))

    if "CORRELATION" in str(analysis_type).upper():
        analysis_results.update(correlation_analysis(df))

    if "REGRESSION" in str(analysis_type).upper():
        analysis_results.update(regression_analysis(df))

    if "FEATURE EXTRACTION" in str(analysis_type).upper():
        analysis_results.update(feature_extraction(df))

    if "TIME SERIES" in str(analysis_type).upper():
        if 'datetime' in column_info:
            analysis_results.update(time_series_analysis(df, column_info['datetime']))
        else:
            analysis_results['Time Series Analysis'] = "Analysis not possible: Missing datetime column"

    if "CLUSTERING" in str(analysis_type).upper():
        analysis_results.update(clustering_analysis(df))

    if "GEOGRAPHIC" in str(analysis_type).upper():
        if 'latitude' in column_info and 'longitude' in column_info:
            analysis_results.update(geographic_analysis(df, column_info['latitude'], column_info['longitude']))
        else:
            analysis_results['Geographic Analysis'] = "Analysis not possible: Missing latitude/longitude columns"

    if "NETWORK" in str(analysis_type).upper():
        if 'source' in column_info and 'target' in column_info:
            analysis_results.update(network_analysis(df, column_info['source'], column_info['target']))
        else:
            analysis_results['Network Analysis'] = "Analysis not possible: Missing source/target columns"

    return analysis_results

# -------------------------------------------------------------------------------------------------------------------
# FOR LLM STORY
# Define AI Proxy request
def request_to_aiproxy_for_story(prompt):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "gpt-4o-mini",  # Use GPT-4o-Mini model
        "messages": [
            {"role": "system", "content": "You are a data science assistant who provides concise, precise answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,  # Adjust as needed for creativity
        "max_tokens": 300,   # Adjust based on expected response length
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def generate_analysis_story(analysis_results, df_head, analysis_type):
    story_prompt = f'''these were the results - {analysis_results} of the dataframe with df.head() - {df_head} and generic analysis as - {analysis_results}
    write a story about your analysis. Have it describe:
    The data you received, briefly
    The analysis you carried out
    The insights you discovered
    The implications of your findings (i.e. what to do with the insights)
    make sure to place the charts correctly and according to the context'''

    return story_prompt

# Function to generate README from a prompt
def generate_readme(prompt, output_file="README.md"):
    try:
        # Call the AI Proxy for GPT-4o-Mini
        story_response = request_to_aiproxy_for_story(prompt)

        # Extract the text from the response
        llm_output = story_response['choices'][0]['message']['content'].strip()

        # Save the output to a README file
        with open(output_file, "w") as file:
            file.write(llm_output)

        print(f"README file generated successfully: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")



# -------------------------------------------------------------------------------------------------------------------
# MAIN LOGIC

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        return

    dataset_path = sys.argv[1]
    df = load_dataframe(dataset_path)

    if df is None:
        print("Failed to load dataset.")
        return

    analysis_results = analyze_data(df)

    # Print analysis results
    for section, result in analysis_results.items():
        print(f"\n{section}:\n", result)

    df_head = df.head()

    # Choose analysis type based on the dataframe
    analysis_choice = choose_analysis(df_head, analysis_results)

    # Parse the analysis type and columns
    analysis_type, column_info = parse_analysis_choice(analysis_choice)
    print(f"Chosen Analysis: {analysis_type}")
    if column_info:
        print(f"Required Columns: {column_info}")

    analysis_results = execute_analysis(df, analysis_type, column_info)

    story_prompt=generate_analysis_story(analysis_results, df_head, analysis_type)

    generate_readme(story_prompt, output_file="README.md")


if __name__ == "__main__":
    main()
