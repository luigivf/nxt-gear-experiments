import mlflow
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import FeatureAgglomeration

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

import pandas as pd
import time
import os

mlflow.set_tracking_uri("http://10.251.21.10:5000")

is_dev = False #os.environ["cluster_experiment"] == "dev"

def load_data():
    print("Loading data...")
    df_vs = pd.read_csv("./data/gear_value_stram.csv", header=0, delimiter=",")

    return df_vs

def load_tag_data():
    print("Loading data...")
    df = pd.read_csv("./data/df_station_tag.csv", header=0, delimiter=",")

    return df

def preprocess_tag_data(df):
    print("Preprocessing tag data...")
    
    df_tag = df[df['year_month'].str.find('2025-3') > -1].copy()

    df_true_false = pd.DataFrame(df_tag.drop(columns=['year_month']) > 0)
    for col in df_true_false.columns:
        df_true_false[col] = df_true_false[col].astype(int)

    df_true_false['year_month'] = df_tag['year_month']

    return df_tag, df_true_false

def has_ms(ts):
    if "." in ts:
        return ts
    else:
        return ts + ".000000"

def preprocess_data(df_vs):
    print("Preprocessing data...")
    vs_columns = df_vs.columns
    vs_columns_ts = vs_columns[vs_columns.str.contains("_ts_") | vs_columns.str.contains("_Binding")]
    vs_columns_lead = vs_columns[vs_columns.str.contains("lead")]
    vs_columns_status = vs_columns[vs_columns.str.contains("status")]

    vs_columns_serials = ['First_AssistMech','Last_AssistMech','Gear', 'Serial']
    vs_columns_stats = ['num_of_am_binded', 'number_of_cycles_in_am','number_of_cycles_in_final']

    vs_columns_stations = vs_columns.difference(vs_columns_lead)\
                                    .difference(vs_columns_ts)\
                                    .difference(vs_columns_serials)\
                                    .difference(vs_columns_status)\
                                    .difference(vs_columns_stats)

    df_vs_filled = df_vs.copy()

    # Fill NaN values with 0 for stations columns
    df_vs_filled[vs_columns_stations] = df_vs[vs_columns_stations].fillna(0)
    for c in vs_columns_stations:
        df_vs_filled[c] = df_vs_filled[c].astype(int)

    # Fill NaN values with 'not_available' for serials columns
    df_vs_filled[vs_columns_serials] = df_vs[vs_columns_serials].fillna('not_available')
    for c in vs_columns_serials:
        df_vs_filled[c] = df_vs_filled[c].astype(str)
    
    # Fill NaN values with -1 for lead columns
    df_vs_filled[vs_columns_lead] = df_vs[vs_columns_lead].fillna(-1)
    for c in vs_columns_lead:
        df_vs_filled[c] = df_vs_filled[c].astype(int)

    # Fill NaN values with '1970-01-01 00:00:00' for ts columns
    df_vs_filled[vs_columns_ts] = df_vs[vs_columns_ts].fillna('1970-01-01 00:00:00')
    for c in vs_columns_ts:
        df_vs_filled[c] = df_vs_filled[c].apply(has_ms)
        df_vs_filled[c] = pd.to_datetime(df_vs_filled[c], format='%Y-%m-%d %H:%M:%S.%f')

    # Fill NaN values with 'not_available' for status columns
    df_vs_filled[vs_columns_status] = df_vs[vs_columns_status].fillna('not_available')
    for c in vs_columns_status:
        df_vs_filled[c] = df_vs_filled[c].astype(str)
    
    # Fill NaN values with 0 for stats columns
    df_vs_filled[vs_columns_stats] = df_vs[vs_columns_stats].fillna(0)
    for c in vs_columns_stats:
        df_vs_filled[c] = df_vs_filled[c].astype(int)

    # Add day column
    df_vs_filled['day'] = df_vs_filled['begin_ts_gear'].apply(lambda x: x.date())

    df_vs_stations = df_vs_filled[vs_columns_stations].copy()

    return df_vs_stations

def experiment_tag_std_hierarchical_clustering(df_tag, experiment, reduce_dimensionality=False):
    
    if is_dev:
        experiment_name = "Local " + experiment
    else:
        experiment_name = experiment

    mlflow.set_experiment(experiment_name)
    print(experiment_name)

    year_month_list = df_tag['year_month'].unique().tolist()

    if is_dev:
        params = {
            "n_year_month": ['2025-1'],
            "n_clusters": [2],
            "linkage": ["ward"], #, "complete", "average", "single"],
            "metric": ["euclidean"] #, "manhattan", "cosine"],
        }
    else:
        params = {
            "n_year_month": year_month_list,
            "n_clusters": [2,3,4,5],#,6,7,8,9,10],
            "linkage": ["ward", "complete", "average", "single"],
            "metric": ["euclidean", "manhattan", "cosine"],
        }

    for year_month in params["n_year_month"]:
        for n_clusters in params["n_clusters"]:
            for linkage in params["linkage"]:
                for metric in params["metric"]:
        

                    run_name = f"hc_std_{n_clusters}_{year_month}_{linkage}_{metric}"
                    if reduce_dimensionality:
                        run_name = run_name + "_reduced"
                    print(run_name)

           
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_param("model_type", "Hierarchical Clustering")
                        mlflow.log_param("n_clusters", n_clusters)
                        mlflow.log_param("year_month", year_month)
                        mlflow.log_param("linkage", linkage)
                        mlflow.log_param("metric", metric)


                        # Sample the dataframe  
                        df_vs_stations_sample = df_tag[df_tag['year_month'] == year_month]\
                                                            .drop(columns=['year_month'])\
                                                            .copy()
                        
                        mlflow.log_param("n_rows", df_vs_stations_sample.shape[0])
                        
                        # Standardize the dataframe
                        scaler = StandardScaler()
                        df_vs_stations_sample_standardized = pd.DataFrame(
                                                    scaler.fit_transform(df_vs_stations_sample)
                                                    , columns=df_vs_stations_sample.columns)

                        if reduce_dimensionality:
                            feature_agg = FeatureAgglomeration(n_clusters=20)
                            df_vs_stations_sample_standardized = feature_agg.fit_transform(df_vs_stations_sample_standardized)



                        # Ajustando o modelo aos dados
                        can_run = True

                        start_time = time.time()
                        try:
                            # Criando o modelo de clustering hierárquico
                            if linkage == "ward":
                                if metric == "euclidean":
                                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
                                else:
                                    raise ValueError("Invalid metric for ward linkage")
                            else:
                                hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)

                            clusters = hierarchical.fit_predict(df_vs_stations_sample_standardized)
                        except:
                            can_run = False
                        else:
                            log_results(df_vs_stations_sample, 
                                        df_vs_stations_sample_standardized, 
                                        clusters, 
                                        run_name, 
                                        hierarchical, 
                                        "hierarchical_clustering_model",
                                        experiment_name)

                        finally:
                            end_time = time.time()
                            execution_time = end_time - start_time
                            mlflow.log_param("execution_time",execution_time)
                            mlflow.log_param("can_run", can_run)
                            
                            print("can_run: ", can_run)
                            print("time: ", execution_time)
                
def experiment_tag_std_kmeans_clustering(df_tag, experiment, reduce_dimensionality=False):
    
    if is_dev:
        experiment_name = "Local " + experiment
    else:
        experiment_name = experiment

    mlflow.set_experiment(experiment_name)
    print(experiment_name)

    year_month_list = df_tag['year_month'].unique().tolist()

    if is_dev:
        params = {
            "n_year_month": ['2025-1'],
            "n_clusters": [2],
            "algorithm": ["lloyd"]#, "elkan"],
        }
    else:
        params = {
            "n_year_month": year_month_list,
            "n_clusters": [2,3,4,5],
            "algorithm": ["lloyd", "elkan"],
        }

    for year_month in params["n_year_month"]:
        for n_clusters in params["n_clusters"]:
            for algorithm in params["algorithm"]:


                    run_name = f"kmeans_std_{n_clusters}_{year_month}_{algorithm}"
                    if reduce_dimensionality:
                        run_name = run_name + "_reduced"
                    print(run_name)

           
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_param("model_type", "KMeans Clustering")
                        mlflow.log_param("n_clusters", n_clusters)
                        mlflow.log_param("year_month", year_month)
                        mlflow.log_param("algorithm", algorithm)

                        # Sample the dataframe  
                        df_vs_stations_sample = df_tag[df_tag['year_month'] == year_month]\
                                                            .drop(columns=['year_month'])\
                                                            .copy()
                        
                        mlflow.log_param("n_rows", df_vs_stations_sample.shape[0])
                        
                        # Standardize the dataframe
                        scaler = StandardScaler()
                        df_vs_stations_sample_standardized = pd.DataFrame(
                                                    scaler.fit_transform(df_vs_stations_sample)
                                                    , columns=df_vs_stations_sample.columns)

                        if reduce_dimensionality:
                            feature_agg = FeatureAgglomeration(n_clusters=20)
                            df_vs_stations_sample_standardized = feature_agg.fit_transform(df_vs_stations_sample_standardized)

                        

                        # Ajustando o modelo aos dados
                        can_run = True

                        start_time = time.time()
                        try:
                            model = KMeans(n_clusters=n_clusters, algorithm=algorithm)
                            model.fit(df_vs_stations_sample_standardized)
                            clusters = model.labels_
                        except:
                            can_run = False
                        else:
                            
                            log_results(df_vs_stations_sample, 
                                        df_vs_stations_sample_standardized, 
                                        clusters, 
                                        run_name, 
                                        model, 
                                        "kmeans_clustering_model",
                                        experiment_name)
                            
                        finally:
                            end_time = time.time()
                            execution_time = end_time - start_time
                            mlflow.log_param("execution_time",execution_time)
                            mlflow.log_param("can_run", can_run)
                            
                            print("can_run: ", can_run)
                            print("time: ", execution_time)

def log_results(df, df_standardized, clusters, run_name, model_to_register, model_to_register_name,experiment_name):
    mlflow.log_metric("davies_bouldin_score", davies_bouldin_score(df_standardized, clusters))
    mlflow.log_metric("silhouette_score", silhouette_score(df_standardized, clusters))
    mlflow.log_metric("calinski_harabasz_score", calinski_harabasz_score(df_standardized, clusters))

    # Adicionando os clusters ao dataframe original
    cluster_tag = f"Cluster"
    df[cluster_tag] = clusters

    file_path = f"./artifacts/df_{run_name}.csv"
    df.to_csv(file_path, index=False)
    mlflow.log_artifact(file_path)
    os.remove(file_path)

    df_cluster_size = df.groupby('Cluster', as_index=False).size()
    mlflow.log_table(df_cluster_size, "cluster_size.json")

    mlflow.log_metric("cluster_size_min", df_cluster_size['size'].min())
    mlflow.log_metric("cluster_size_max", df_cluster_size['size'].max())
    mlflow.log_metric("cluster_size_mean", df_cluster_size['size'].mean())
    mlflow.log_metric("cluster_size_median", df_cluster_size['size'].median())
    mlflow.log_metric("cluster_size_std", df_cluster_size['size'].std())
                            
    # Log the model 
    mlflow.sklearn.log_model(model_to_register, 
          model_to_register_name, 
          registered_model_name=experiment_name.replace(" ", "_").replace("-", "_").replace("/", "_"))

def main():

    
    df_tag = load_tag_data()
    df_tag, df_true_false = preprocess_tag_data(df_tag)

    experiment_tag_std_kmeans_clustering(df_tag, "Gear Clustering - Cycle")
    experiment_tag_std_hierarchical_clustering(df_tag, "Gear Clustering - Cycle")
    
    experiment_tag_std_kmeans_clustering(df_true_false, "Gear Clustering - True/False")
    experiment_tag_std_hierarchical_clustering(df_true_false, "Gear Clustering - True/False")
    
    experiment_tag_std_kmeans_clustering(df_tag, "Gear Clustering - Cycle", reduce_dimensionality=True)
    experiment_tag_std_hierarchical_clustering(df_tag, "Gear Clustering - Cycle", reduce_dimensionality=True)
    
    experiment_tag_std_kmeans_clustering(df_true_false, "Gear Clustering - True/False", reduce_dimensionality=True)
    experiment_tag_std_hierarchical_clustering(df_true_false, "Gear Clustering - True/False", reduce_dimensionality=True)
    


if __name__ == "__main__":
    main()


