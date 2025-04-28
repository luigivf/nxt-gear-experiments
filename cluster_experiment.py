import mlflow
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

import pandas as pd
import time
import os

mlflow.set_tracking_uri("http://10.251.21.10:5000")

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
    df_tag = df.copy()

    return df_tag

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

def experiment_hierarchical_clustering(df_vs_stations):
    mlflow.set_experiment("v3 Gear Value Stream - Cluster")
    print("Experimenting hierarchical clustering...")

    params = {
        "n_clusters": [2,3,4,5,6,7,8,9,10],
        "n_samples": [25000, 50000, 75000, 80000, 90000, 95000]
    }
    

    #params = {
    #    "n_clusters": [2,3],
    #    "n_samples": [100000,120]
    #}

    for n_samples in params["n_samples"]:
        for n_clusters in params["n_clusters"]:
        

            run_name = f"hc_{n_clusters}_clusters_{n_samples}_samples"
            print(run_name)

           
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("model_type", "Hierarchical Clustering")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param("n_samples", n_samples)

                 # Sample the dataframe  
                df_vs_stations_sample = df_vs_stations.sample(n_samples, random_state=42)
                
                # Standardize the dataframe
                scaler = StandardScaler()
                df_vs_stations_sample_standardized = pd.DataFrame(
                                            scaler.fit_transform(df_vs_stations_sample)
                                            , columns=df_vs_stations_sample.columns)

                # Criando o modelo de clustering hierárquico
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

                # Ajustando o modelo aos dados
                can_run = True

                start_time = time.time()
                try:
                    clusters = hierarchical.fit_predict(df_vs_stations_sample_standardized)
                except:
                    can_run = False
                else:
                    can_run = True

                    # Adicionando os clusters ao dataframe original
                    cluster_tag = f"Cluster_{n_clusters}"
                    df_vs_stations_sample[cluster_tag] = clusters

                    file_path = f"./artifacts/df_vs_stations_sample_clustered_{n_clusters}_{n_samples}.csv"
                    df_vs_stations_sample.to_csv(file_path, index=True)
                    os.remove(file_path)

                    mlflow.log_artifact(file_path)
                    

                    mlflow.sklearn.log_model(hierarchical, 
                                            "hierarchical_clustering_model", 
                                            registered_model_name="ValueStreamClustering")
                    
                    os.remove(file_path)
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    mlflow.log_param("execution_time",execution_time)
                    mlflow.log_param("can_run", can_run)
                    
                    print("can_run: ", can_run)
                    print("time: ", execution_time)

def experiment_tag_std_hierarchical_clustering(df_tag):
    mlflow.set_experiment("Local HC Gear Value Stream")
    print("Local HC Gear Value Stream")

    year_month_list = df_tag['year_month'].unique().tolist()


    params = {
        "n_year_month": year_month_list,
        "n_clusters": [2,3,4,5,6,7,8,9,10],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine"],
    }

    #params = {
    #    "n_clusters": [2],
    #    "n_year_month": ['2024-1']
    #}


    #params = {
    #    "n_year_month": ['2024-1'],
    #    "n_clusters": [2],
    #    "linkage": ["ward", "complete", "average", "single"],
    #    "metric": ["euclidean", "manhattan", "cosine"],
    # }

    for year_month in params["n_year_month"]:
        for n_clusters in params["n_clusters"]:
            for linkage in params["linkage"]:
                for metric in params["metric"]:
        

                    run_name = f"hc_std_{n_clusters}_{year_month}_{linkage}_{metric}"
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
                            can_run = True

                            mlflow.log_metric("davies_bouldin_score", davies_bouldin_score(df_vs_stations_sample_standardized, clusters))
                            mlflow.log_metric("silhouette_score", silhouette_score(df_vs_stations_sample_standardized, clusters))
                            mlflow.log_metric("calinski_harabasz_score", calinski_harabasz_score(df_vs_stations_sample_standardized, clusters))

                            # Adicionando os clusters ao dataframe original
                            cluster_tag = f"Cluster"
                            df_vs_stations_sample[cluster_tag] = clusters

                            file_path = f"./artifacts/df_vs_stations_sample_clustered_std_{n_clusters}_{year_month}_{linkage}_{metric}.csv"
                            df_vs_stations_sample.to_csv(file_path, index=False)
                        
                            mlflow.log_artifact(file_path)
                        
                            mlflow.sklearn.log_model(hierarchical, 
                                                    "hierarchical_clustering_model", 
                                                    registered_model_name="ValueStreamClusteringMonthly")
                            
                            os.remove(file_path)
                        finally:
                            end_time = time.time()
                            execution_time = end_time - start_time
                            mlflow.log_param("execution_time",execution_time)
                            mlflow.log_param("can_run", can_run)
                            
                            print("can_run: ", can_run)
                            print("time: ", execution_time)
                

def experiment_tag_mm_hierarchical_clustering(df_tag):
    mlflow.set_experiment("Local TagYMmm Gear Value Stream")
    print("Local TagYMmm Gear Value Stream")

    year_month_list = df_tag['year_month'].unique().tolist()


    params = {
        "n_clusters": [2,3,4,5,6,7,8,9,10],
        "n_year_month": year_month_list
    }

    #params = {
    #    "n_clusters": [2],
    #    "n_year_month": ['2024-1']
    #}

    for year_month in params["n_year_month"]:
        for n_clusters in params["n_clusters"]:
        

            run_name = f"hc_mm_{n_clusters}_clusters_{year_month}"
            print(run_name)

           
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("model_type", "Hierarchical Clustering")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param("year_month", year_month)

                 # Sample the dataframe  
                df_vs_stations_sample = df_tag[df_tag['year_month'] == year_month]\
                                                    .drop(columns=['year_month'])\
                                                    .copy()
                
                mlflow.log_param("n_rows", df_vs_stations_sample.shape[0])
                
                # Standardize the dataframe
                scaler = MinMaxScaler()
                df_vs_stations_sample_standardized = pd.DataFrame(
                                            scaler.fit_transform(df_vs_stations_sample)
                                            , columns=df_vs_stations_sample.columns)

                # Criando o modelo de clustering hierárquico
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

                # Ajustando o modelo aos dados
                can_run = True

                start_time = time.time()
                try:
                    clusters = hierarchical.fit_predict(df_vs_stations_sample_standardized)
                except:
                    can_run = False
                else:
                    can_run = True

                    # Adicionando os clusters ao dataframe original
                    cluster_tag = f"Cluster_{n_clusters}"
                    df_vs_stations_sample[cluster_tag] = clusters

                    file_path = f"./artifacts/df_vs_stations_sample_clustered_mm_{n_clusters}_{year_month}.csv"
                    df_vs_stations_sample.to_csv(file_path, index=False)
                
                    mlflow.log_artifact(file_path)
                
                    mlflow.sklearn.log_model(hierarchical, 
                                            "hierarchical_clustering_model", 
                                            registered_model_name="ValueStreamClusteringMonthly")
                    
                    os.remove(file_path)
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    mlflow.log_param("execution_time",execution_time)
                    mlflow.log_param("can_run", can_run)
                    
                    print("can_run: ", can_run)
                    print("time: ", execution_time)
def main():

    #df_vs_stations = load_data()
    #df_vs_stations = preprocess_data(df_vs_stations)
    #experiment_hierarchical_clustering(df_vs_stations)

    df_tag = load_tag_data()
    df_tag = preprocess_tag_data(df_tag)
    experiment_tag_std_hierarchical_clustering(df_tag)
    #experiment_tag_mm_hierarchical_clustering(df_tag)

if __name__ == "__main__":
    main()


