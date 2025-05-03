from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors

import matplotlib
import matplotlib.pyplot as plt

# 1) Spark setup
sc = SparkContext(appName="K-Mean_IDS")
sc.setLogLevel("WARN")
sql = SQLContext(sc)

# 2) Load KDD Cup '99 data (full or 10% sample)
data_path = "hdfs:///user/kingsmen/kdd_1999/kddcup.data_10_percent.csv"
lines = sql.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "false").load(data_path)

# 3) Assign column names (41 features + label)
cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
        "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label"
]

# 4) Data preprocessing

lines = lines.toDF(*cols)
lines = lines.drop("label")

protocol_idx = StringIndexer(inputCol="protocol_type", outputCol="protocol_typeIndex")
service_idx  = StringIndexer(inputCol="service",       outputCol="serviceIndex")
flag_idx     = StringIndexer(inputCol="flag",          outputCol="flagIndex")

protocol_enc = OneHotEncoder(inputCol="protocol_typeIndex", outputCol="protocol_typeVec")
service_enc  = OneHotEncoder(inputCol="serviceIndex",   outputCol="serviceVec")
flag_enc     = OneHotEncoder(inputCol="flagIndex",      outputCol="flagVec")

indexers = [protocol_idx, service_idx, flag_idx]
encoders = [protocol_enc, service_enc, flag_enc]

# 4a) Assemble features into single vector
assembler = VectorAssembler(
        inputCols=[
            "duration","protocol_typeVec","serviceVec","flagVec","src_bytes","dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
            "root_shell","su_attempted","num_root","num_file_creations","num_shells",
            "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
            "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
            "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
            "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
        ], outputCol="features")

# 5) Build and Run pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])
preproc_model = pipeline.fit(lines)
df = preproc_model.transform(lines)

# 6) Convert to rdd, cache, and train model on kmeans
data = df.select("features").rdd.map(lambda row: row.features)
data.cache()
model = KMeans.train(data,k=6,maxIterations=10,runs=1,initializationMode="k-means||",initializationSteps=5,seed=42)

centers = model.clusterCenters

# 9) Score distances for each point
def sq_dist_to_center(v):
    c = centers[model.predict(v)]
    return float(sum((v[i] - c[i])**2 for i in range(len(c))))

distances = data.map(lambda v: sq_dist_to_center(v))

# 10) Pick a 95th-percentile threshold (sample to the driver)
sample = distances.sample(False, 0.15, seed=42).collect()
sample.sort()
threshold = sample[int(0.90 * (len(sample)-1))]

sc.stop()
