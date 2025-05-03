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
#print(lines.first())

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

#print(assembler.getInputCols())

pipeline = Pipeline(stages=indexers + encoders + [assembler])
preproc_model = pipeline.fit(lines)
df = preproc_model.transform(lines)
#print(df.first())

data = df.select("features").rdd.map(lambda row: row.features)
data.cache()
#print(data.first())

#Tunning

#costs = []
#for k in range(2,11):
#    model = KMeans.train(data, k, maxIterations=20, runs=1, initializationMode="k-means||")
#    costs.append((k, model.computeCost(data)))

#ks, cs = zip(*costs)

# plot
#plt.figure()
#plt.plot(ks, cs, marker='o', linestyle='-')
#plt.xlabel('Number of clusters (k)')
#plt.ylabel('Within-cluster sum of squares (cost)')
#plt.title('Elbow Method for K-means')
#plt.grid(True)
#plt.show()

#def cost_for_steps(steps):
#    model = KMeans.train(
#        data,
#        k=6,
#        maxIterations=20,            # leave iterations modest for now
#        runs=1,                       # one run per initialization strategy
#        initializationMode="k-means||",
#        initializationSteps=steps
#    )
#    return model.computeCost(data)

#for steps in [2, 5, 10, 20]:
#    wssse = cost_for_steps(steps)
#    print("initSteps =", steps, "->WSSSE =", wssse)

#def cost_for_runs(runs):
#    model = KMeans.train(
#        data,
#        k=6,
#        maxIterations       = 20,              # keep modest for now
#        runs                = runs,
#        initializationMode  = "k-means||",
#        initializationSteps = 5,
#        seed                = 42
#    )
#    return model.computeCost(data)

#for runs in [1, 3, 5, 10]:
#    wssse = cost_for_runs(runs)
#    print("runs = {:2d} ->WSSSE = {:.2e}".format(runs, wssse))

#def cost_for_iters(iters):
#    model = KMeans.train(
#        data,
#        k=6,
#        maxIterations       = iters,
#        runs                = 1,               # single run is sufficient
#        initializationMode  = "k-means||",
#        initializationSteps = 5,
#        seed                = 42
#    )
#    return model.computeCost(data)

#for iters in [10, 20, 50, 100]:
#    wssse = cost_for_iters(iters)
#    print("maxIter = {:3d} ->WSSSE = {:.2e}".format(iters, wssse))

final_model = KMeans.train(data,k=6,maxIterations=10,runs=1,initializationMode="k-means||",initializationSteps=5,seed=42)

#centers = final_model.clusterCenters
#pts     = [c.toArray() if hasattr(c, "toArray") else c for c in centers]

# 2. Grab just the first two coordinates
#xs = [p[0] for p in pts]
#ys = [p[1] for p in pts]

#plt.figure()
#plt.scatter(xs, ys, s=100, marker='x')
#for i, (x, y) in enumerate(zip(xs, ys)):
#    plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')
#plt.xlabel('Feature 0')
#plt.ylabel('Feature 1')
#plt.title('Cluster Centers (first two features)')
#plt.show()





#sc.stop()
