from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def preprocess(SQL, data_path, pipelineModel=None):
# 2) Load KDD Cup '99 data (full or 10% sample)
    lines = SQL.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "false").load(data_path)

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
    num = len(lines.columns)
    if num == len(cols):
        lines = lines.toDF(*cols)
        lines = lines.drop("label")
    elif num == len(cols)-1:
        lines= lines.toDF(*cols[:-1])
    else:
        raise ValueError("Unexpected column count: {}".format(num))

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
    if pipelineModel is None:
        pipelineModel = pipeline.fit(lines)

    feats = pipelineModel.transform(lines)
    data = feats.select("features").rdd.map(lambda row: row.features)
    data.cache()
    return data, pipelineModel
