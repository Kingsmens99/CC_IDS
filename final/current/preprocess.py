from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, VectorAssembler

def preprocess(SQL, data_path, pipelineModel=None, includeLabels=False, splitData=False):
# 1) Load dataset
    if splitData == True:
        lines = data_path
    else:
        lines = SQL.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "false").load(data_path)

# 2) Assign column names (41 features + label)
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

# 3) Data preprocessing
    num = len(lines.columns)

    # If dataset has 42 attributes then it includes label (labeled data)
    if num == len(cols):
        lines = lines.toDF(*cols)

        # We want to return label to conduct binary classification
        if includeLabels:
            labels = lines.select("label").rdd.map(lambda row: row.label)
        
        lines = lines.drop("label")
    
    # If dataset has 41 attributes then it does not have label (unlabeled data)
    elif num == len(cols)-1:
        lines= lines.toDF(*cols[:-1])

    # Error checking for wrong or messed up dataset
    else:
        raise ValueError("Unexpected column count: {}".format(num))

    # We have to do something about the string characters in KDD CUP 99 dataset
    # So I had to pre-process the data to tokenized and hash the string attributes
    protocol_idx = Tokenizer(inputCol="protocol_type", outputCol="protocol_typeIndex")
    service_idx  = Tokenizer(inputCol="service",       outputCol="serviceIndex")
    flag_idx     = Tokenizer(inputCol="flag",          outputCol="flagIndex")

    protocol_enc = HashingTF(inputCol="protocol_typeIndex", outputCol="protocol_typeVec", numFeatures=8)
    service_enc  = HashingTF(inputCol="serviceIndex",   outputCol="serviceVec", numFeatures=16)
    flag_enc     = HashingTF(inputCol="flagIndex",      outputCol="flagVec", numFeatures=8)

    Token = [protocol_idx, service_idx, flag_idx]
    Hashing = [protocol_enc, service_enc, flag_enc]

# 4) Assemble features into single vector due to K-mean clustering algorithm
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

# 5) Build and Run pipeline to create pre-processor model for data fitting
    pipeline = Pipeline(stages=Token + Hashing + [assembler])
    
    # If none then we are building and training a new model altogether
    if pipelineModel is None:
        pipelineModel = pipeline.fit(lines)
    
    # fit the dataset into model and convert dataframe into rdd
    feats = pipelineModel.transform(lines)
    data = feats.select("features").rdd.map(lambda row: row.features).cache()

    # if true want binary classification (requires labeled data)
    if includeLabels:
        return data.zip(labels), pipelineModel
    else:
        return data, pipelineModel
