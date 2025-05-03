# driver.py
from pyspark import SparkContext
from pyspark.sql import SQLContext

import preprocess
import train
import evaluate

def main(train_path,test_path):
    sc  = SparkContext(appName="IDS_Driver")
    SQL = SQLContext(sc)
    sc.setLogLevel("ERROR")

    # 1) Preprocess training (labeled) data
    train_features, pmodel = preprocess.preprocess(SQL, train_path)
    print("Preprocessing done.... Feature count =", train_features.count())

    # 2) Train
    model, threshold = train.train(train_features)
    print("Training done... Threshold = {:.4f}".format(threshold))

    # 3) Preprocess testing (unlabled) Data
    test_features, _ = preprocess.preprocess(SQL, test_path, pipelineModel=pmodel, includeLabels=True)
    print("Preprocessing done... Feature count =", test_features.count())

    # 3) Evaluate
    results = evaluate.evaluate(test_features, model, threshold)
    print(results)

    sc.stop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: spark-submit driver.py train_data test_data")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
