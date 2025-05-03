# driver.py
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import KMeansModel
from tabulate import tabulate

import shutil, os, sys
import preprocess, train, evaluate

sc  = SparkContext(appName="IDS_Driver")
SQL = SQLContext(sc)
sc.setLogLevel("ERROR")

def sp(dataset):
    raw = SQL.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "false").load(dataset)
    return raw.randomSplit([0.6, 0.4], seed=42)

def usage_and_exit():
    print("Usage:")
    print("  #1 Evaluate unlabeled data against a pre-trained model:")
    print("    spark-submit driver.py /path/to/unlabeled.csv /path/to/model /path/to/threshold /path/to/save") #4 args
    print("  #2 Train & evaluate on a single labeled dataset:")
    print("    spark-submit driver.py /path/to/labeled.csv /path/to/save") #2 args
    print("  #3 Train on unlabeled, evaluate on labeled:")
    print("    spark-submit driver.py /path/to/unlabeled.csv /path/to/labeled.csv /path/to/save") #3 args
    sc.stop()
    sys.exit(1)

def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) > 4:
        usage_and_exit()
    elif len(args) == 2:
        mode = "Train & Eval on Single Dataset"
    elif len(args) == 4:
        mode = "Eval with pre-trained model"
    elif len(args) == 3:
        mode = "Train & Eval on Double Dataset"

    print("--> Running on mode:", mode)


    if mode == "Train & Eval on Single Dataset":
        dataset_path, path = args

        # 1) split dataset into: train & testing sets
        train_path, test_path = sp(dataset_path)

        # 2) preprocess(labeled_path) ->train_features, pipelineModel
        train_features, pmodel = preprocess.preprocess(SQL, train_path, splitData=True)
        print("Preprocessing done.... Feature count =", train_features.count())
        
        # 3) train(train_features) ->model, threshold
        model, threshold = train.train(train_features)
        print("Training done... Threshold = {:.4f}".format(threshold))
        
        # 4) preprocess(labeled_path, pipelineModel) ->test_features
        test_features, _ = preprocess.preprocess(SQL, test_path, pipelineModel=pmodel, includeLabels=True, splitData=True)
        print("Preprocessing done... Feature count =", test_features.count())
        
        # 5) evaluate(test_features, model, threshold)
        results = evaluate.evaluate(test_features, model, threshold)
        tab = [(k, v) for k, v in results.items()]
        print(tabulate(tab, headers=["Metric", "Value"], floatfmt=".4f", tablefmt="github"))

        #keep = input("Keep trained model and threshold (Y/N): ")
        #if keep == "Y":
        #    path = input("Insert full path to directory: ")
        print("Saving Model and Threshold @ ", path)
        #sc.parallelize([threshold], 1).saveAsTextFile(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save(sc, "file://" + path)
        fp = os.path.join(path, "threshold.txt")
        with open(fp, "w") as f:
            f.write(str(threshold))
        print("Model and Threshold saved, exiting...")
        #else:
        #    print("Exiting...")

    elif mode == "Train & Eval on Double Dataset":
        train_path, test_path, path = args

        # 1) preprocess(unlabeled_path) ->train_features, pipelineModel
        train_features, pmodel = preprocess.preprocess(SQL, train_path)
        print("Preprocessing done.... Feature count =", train_features.count())
        
        # 2) train(train_features) ->model, threshold
        model, threshold = train.train(train_features)
        print("Training done... Threshold = {:.4f}".format(threshold))
        
        # 3) preprocess(labeled_path, pipelineModel) ->test_features
        test_features, _ = preprocess.preprocess(SQL, test_path, pipelineModel=pmodel, includeLabels=True)
        print("Preprocessing done... Feature count =", test_features.count())
        
        # 4) evaluate(test_features, model, threshold)
        results = evaluate.evaluate(test_features, model, threshold)
        tab = [(k, v) for k, v in results.items()]
        print(tabulate(tab, headers=["Metric", "Value"], floatfmt=".4f", tablefmt="github"))

        #keep = raw_input("Keep trained model and threshold (Y/N):")
        #if keep == "Y":
        #    path = raw_input("Insert full path to directory: ")
        print("Saving Model and Threshold @ ", path)
        #sc.parallelize([threshold], 1).saveAsTextFile(path)

        if os.path.exists(path):
            shutil.rmtree(path)
        model.save(sc, "file://" + path)
        fp = os.path.join(path, "threshold.txt")
        with open(fp, "w") as f:
            f.write(str(threshold))
        print("Model and Threshold saved, exiting...")
        #else:
        #    print("Exiting...")

    elif mode == "Eval with pre-trained model":
        test_path, path_model, threshold_path, path = args

        # 1) load pre-trained KMeansModel, threshold
        model = KMeansModel.load(sc, "file://" + path_model)
        threshold = float(sc.textFile("file://" + threshold_path).first())

        # 2) preprocess unlabeled_path ->test_features
        test_features, _ = preprocess.preprocess(SQL, test_path)
        print("Preprocessing done.... Feature count =", test_features.count())
        
        # 3) evaluate(test_features, model, threshold)
        results = evaluate.evaluate(test_features, model, threshold)
        tab = [(k, v) for k, v in results.items()]
        print(tabulate(tab, headers=["Metric", "Value"], floatfmt=".4f", tablefmt="github"))
        
        print("Saving results @ ", path)
        fp = os.path.join(path, "results.txt")
        with open(fp, "w") as f:
            for k,v in results.items():
                f.write("{}={}\n".format(k, v))
        print("Exiting...")

    sc.stop()
    sys.exit(1)

if __name__ == "__main__":
    main()
