from pyspark.mllib.clustering import KMeans

def train(features):
    # 1) Training model and getting cluster centers
    model = KMeans.train(features,k=6,maxIterations=10,runs=1,initializationMode="k-means||",initializationSteps=5,seed=42)
    centers = model.clusterCenters

    # 2) Score distances for each point
    def sq_dist_to_center(v):
        c = centers[model.predict(v)]
        return float(sum((v[i] - c[i])**2 for i in range(len(c))))

    distances = features.map(lambda v: sq_dist_to_center(v))

    # 3) Pick a 95th-percentile threshold (sample to the driver)
    sample = distances.sample(False, 0.10, seed=42).collect()
    sample.sort()
    threshold = sample[int(0.95 * (len(sample)-1))]
    return model, threshold
