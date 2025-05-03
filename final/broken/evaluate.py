def evaluate(features, model, threshold):
    # compute distances and flag anomalies
    centers = model.clusterCenters
    def sq_dist(v):
        c = centers[model.predict(v)]
        return float(v.squared_distance(c))

    scored = features.map(lambda v: (v, sq_dist(v)))
    anomalies = scored.filter(lambda (_,dist): dist > threshold)

    # return count and a small sample
    return {"anomaly_count": anomalies.count(),"sample": anomalies.take(10)}
