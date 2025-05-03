def evaluate(features, model, threshold):
    # compute distances and flag anomalies
    centers = model.clusterCenters
    size = features.take(1)

    def sq_dist(v):
        c = centers[model.predict(v)]
        return float(v.squared_distance(c))

    def to_binary(pair):
        v,label = pair
        dist     = sq_dist(v)
        pred     = 1.0 if dist > threshold else 0.0
        true     = 0.0 if label == "normal." else 1.0
        return (true, pred)

    def anomalies():
        scored = features.map(lambda v: (v, sq_dist(v)))
        anomalies = scored.filter(lambda (_,dist): dist > threshold)

        # return count
        return {"anomaly_count": anomalies.count()}

    def binaries():
        binaries = features.map(to_binary)

        tp = binaries.filter(lambda (t,p): t==1.0 and p==1.0).count()
        tn = binaries.filter(lambda (t,p): t==0.0 and p==0.0).count()
        fp = binaries.filter(lambda (t,p): t==0.0 and p==1.0).count()
        fn = binaries.filter(lambda (t,p): t==1.0 and p==0.0).count()
        an = binaries.filter(lambda (t,p): p==1).count()

        # 5) derived metrics
        total     = float(tp + tn + fp + fn)
        precision = tp/float(tp+fp)   if tp+fp>0 else 0.0
        recall    = tp/float(tp+fn)   if tp+fn>0 else 0.0
        accuracy  = (tp+tn)/total     if total>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        fpr       = fp/float(fp+tn)   if fp+tn>0 else 0.0

        return {
          "TP": tp,
          "TN": tn,
        "FP": fp,
        "FN": fn,
        "Anomaly": an,
        "False Positive Rate": fpr,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
        }

    if isinstance(size[0], tuple) and len(size[0]) == 2:
        return binaries()
    else:
        return anomalies()
