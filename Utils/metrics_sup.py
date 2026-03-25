import numpy as np

def metrics(original_pc, detect_pc, threshold, T):
    original_pc = np.array(original_pc)
    detect_pc = np.array(detect_pc)

    candidates = []

    for i, o in enumerate(original_pc):
        for j, d in enumerate(detect_pc):
            dist = abs(o - d)
            if dist <= threshold:
                candidates.append((dist, i, j))

    candidates.sort(key=lambda x: (x[0], x[1]))

    assigned_originals = set()
    assigned_detected = set()
    match_points = {}
    distances = []

    for dist, i, j in candidates:
        if i not in assigned_originals and j not in assigned_detected:
            assigned_originals.add(i)
            assigned_detected.add(j)
            match_points[original_pc[i]] = detect_pc[j]
            distances.append(dist)

    distances = np.array(distances)

    TP = len(distances)
    FP = len(detect_pc) - TP
    FN = len(original_pc) - TP
    TN = T - TP - FP - FN
    
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    metrics = {
        'Mean Location Error': np.mean(distances) if len(distances) > 0 else np.nan,
        'Precision': Precision,
        'Recall': Recall,
        'F1 Score': F1,
        'Accuracy': Accuracy,
        'Falsos Positivos': FP,
        'Falsos Negativos': FN,
        'Verdaderos Positivos': TP,
        #'Consistencia': 
    }

    return metrics, match_points
