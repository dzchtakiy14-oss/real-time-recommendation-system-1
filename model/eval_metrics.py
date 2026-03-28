import numpy as np

def recall_at_k(ranked_items, positive_items, k):
    """
    Args:
        ranked_items: list of item IDs مرتبة حسب أعلى تشابه (من الأعلى إلى الأدنى)
        positive_items: list of item IDs التي تفاعل معها المستخدم بالفعل
        k: عدد العناصر في القمة
    Returns:
        recall: float بين 0 و 1
    """
    if not positive_items:
        return 0.0
    # خذ أول k عنصر من ranked_items
    top_k = set(ranked_items[:k])
    hits = len(top_k & set(positive_items))
    return hits / len(positive_items)


import numpy as np

def ndcg_at_k(ranked_items, positive_items, k):
    """
    Args:
        ranked_items: list مرتبة من الأعلى إلى الأدنى
        positive_items: list of relevant items
        k: عدد العناصر في القمة
    Returns:
        ndcg: float بين 0 و 1
    """
    if not positive_items:
        return 0.0
    # relevance vector: 1 إذا كان العنصر في positive_items
    relevance = [1 if item in positive_items else 0 for item in ranked_items[:k]]
    # DCG
    dcg = sum(rel / np.log2(i+2) for i, rel in enumerate(relevance))  # i+2 لأن i تبدأ من 0
    # IDCG (ideal DCG) – أفضل ترتيب ممكن (جميع العناصر الموجبة أولاً)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i+2) for i, rel in enumerate(ideal_relevance))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(ranked_items, positive_items, k):
    """
    Args:
        ranked_items: list مرتبة
        positive_items: list
        k: عدد العناصر في القمة
    Returns:
        ap: float (متوسط الدقة)
    """
    if not positive_items:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in positive_items:
            hits += 1
            sum_precision += hits / (i + 1)  # precision@i
    return sum_precision / min(k, len(positive_items))

def map_at_k(user_ranked_dict, user_positives_dict, k):
    """
    Args:
        user_ranked_dict: {user_id: ranked_items_list}
        user_positives_dict: {user_id: positive_items_list}
        k: عدد العناصر في القمة
    Returns:
        map_score: float
    """
    aps = []
    for user in user_ranked_dict:
        if user in user_positives_dict and user_positives_dict[user]:
            ap = average_precision_at_k(user_ranked_dict[user], user_positives_dict[user], k)
            aps.append(ap)
    return np.mean(aps) if aps else 0.0