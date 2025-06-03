import numpy as np

def apply_mmr(items, relevance_scores, categories, lambda_param, top_n):
    selected = []
    selected_ids = set()
    candidate_indices = list(range(len(items)))

    while len(selected) < top_n and candidate_indices:
        mmr_scores = []
        for i in candidate_indices:
            relevance = relevance_scores[i]
            if not selected:
                diversity = 0
            else:
                diversity = max([1 if categories[i] == categories[j] else 0 for j in selected])
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))

        # Select the highest MMR score
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        selected_ids.add(items[best_idx])
        candidate_indices.remove(best_idx)

    return [items[i] for i in selected]
