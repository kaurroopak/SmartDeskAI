def compute_confidence(similarity_score: float, classification_probability: float) -> float:
    """
    Weighted confidence formula:
    confidence = (0.6 * similarity_score) + (0.4 * classification_probability)
    """
    return round((0.6 * similarity_score) + (0.4 * classification_probability), 4)


def get_confidence_label(confidence: float) -> dict:
    if confidence >= 0.8:
        return {"label": "High", "color": "green", "emoji": "✅"}
    elif confidence >= 0.5:
        return {"label": "Medium", "color": "orange", "emoji": "⚠️"}
    else:
        return {"label": "Low", "color": "red", "emoji": "🔴"}
