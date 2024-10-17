import numpy as np
from scipy.stats import entropy
from sklearn.metrics import histogram_intersection_loss
from typing import List, Tuple

def calculate_entropy(probabilities: np.ndarray) -> float:
    """Calculate the entropy of a probability distribution."""
    return entropy(probabilities)

def create_histogram(confidences: List[float], bins: int = 10) -> np.ndarray:
    """Create a histogram from a list of confidence scores."""
    return np.histogram(confidences, bins=bins, range=(0, 1), density=True)[0]

def histogram_loss(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Calculate the histogram loss between two histograms."""
    return np.sum((hist1 - hist2) ** 2)

def histogram_loss_metric(original_confidences: List[List[float]], 
                          sr_confidences: List[List[float]], 
                          num_classes: int) -> float:
    """
    Calculate the Histogram Loss metric.
    
    :param original_confidences: List of confidence scores for each class for original images
    :param sr_confidences: List of confidence scores for each class for super-resolved images
    :param num_classes: Number of emotion classes
    :return: Histogram Loss
    """
    total_loss = 0
    for c in range(num_classes):
        original_hist = create_histogram([conf[c] for conf in original_confidences])
        sr_hist = create_histogram([conf[c] for conf in sr_confidences])
        total_loss += histogram_loss(original_hist, sr_hist)
    return total_loss

def avg_confidence_difference(original_confidences: List[List[float]], 
                              sr_confidences: List[List[float]]) -> float:
    """
    Calculate the Average Difference in Predictive Confidence metric.
    
    :param original_confidences: List of confidence scores for each class for original images
    :param sr_confidences: List of confidence scores for each class for super-resolved images
    :return: Average Difference in Predictive Confidence
    """
    differences = []
    for orig_conf, sr_conf in zip(original_confidences, sr_confidences):
        orig_entropy = calculate_entropy(orig_conf)
        sr_entropy = calculate_entropy(sr_conf)
        differences.append(abs(orig_entropy - sr_entropy))
    return np.mean(differences)

def evaluate_emotion_consistency(original_confidences: List[List[float]], 
                                 sr_confidences: List[List[float]], 
                                 num_classes: int) -> Tuple[float, float]:
    """
    Evaluate emotion consistency using both metrics.
    
    :param original_confidences: List of confidence scores for each class for original images
    :param sr_confidences: List of confidence scores for each class for super-resolved images
    :param num_classes: Number of emotion classes
    :return: Tuple of (Histogram Loss, Average Difference in Predictive Confidence)
    """
    hist_loss = histogram_loss_metric(original_confidences, sr_confidences, num_classes)
    avg_diff = avg_confidence_difference(original_confidences, sr_confidences)
    return hist_loss, avg_diff

# # Example usage
# if __name__ == "__main__":
#     # Simulated data
#     num_samples = 1000
#     num_classes = 7  # Assuming 7 emotion classes
    
#     # Generate random confidence scores for demonstration
#     original_confidences = [np.random.dirichlet(np.ones(num_classes)) for _ in range(num_samples)]
#     sr_confidences = [np.random.dirichlet(np.ones(num_classes)) for _ in range(num_samples)]
    
#     hist_loss, avg_diff = evaluate_emotion_consistency(original_confidences, sr_confidences, num_classes)
    
#     print(f"Histogram Loss: {hist_loss}")
#     print(f"Average Difference in Predictive Confidence: {avg_diff}")
