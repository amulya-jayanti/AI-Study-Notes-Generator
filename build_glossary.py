"""
Complete AI/ML/Data Science Glossary Builder
Creates comprehensive glossary with 115 high-quality terms
"""

import json
import os
import pandas as pd

def create_complete_glossary():
    """Create complete AI/ML/DS glossary with all 115 terms"""
    
    glossary = {
        "metadata": {
            "created_for": "NLP Study Notes Generator - AI/ML/Data Science",
            "total_terms": 115,
            "subjects": ["machine_learning", "deep_learning", "nlp", "data_science", 
                        "statistics_probability", "computer_vision"],
            "domain": "Artificial Intelligence, Machine Learning, Data Science",
            "purpose": "Ground truth glossary for evaluation"
        },
        "terms": []
    }
    
    # All 115 terms organized by subject
    all_terms = {
        "machine_learning": [
            {"term": "supervised learning", "definition": "Machine learning paradigm where model learns from labeled training data with input-output pairs", "category": "ML Paradigm"},
            {"term": "unsupervised learning", "definition": "Learning from unlabeled data to discover hidden patterns, structures, or groupings", "category": "ML Paradigm"},
            {"term": "reinforcement learning", "definition": "Learning through interaction with environment using rewards and penalties to maximize cumulative reward", "category": "ML Paradigm"},
            {"term": "overfitting", "definition": "Model performs well on training data but poorly on unseen test data due to excessive complexity", "category": "Problem"},
            {"term": "underfitting", "definition": "Model is too simple to capture underlying patterns in data, performing poorly on both train and test", "category": "Problem"},
            {"term": "bias-variance tradeoff", "definition": "Balance between model's ability to fit training data and generalize to new data", "category": "Concept"},
            {"term": "cross-validation", "definition": "Technique assessing model performance by partitioning data into multiple train/test splits", "category": "Technique"},
            {"term": "hyperparameter", "definition": "Configuration variable set before training that controls learning process", "category": "Parameter"},
            {"term": "gradient descent", "definition": "Optimization algorithm iteratively adjusting parameters to minimize loss using gradients", "category": "Algorithm"},
            {"term": "loss function", "definition": "Objective function measuring how well model predictions match actual values", "category": "Metric"},
            {"term": "regularization", "definition": "Technique preventing overfitting by adding penalty term to loss function", "category": "Technique"},
            {"term": "feature engineering", "definition": "Process creating, transforming, or selecting features to improve model performance", "category": "Process"},
            {"term": "ensemble learning", "definition": "Combining predictions from multiple models to achieve better overall performance", "category": "Technique"},
            {"term": "bagging", "definition": "Bootstrap aggregating - training multiple models on random subsets and averaging predictions", "category": "Ensemble Method"},
            {"term": "boosting", "definition": "Sequentially training models where each focuses on correcting previous errors", "category": "Ensemble Method"},
            {"term": "random forest", "definition": "Ensemble of decision trees trained on random subsets of features and data", "category": "Algorithm"},
            {"term": "decision tree", "definition": "Tree-structured model making decisions by splitting data based on feature values", "category": "Algorithm"},
            {"term": "k-nearest neighbors", "definition": "Algorithm predicting based on k closest training examples in feature space", "category": "Algorithm"},
            {"term": "support vector machine", "definition": "Algorithm finding optimal hyperplane separating classes with maximum margin", "category": "Algorithm"},
            {"term": "learning rate", "definition": "Hyperparameter controlling step size in gradient descent optimization", "category": "Hyperparameter"}
        ],
        "deep_learning": [
            {"term": "neural network", "definition": "Computing system with interconnected layers of nodes inspired by biological neurons", "category": "Architecture"},
            {"term": "activation function", "definition": "Non-linear function applied to neuron output enabling complex pattern learning", "category": "Function"},
            {"term": "backpropagation", "definition": "Algorithm computing gradients of loss with respect to network weights using chain rule", "category": "Algorithm"},
            {"term": "convolutional neural network", "definition": "Architecture specialized for grid-like data using convolutional layers", "category": "Architecture"},
            {"term": "recurrent neural network", "definition": "Network with cyclic connections allowing information persistence across sequences", "category": "Architecture"},
            {"term": "LSTM", "definition": "Long Short-Term Memory - RNN variant with gates handling long-term dependencies", "category": "Architecture"},
            {"term": "transformer", "definition": "Architecture using self-attention mechanism to process sequences in parallel", "category": "Architecture"},
            {"term": "attention mechanism", "definition": "Allows model to dynamically focus on relevant input parts when predicting", "category": "Mechanism"},
            {"term": "dropout", "definition": "Regularization randomly deactivating neurons during training to prevent overfitting", "category": "Regularization"},
            {"term": "batch normalization", "definition": "Normalizing layer inputs across mini-batch to stabilize and accelerate training", "category": "Technique"},
            {"term": "transfer learning", "definition": "Using pre-trained model knowledge to improve performance on new related task", "category": "Technique"},
            {"term": "fine-tuning", "definition": "Adapting pre-trained model to new task by continuing training on specific data", "category": "Technique"},
            {"term": "embedding", "definition": "Dense low-dimensional vector representation capturing semantic relationships", "category": "Representation"},
            {"term": "autoencoder", "definition": "Network learning compressed representation by reconstructing input through bottleneck", "category": "Architecture"},
            {"term": "GAN", "definition": "Generative Adversarial Network - generator and discriminator competing for realism", "category": "Architecture"},
            {"term": "epoch", "definition": "One complete pass through entire training dataset during model training", "category": "Training Concept"},
            {"term": "batch size", "definition": "Number of training examples processed together in one forward/backward pass", "category": "Hyperparameter"},
            {"term": "optimizer", "definition": "Algorithm updating network weights based on gradients (Adam, SGD, RMSprop)", "category": "Algorithm"},
            {"term": "vanishing gradient", "definition": "Problem where gradients become too small in deep networks, hindering learning", "category": "Problem"},
            {"term": "exploding gradient", "definition": "Problem where gradients become too large, causing unstable training", "category": "Problem"}
        ],
        "nlp": [
            {"term": "tokenization", "definition": "Splitting text into meaningful units like words, subwords, or characters", "category": "Preprocessing"},
            {"term": "word embedding", "definition": "Dense vector representation capturing semantic and syntactic word relationships", "category": "Representation"},
            {"term": "TF-IDF", "definition": "Term Frequency-Inverse Document Frequency - weights words by importance", "category": "Metric"},
            {"term": "named entity recognition", "definition": "Identifying and classifying named entities like persons, organizations, locations", "category": "Task"},
            {"term": "part-of-speech tagging", "definition": "Assigning grammatical categories to each word in text", "category": "Task"},
            {"term": "sentiment analysis", "definition": "Determining emotional tone or opinion expressed in text", "category": "Task"},
            {"term": "language model", "definition": "Probabilistic model predicting word probability given context", "category": "Model"},
            {"term": "BERT", "definition": "Bidirectional Encoder Representations from Transformers - pre-trained masked model", "category": "Model"},
            {"term": "GPT", "definition": "Generative Pre-trained Transformer - autoregressive language model", "category": "Model"},
            {"term": "seq2seq", "definition": "Sequence-to-sequence encoder-decoder mapping input to output sequences", "category": "Architecture"},
            {"term": "beam search", "definition": "Decoding algorithm exploring multiple promising candidate sequences simultaneously", "category": "Algorithm"},
            {"term": "perplexity", "definition": "Metric measuring how well probability model predicts text; lower is better", "category": "Metric"},
            {"term": "attention weights", "definition": "Learned scores indicating relative importance of input tokens for output", "category": "Mechanism"},
            {"term": "context window", "definition": "Maximum number of tokens model can process at once", "category": "Parameter"},
            {"term": "zero-shot learning", "definition": "Making predictions on new tasks without any task-specific training examples", "category": "Technique"},
            {"term": "few-shot learning", "definition": "Learning to perform task from just a few examples in prompt", "category": "Technique"},
            {"term": "prompt engineering", "definition": "Crafting effective input prompts to elicit desired model responses", "category": "Technique"},
            {"term": "text classification", "definition": "Assigning predefined categories or labels to text documents", "category": "Task"},
            {"term": "question answering", "definition": "Task where model generates answer to question based on context", "category": "Task"},
            {"term": "machine translation", "definition": "Automatically translating text from one language to another", "category": "Task"}
        ],
        "data_science": [
            {"term": "data preprocessing", "definition": "Cleaning and transforming raw data by handling missing values and outliers", "category": "Process"},
            {"term": "exploratory data analysis", "definition": "Analyzing datasets to summarize characteristics using statistical graphics", "category": "Process"},
            {"term": "feature selection", "definition": "Identifying most relevant features for modeling to reduce dimensionality", "category": "Technique"},
            {"term": "dimensionality reduction", "definition": "Reducing number of features while preserving important information", "category": "Technique"},
            {"term": "PCA", "definition": "Principal Component Analysis - linear technique projecting data onto components maximizing variance", "category": "Algorithm"},
            {"term": "clustering", "definition": "Unsupervised task grouping similar data points based on features", "category": "Task"},
            {"term": "k-means", "definition": "Clustering algorithm partitioning data into k clusters minimizing within-cluster variance", "category": "Algorithm"},
            {"term": "classification", "definition": "Supervised task predicting discrete categorical labels for data points", "category": "Task"},
            {"term": "regression", "definition": "Supervised task predicting continuous numerical values", "category": "Task"},
            {"term": "confusion matrix", "definition": "Table showing true/false positives and negatives for classification evaluation", "category": "Evaluation"},
            {"term": "precision", "definition": "Proportion of positive predictions that are actually correct", "category": "Metric"},
            {"term": "recall", "definition": "Proportion of actual positives correctly identified by model", "category": "Metric"},
            {"term": "F1 score", "definition": "Harmonic mean of precision and recall balancing both metrics", "category": "Metric"},
            {"term": "ROC curve", "definition": "Receiver Operating Characteristic - plots true vs false positive rates", "category": "Evaluation"},
            {"term": "AUC", "definition": "Area Under ROC Curve - measures classifier's discrimination ability", "category": "Metric"},
            {"term": "train-test split", "definition": "Dividing dataset into separate subsets for training and evaluation", "category": "Technique"},
            {"term": "missing data", "definition": "Absent values in dataset requiring imputation or removal strategies", "category": "Problem"},
            {"term": "outlier", "definition": "Data point significantly different from other observations", "category": "Concept"},
            {"term": "imbalanced dataset", "definition": "Dataset where class distribution is heavily skewed toward certain classes", "category": "Problem"},
            {"term": "data leakage", "definition": "Information from test set inadvertently influencing training", "category": "Problem"}
        ],
        "statistics_probability": [
            {"term": "probability distribution", "definition": "Mathematical function describing likelihood of different outcomes", "category": "Concept"},
            {"term": "normal distribution", "definition": "Bell-shaped continuous distribution symmetric around mean", "category": "Distribution"},
            {"term": "mean", "definition": "Average value calculated by summing values and dividing by count", "category": "Statistic"},
            {"term": "median", "definition": "Middle value when data is sorted; robust to outliers", "category": "Statistic"},
            {"term": "standard deviation", "definition": "Measure of spread quantifying average distance from mean", "category": "Statistic"},
            {"term": "variance", "definition": "Average of squared deviations from mean", "category": "Statistic"},
            {"term": "correlation", "definition": "Statistical measure of linear relationship between two variables", "category": "Concept"},
            {"term": "covariance", "definition": "Measure of joint variability of two random variables", "category": "Statistic"},
            {"term": "hypothesis testing", "definition": "Statistical method making inferences about population using sample data", "category": "Method"},
            {"term": "p-value", "definition": "Probability of obtaining results as extreme as observed under null hypothesis", "category": "Statistic"},
            {"term": "confidence interval", "definition": "Range of values likely containing true population parameter", "category": "Statistic"},
            {"term": "Bayes theorem", "definition": "Formula calculating conditional probability and updating beliefs with evidence", "category": "Theorem"},
            {"term": "maximum likelihood", "definition": "Method estimating parameters by maximizing likelihood of observed data", "category": "Method"},
            {"term": "sampling", "definition": "Selecting representative subset of population for inference", "category": "Technique"},
            {"term": "central limit theorem", "definition": "Sample means approach normal distribution as sample size increases", "category": "Theorem"},
            {"term": "chi-square test", "definition": "Statistical test examining independence between categorical variables", "category": "Test"},
            {"term": "ANOVA", "definition": "Analysis of Variance - tests whether means of multiple groups differ significantly", "category": "Test"},
            {"term": "t-test", "definition": "Statistical test comparing means of two groups for significance", "category": "Test"},
            {"term": "Type I error", "definition": "False positive - rejecting null hypothesis when actually true", "category": "Error"},
            {"term": "Type II error", "definition": "False negative - failing to reject null when alternative is true", "category": "Error"}
        ],
        "computer_vision": [
            {"term": "image classification", "definition": "Assigning single label to entire image from predefined categories", "category": "Task"},
            {"term": "object detection", "definition": "Identifying and localizing multiple objects with bounding boxes and labels", "category": "Task"},
            {"term": "semantic segmentation", "definition": "Assigning class label to every pixel in image for dense prediction", "category": "Task"},
            {"term": "instance segmentation", "definition": "Identifying individual object instances and segmenting each separately", "category": "Task"},
            {"term": "convolution", "definition": "Mathematical operation applying learnable filter to extract spatial features", "category": "Operation"},
            {"term": "pooling", "definition": "Downsampling operation reducing spatial dimensions", "category": "Operation"},
            {"term": "stride", "definition": "Step size of filter moving across image during convolution", "category": "Parameter"},
            {"term": "padding", "definition": "Adding border pixels to maintain spatial dimensions after convolution", "category": "Technique"},
            {"term": "ResNet", "definition": "Residual Network - architecture using skip connections for very deep networks", "category": "Architecture"},
            {"term": "VGG", "definition": "Very Deep Convolutional Network - architecture using small 3x3 filters", "category": "Architecture"},
            {"term": "data augmentation", "definition": "Artificially expanding training data by applying transformations", "category": "Technique"},
            {"term": "bounding box", "definition": "Rectangle defined by coordinates specifying object location and size", "category": "Annotation"},
            {"term": "IoU", "definition": "Intersection over Union - measures overlap between predicted and ground truth", "category": "Metric"},
            {"term": "image preprocessing", "definition": "Preparing images through resizing, normalization, color conversion", "category": "Process"},
            {"term": "feature map", "definition": "Output of convolutional layer representing detected features at locations", "category": "Concept"}
        ]
    }
    
    # Combine all terms with subject labels
    for subject, terms in all_terms.items():
        for term in terms:
            term["subject"] = subject
            glossary["terms"].append(term)
    
    return glossary

def main():
    print("=" * 70)
    print("COMPLETE AI/ML/DATA SCIENCE GLOSSARY BUILDER")
    print("=" * 70)
    print("\nCreating comprehensive glossary with 115 terms...")
    
    # Create glossary
    glossary = create_complete_glossary()
    
    # Save JSON
    os.makedirs("datasets/custom", exist_ok=True)
    with open("datasets/custom/custom_glossary.json", "w") as f:
        json.dump(glossary, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame(glossary["terms"])
    df.to_csv("datasets/custom/custom_glossary.csv", index=False)

if __name__ == "__main__":
    main()
