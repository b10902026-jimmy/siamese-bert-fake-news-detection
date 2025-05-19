# Fake News Detection using Siamese BERT on LIAR-PLUS Dataset

## Executive Summary

This project addresses the critical challenge of fake news detection using advanced deep learning techniques. We implemented a novel Triple Branch Siamese BERT architecture that leverages not only the news statements but also their justifications and metadata to improve classification accuracy. Our model achieves approximately 74.6% accuracy on the binary classification task (real vs. fake news), demonstrating the effectiveness of our approach.

The key innovations of our approach include:
1. A Siamese network architecture with BERT embeddings
2. Integration of multiple information sources (statements, justifications, and metadata)
3. A credit score mechanism that weighs the credibility of news sources
4. Comprehensive model analysis and visualization techniques for interpretability

This report details our methodology, findings, and analysis, providing insights into the patterns that distinguish real from fake news and offering a robust framework for automated fake news detection.

## 1. Introduction

### 1.1 Problem Statement and Motivation

The proliferation of fake news in today's digital media landscape poses significant challenges to society, potentially influencing public opinion, political outcomes, and social stability. The ability to automatically detect fake news has become increasingly important as the volume of information continues to grow exponentially, making manual fact-checking impractical.

Traditional approaches to fake news detection often rely solely on the content of news statements, ignoring valuable contextual information such as the speaker's credibility, justifications, and metadata. Our project aims to address this limitation by developing a more comprehensive approach that leverages multiple sources of information.

### 1.2 Research Objectives

The primary objectives of this research are:

1. To develop a robust fake news detection model that achieves high accuracy on the LIAR-PLUS dataset
2. To explore the effectiveness of a Siamese BERT architecture for this task
3. To investigate the impact of incorporating multiple information sources (statements, justifications, metadata)
4. To analyze the patterns and characteristics that distinguish real from fake news
5. To provide interpretable results that can enhance our understanding of fake news detection

### 1.3 Dataset Overview

The LIAR-PLUS dataset is an extension of the LIAR dataset, which contains 12,836 short statements labeled for veracity by human fact-checkers. The LIAR-PLUS dataset enhances the original by adding justifications for the fact-checking decisions. Each statement in the dataset is labeled as one of six categories: "true," "mostly-true," "half-true," "barely-true," "false," or "pants-fire" (completely false).

The dataset also includes metadata such as:
- The speaker making the statement
- The speaker's job title, party affiliation, and state
- The context in which the statement was made
- Credit history (counts of statements in each veracity category)

For our binary classification task, we grouped the labels into two categories:
- Real news: "true," "mostly-true," and "half-true"
- Fake news: "barely-true," "false," and "pants-fire"

## 2. Data Exploration and Analysis

### 2.1 Label Distribution

The LIAR-PLUS dataset exhibits an imbalanced distribution of labels. In the binary classification setting, approximately 56% of statements are categorized as real (true, mostly-true, half-true) and 44% as fake (barely-true, false, pants-fire). This relatively balanced distribution allows for effective model training without requiring extensive class balancing techniques.

### 2.2 Speaker and Party Analysis

Our analysis reveals that certain speakers appear more frequently in the dataset, with politicians being particularly prominent. The top speakers include Barack Obama, Donald Trump, and other political figures. The Republican and Democratic parties dominate the party affiliations in the dataset.

Interestingly, different speakers show varying patterns of truthfulness. Some speakers consistently make more truthful statements, while others tend toward false claims. This pattern suggests that speaker identity could be a valuable feature for fake news detection.

### 2.3 Text Length Analysis

The length of statements varies considerably across the dataset, with most statements containing between 10 and 30 words. Justifications tend to be longer, typically ranging from 50 to 300 words. Our analysis shows a slight correlation between statement length and veracity, with true statements being marginally longer on average than false ones.

### 2.4 Credit Score Analysis

We developed a credit score mechanism that weighs the historical credibility of news sources. The credit score is calculated as a weighted average of the speaker's history of true and false statements:

```
Credit Score = (0.2*barely_true + 0.1*false + 0.5*half_true + 0.8*mostly_true + 1.0*true) / (sum of all counts)
```

This score ranges from 0 to 1, with higher values indicating more credible sources. Our analysis shows a positive correlation between credit scores and statement veracity, confirming the intuition that speakers with better track records tend to make more truthful statements.

### 2.5 Word Frequency Analysis

Word frequency analysis reveals distinct patterns in the vocabulary used in true versus false statements. True statements tend to contain more specific, factual terms, while false statements often include more emotional or exaggerated language. Common words in true statements include terms related to legislation, economics, and specific policies, while false statements more frequently contain superlatives and emotionally charged words.

## 3. Methodology

### 3.1 Model Architecture Evolution

Our approach evolved through three increasingly sophisticated architectures:

#### 3.1.1 Single Branch BERT

The initial model used a standard BERT architecture fine-tuned for classification. This model processed only the news statements and achieved approximately 60% accuracy on the binary classification task.

#### 3.1.2 Dual Branch Siamese BERT

The second iteration implemented a Siamese network with two BERT branches sharing weights. One branch processed the news statements, while the other processed the justifications. The outputs were concatenated and passed through a fully connected layer for classification. This model achieved approximately 65.4% accuracy.

#### 3.1.3 Triple Branch Siamese BERT

Our final and most effective architecture is a Triple Branch Siamese BERT network. This model includes:
- Branch 1: Processes news statements (sequence length: 64)
- Branch 2: Processes justifications (sequence length: 256)
- Branch 3: Processes metadata (sequence length: 32)

The outputs from all three branches are concatenated and combined with the credit score before being passed through a fully connected layer for classification. This model achieved approximately 74.6% accuracy on the binary classification task.

### 3.2 Implementation Details

The model was implemented using PyTorch and the pytorch_pretrained_bert library. We used the pre-trained BERT-base-uncased model as the foundation for our architecture. Key implementation details include:

- BERT model: bert-base-uncased (12 layers, 768 hidden size, 12 attention heads)
- Optimizer: Adam with learning rate 1e-5 for BERT parameters and 1e-4 for classification layer
- Loss function: Cross-entropy loss
- Batch size: 32
- Training epochs: 5
- Sequence lengths: 64 for statements, 256 for justifications, 32 for metadata

### 3.3 Credit Score Integration

The credit score was integrated into the model by adding it to the concatenated BERT outputs before classification. This approach allows the model to consider the speaker's historical credibility when making predictions.

## 4. Results and Analysis

### 4.1 Performance Metrics

Our Triple Branch Siamese BERT model achieved the following performance on the test set:

- Accuracy: 74.6%
- Precision (Real): 73%
- Recall (Real): 87%
- Precision (Fake): 78%
- Recall (Fake): 59%
- F1-score (weighted average): 74%

These results demonstrate that our model performs well on the binary classification task, with particularly high recall for real news and precision for fake news.

### 4.2 Confusion Matrix Analysis

The confusion matrix reveals that our model is more likely to misclassify fake news as real (229 instances) than real news as fake (93 instances). This pattern suggests that the model is somewhat conservative in labeling statements as fake, which may be desirable in applications where falsely accusing a statement of being fake is more problematic than failing to detect some fake news.

### 4.3 Error Analysis

Our error analysis identified several patterns in misclassifications:

1. Statements with ambiguous language or partial truths are often misclassified
2. Statements from speakers with inconsistent credibility histories pose challenges
3. Statements requiring specific domain knowledge for verification are more difficult to classify correctly
4. Very short statements provide limited linguistic cues for classification

### 4.4 Model Comparison

Comparing our three model architectures:

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Single Branch BERT | 60.0% | 0.61 | 0.60 | 0.60 |
| Dual Branch Siamese BERT | 65.4% | 0.66 | 0.65 | 0.65 |
| Triple Branch Siamese BERT | 74.6% | 0.75 | 0.75 | 0.74 |

These results clearly demonstrate the benefits of incorporating multiple information sources through our Siamese architecture.

## 5. Model Interpretability

### 5.1 Feature Importance

To understand which features contribute most to the model's decisions, we analyzed the attention weights from the BERT model. The analysis revealed that:

1. In true statements, the model pays more attention to specific facts, numbers, and references
2. In false statements, the model focuses more on exaggerated claims and emotional language
3. The justification branch contributes significantly to the model's decisions, particularly for ambiguous statements
4. The metadata branch helps identify patterns related to specific speakers or contexts

### 5.2 Case Studies

#### Example 1: High-Confidence Correct Prediction (True Statement)
Statement: "Statistics indicate that one in eight children, and one in 18 adults in Oregon suffers from mental illness."
- Predicted: Real (Confidence: 0.9999)
- Actual: Real
- Analysis: The model correctly identified this as a true statement, likely due to the specific statistics and neutral language.

#### Example 2: High-Confidence Incorrect Prediction (False Statement)
Statement: "When Atlanta Police Chief George Turner was interim head of the department, overall crime fell 14 percent and violent crime dropped 22.7 percent."
- Predicted: Real (Confidence: 0.9972)
- Actual: Fake
- Analysis: The model incorrectly classified this statement as real, possibly due to the specific statistics and official context, which are common features in true statements.

### 5.3 Linguistic Patterns

Our analysis of word usage patterns revealed distinctive linguistic features that differentiate real from fake news:

1. Real news tends to use more precise language, specific numbers, and references to verifiable sources
2. Fake news often employs more superlatives, emotional language, and vague references
3. Real news typically presents balanced perspectives, while fake news may present one-sided arguments
4. Fake news sometimes uses more complex sentence structures to obscure false claims

## 6. Discussion

### 6.1 Interpretation of Results

The performance of our Triple Branch Siamese BERT model demonstrates the value of incorporating multiple information sources for fake news detection. The significant improvement from the Single Branch (60%) to the Triple Branch model (74.6%) confirms our hypothesis that contextual information such as justifications and metadata enhances detection accuracy.

The model's higher recall for real news and precision for fake news suggests a conservative approach to labeling statements as fake. This behavior may be appropriate for applications where falsely accusing a statement of being fake carries higher costs than missing some fake news.

### 6.2 Limitations

Despite its strong performance, our approach has several limitations:

1. Reliance on justifications: In real-world scenarios, justifications may not be readily available for new statements
2. Domain specificity: The model is trained on political statements and may not generalize well to other domains
3. Language constraints: The model is trained on English text and may not perform well on other languages
4. Temporal limitations: News veracity may change over time as new information becomes available
5. Computational requirements: The triple BERT architecture is computationally intensive, which may limit deployment in resource-constrained environments

### 6.3 Ethical Considerations

Automated fake news detection raises several ethical considerations:

1. Potential for bias: Models may inherit biases present in the training data
2. Freedom of expression: Automated labeling of content as "fake" could impact freedom of expression
3. Transparency: The black-box nature of deep learning models may reduce transparency in content moderation
4. Accountability: Determining responsibility for misclassifications is challenging with automated systems
5. Adversarial manipulation: Bad actors may attempt to manipulate the model to avoid detection

## 7. Conclusion and Future Work

### 7.1 Summary of Contributions

This project has made several key contributions to the field of fake news detection:

1. Developed a novel Triple Branch Siamese BERT architecture that leverages multiple information sources
2. Demonstrated the effectiveness of incorporating speaker credibility through a credit score mechanism
3. Provided comprehensive analysis and visualization of model performance and error patterns
4. Identified linguistic patterns that differentiate real from fake news

### 7.2 Future Directions

Several promising directions for future work include:

1. Exploring more sophisticated methods for integrating the credit score mechanism
2. Incorporating temporal information to track changes in statement veracity over time
3. Extending the model to handle multi-modal inputs (text, images, videos)
4. Developing more interpretable models that can provide explanations for their decisions
5. Investigating transfer learning approaches to adapt the model to other domains and languages
6. Implementing adversarial training to improve robustness against manipulation attempts

### 7.3 Practical Applications

Our fake news detection model has potential applications in various domains:

1. Social media platforms: Automated flagging of potentially false content
2. Journalism: Assisting fact-checkers by prioritizing statements for verification
3. Education: Teaching critical media literacy by highlighting characteristics of fake news
4. Research: Studying patterns and evolution of misinformation
5. Public policy: Informing strategies to combat misinformation

## 8. References

1. LIAR-PLUS dataset: https://github.com/Tariq60/LIAR-PLUS
2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
3. Attention Is All You Need: https://arxiv.org/abs/1706.03762
4. GloVe: Global Vectors for Word Representation: https://nlp.stanford.edu/projects/glove/
5. Where is your Evidence: Improving Fact-checking by Justification Modeling: https://aclweb.org/anthology/W18-5513

## 9. Appendix: Visualizations

The project includes various visualizations to aid in understanding the data and model performance:

1. Confusion matrices (regular and normalized)
2. Classification performance metrics by class
3. Error distribution analysis
4. Word clouds for real and fake news statements
5. Credit score distributions
6. Speaker truthfulness analysis

These visualizations provide valuable insights into the patterns and characteristics of fake news, helping to enhance our understanding of this complex phenomenon.
