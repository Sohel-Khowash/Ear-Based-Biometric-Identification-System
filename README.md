### 1. Initial Approach: Custom EmbeddingNet
* **Architecture:** Custom 2D CNN with $128 \\times 128 \\times 1$ input, `Conv2D` + `ReLU` layers, `MaxPooling`, Flatten, and Dense embedding layers [cite: 1].
* **Loss:** Softmax / Cross-Entropy [cite: 1].
* **Classifier:** KNN / SVM on extracted features [cite: 1].
* **Performance:** 88–92% Known Class Accuracy, ~30% Unknown Detection Rate [cite: 1].
* **Limitation:** Poor open-set generalization and high false-positive rate for unknown classes [cite: 1].

### 2. Intermediate Approach: EfficientNet-B0
* **Architecture:** Replaced custom CNN backbone with **EfficientNet-B0** [cite: 1].
* **Performance:** 90–94% Known Class Accuracy, ~45% Unknown Detection Rate [cite: 1].
* **Improvement:** Better representation learning and feature abstraction [cite: 1].

### 3. Final Approach: TripletNet (Metric Learning)
* **Loss Function:** **Triplet Loss** to enforce intra-class cohesion and inter-class separation in a 512-dimensional feature embedding space [cite: 1].
* **Embeddings:** 512-D vector output [cite: 1].
* **Post-Embedding Classifiers Evaluated:** K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Logistic Regression using Euclidean and Cosine distances [cite: 1].

---

## 🔍 Unknown / Open-Set Detection Strategies

To address open-set recognition where unseen identities enter the system, multiple detection strategies were evaluated [cite: 1]:

1. **Plain Softmax Thresholding:**
   * Threshold set at 0.70 confidence [cite: 1].
   * Yielded only **18%** unknown detection [cite: 1].
2. **Tuned Confidence Thresholding:**
   * Optimized threshold set at **0.83** [cite: 1].
   * Improved unknown detection rate to **~60%** [cite: 1].
3. **Distance-Based Thresholding:**
   * Nearest-neighbor distance cutoff at **0.50** [cite: 1].
   * Yielded high discrimination for distant feature representations [cite: 1].
4. **Unsupervised Outlier Detection:**
   * **One-Class SVM:** Boundary learning achieved **89%** unknown detection [cite: 1].
   * **Isolation Forest:** Anomaly isolation achieved **55%** unknown detection [cite: 1].
5. **Final Ensemble Detection (Majority Vote):**
   * Combines Confidence Thresholding, Distance Thresholding, One-Class SVM, and Isolation Forest [cite: 1].
   * Achieved an impressive **94.33% Unknown Detection Rate** [cite: 1].

---

## 📈 Performance Summary

| Model Version | Backbone | Loss Function | Classifiers | Known Accuracy | Unknown Detection Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EmbeddingNet** | Custom CNN | Softmax | KNN / SVM | 88–92% | ~30% |
| **EfficientNet** | EfficientNet-B0 | Softmax | KNN / SVM | 90–94% | ~45% |
| **TripletNet (Final)** | EfficientNet-B0 | Triplet Loss | KNN, SVM, Logistic | **96–97%** | **94.33% (Ensemble)** |

---

## 🛠️ Tech Stack & Tools

* **Frameworks & Deep Learning:** PyTorch, Torchvision, Albumentations [cite: 1]
* **Machine Learning & Anomaly Detection:** Scikit-Learn (KNN, SVM, One-Class SVM, Isolation Forest, Logistic Regression) [cite: 1]
* **Visualization & Analysis:** Matplotlib, Seaborn, t-SNE [cite: 1]

---

## 🚀 Key Takeaways & Conclusion

* **Metric Learning Matters:** Shifting to Triplet Loss allowed the network to learn a highly discriminative 512-D embedding space, forming tight clusters for known identities and widely separated regions for unknowns [cite: 1].
* **Pretrained Backbones:** EfficientNet-B0 drastically accelerated feature extraction convergence and enhanced overall system generalization [cite: 1].
* **Ensemble for Open-Set AI:** Combining confidence, distance metrics, One-Class SVM, and Isolation Forest delivered robust rejection of unseen identities, making the system suitable for real-world biometric deployment [cite: 1].
"""

with open("README.md", "w") as f:
    f.write(readme_content.strip())

print("README.md file created successfully.")
