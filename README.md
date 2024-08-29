# Network-Traffic-Classification
This project focuses on building a binary classification model using a Support Vector Machine (SVM). The model is trained to classify data based on a given feature set from my own 'UNSW_NB15_testing-set.csv' dataset. After preprocessing the data and splitting it into training and testing sets, the SVM model was trained and evaluated, achieving high performance in terms of accuracy, precision, recall, and F1-score.

Key Points:
Objective: Classify network traffic data (normal or malicious) using SVM.
Dataset: 'UNSW_NB15_testing-set.csv'
Evaluation Metrics: Accuracy (96.6%), Precision (96.8%), Recall (98.2%), F1-score (97.5%).

The 'UNSW_NB15_testing-set.csv' file contains data with 45 columns and 175,341 rows. The columns represent various features that appear to network traffic analysis, for cybersecurity purposes. Here's a breakdown of some of the key columns:
-id: Identifier for each entry.
-dur: Duration of the network session.
-proto: Protocol used (e.g., TCP).
-service: Type of service (e.g., FTP).
-state: State of the network session (e.g., FIN).
-spkts, dpkts: Number of packets sent and received.
-sbytes, dbytes: Number of bytes sent and received.
-rate: Data rate of the session.
-attack_cat: The category of the attack (e.g., Normal).
-label: A binary label indicating whether the traffic is normal or associated with an attack.

The data is structured to help analyze and categorize network traffic, for identifying and classifying network attacks. The presence of columns like attack_cat and label suggests that this dataset might be used for training or testing machine learning models in network intrusion detection.
