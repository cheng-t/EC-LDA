### EC-LDA: Label Distribution Inference Attack against Federated Graph Learning

This repository contains the official implementation of EC-LDA, a novel label distribution inference attack against Federated Graph Learning.



### Paper Abstract

Graph Neural Networks (GNNs) have been widely used for graph analysis. Federated Graph Learning (FGL) is an emerging learning framework to collaboratively train graph data from various clients. Although FGL allows client data to remain localized, a malicious server can still steal client private data information through uploaded gradient. In this paper, we for the first time propose {\em label distribution attacks} (LDAs\footnote{The term "LDA" here is different from other machine learning terms like Latent Dirichlet Allocation.}) on FGL that aim to infer the label distributions of the client-side data. Firstly, we observe that the effectiveness of LDA is closely related to the variance of node embeddings in GNNs. Next, we analyze the relation between them and propose a new attack named EC-LDA, which significantly improves the attack effectiveness by compressing node embeddings. Then, extensive experiments on node classification and link prediction tasks across six widely used graph datasets show that EC-LDA outperforms the SOTA LDAs. Specifically, EC-LDA can achieve the Cos-sim as high as 1.0 under almost all cases. Finally, we explore the robustness of EC-LDA under differential privacy protection and discuss the potential effective defense methods to EC-LDA.
