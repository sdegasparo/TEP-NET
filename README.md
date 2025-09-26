# TEP-NET
A Deep Neural Network for TCR-Epitope Binding Prediction Using Physicochemical and Embedding Features

## Abstract
TCR-epitope binding is central to adaptive immunity, nevertheless, predicting these interactions remains a major computational challenge. This paper introduces TEP-NET, a deep learning model that integrates PCA-reduced ProtBERT embeddings with physicochemical features encoded via Piecewise Linear Encoding (PLE). In contrast to most previous studies that relied on synthetic negatives, TEP-NET is trained on a curated dataset containing authentic negative examples, thereby enhancing biological realism. The evaluation of generalisation was conducted by categorising TCR-epitope pairs using the TCR-Peptide Pairing (TPP) framework. The experimental findings demonstrate that 64-dimensional PCA-reduced embeddings achieve the optimal trade-off between accuracy and efficiency. This is evidenced by a 90\% reduction in memory and runtime, while maintaining strong predictive performance. A benchmarking process against four state-of-the-art baselines (EPIC-TRACE, PiTE, epiTCR, and TCR-H) demonstrates that TEP-NET achieves superior performance, particularly in challenging generalisation settings with unseen sequences. Multi-faceted interpretability analyses (saliency and LIME) consistently revealed biologically meaningful importance patterns in TCR and epitope positions and physicochemical descriptors. These results establish TEP-NET as a novel, efficient, and interpretable framework for TCR–epitope binding prediction, relevant for applications in immunotherapy, vaccine design, and computational immunology.

## Requirements
`pip install -r requirements.txt`


## Reduction
Train the desired reduction technique:
- For PCA: `PCA_Fitting.py`
- For Transformer Autoencoder: `Transformer_Autoencoder.py`

## Preprocessing
To prepare the data for the model you have to run the following scripts in this order: Feature_enrichment.py -> Reducing_ProtBERT.py -> ProtBERT_matching.py

Don't forget to define the desired reduction.

## Train Model
Run the chosen model using the preprocessed data and the desired hyperparameters.
