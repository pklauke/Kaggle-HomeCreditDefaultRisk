# Kaggle-HomeCreditDefaultRisk

This repository contains my solution for the Kaggle competition <a href='https://www.kaggle.com/c/home-credit-default-risk'>Home Credit Default Risk</a>. It ranked 248 out of 7198 (Top 4%). The goal of this competition was to predict how capable credit applicants are of repaying the loan.

My solution consists of a combination of 3 models:
  * Gradient Boosted Decision Tree
  * Scheduled denoising Autoencoder + Gradient Boosted Decision Tree
  * Scheduled denoising Autoencoder + Neural Net
  
875 features were created. 269 of these were binary columns. Some of those came from one-hot-encoded categoric columns or adding binary columns indicating that the value of a column was missing. The missing values were replaced by a very high negative number (pure LightGBM) or mean imputation (Autoencoder). The architecture of the Autoencoder was 875-1000-1000-1000-875. The data was scaled using RankGauss transformation. The InputSwap noise was reduced during training to capture different kind of features. The features of the Neural net and the Gradient Boosted Decision tree were the 3000 activations of the Autoencoder, reconstruction mean squared error as anomaly detection feature and the 3 strongest features of the original data. An overview can be found in the notebook <a href='https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk/blob/master/Modelling.ipynb'>Modelling.ipynb</a>.
  
The predictions of the 3 models were ensembled using weighted blending (<a href='https://github.com/pklauke/mlopt'>BlendingOptimizer</a>) and averaged with the <a href='https://github.com/neptune-ml/open-solution-home-credit/tree/solution-5'>NeptuneML Open Solution</a>.
