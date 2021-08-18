
# Deep-Learning
# LSTM Stock Predictor

![Banner_Deep_learning](https://user-images.githubusercontent.com/83671629/129911177-8c99be98-9792-4aa0-aae7-1980dd2974ac.jpg)

Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. In this tool I have tried to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In this Tool, I have used deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.

The following initial steps were taken:

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

- - -

### Files

[Closing Prices Starter Notebook](Starter_Code/lstm_stock_predictor_closing.ipynb)

[FNG Starter Notebook](Starter_Code/lstm_stock_predictor_fng.ipynb)

- - -
<img src="https://user-images.githubusercontent.com/83671629/129913577-20f3e400-504b-4f93-9754-4dd319e01074.jpg"  width="400" height="400">

## Procedure

### Preparing the data for training and testing

I created a Jupyter Notebook for each RNN. The Notebook contains a function to create the window of time for the data in each dataset.

For the Fear and Greed model, I used the FNG values to try and predict the closing price. 

For the closing price model, I used the previous closing prices to try and predict the next closing price.

Each model used 70% of the data for training and 30% of the data for testing.

A MinMaxScaler was applied to the X and y values to scale the data for the model.

Finally, reshaped the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

### Building and training custom LSTM RNNs

In each Jupyter Notebook, I created the same custom LSTM RNN architecture. In one notebook, the data was fit using the FNG values. In the second notebook, the data was fit using only closing prices.

The same parameters and training steps were used for each model. This is necessary to compare each model accurately.

### Evaluate the performance of each model

Finally, used the testing data to evaluate each model and compare the performance.

The above info was used to answer the following:

> Which model has a lower loss?
>
> Which model tracks the actual values better over time?
>
> Which window size works best for the model?

- - -

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

- - -

### Hints and Considerations

Experimentd with the model architecture and parameters to see which provides the best results, but used the same architecture and parameters when comparing each model.

For training, used at least 10 estimators for both models.


