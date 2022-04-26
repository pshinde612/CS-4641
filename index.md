### Stock Price Prediction using Machine Learning

## Introduction

In this project, we use the Long short-term memory (LSTM) model, a recurrent neural network (RNN), to predict the S&P 500 index, which is a market-wide index that describes the combined stock performance of all the companies in the S&P 500. We decided to use LSTM to forecast the index because it is especially well suited to making predictions on time series data (index movement with time). We gathered time-series index data ranging 5 years (2017 to 2022) using the Yahoo Finance API. Specifically, we used the closing price of the index for our time-series forecasting. We created our dataset by choosing the index value over k successive days as one row and the value on the k+1 day as the corresponding label. For example, if 1,2,3,4,5,6 were the close prices on different days, our dataset would look like 1,2,3 - 4 I 2,3,4 - 5 I 3,4,5 - 6, where ‘I’ denotes a new row and ‘-‘ denotes the label on the right-hand side for the particular sequence of prices. Therefore, the features in our dataset are nothing but the value of the index on different days, and in order to pick the minimal, viable number of features we implemented a Forward Feature Selection. All in all, we used LSTM to learn the dependencies and patterns within the Close price itself. Moreover, we also used Linear Regression and Support Vector Regression models to compare our original results. Over the years, many researchers have used the LSTM model to predict the prices of financial securities. However, there have been many papers that explore the theoretical aspects of LSTM. One example of an empirical study is the research done by Chen et al, where they used the LSTM method to predict stock returns within the Chinese Stock Market [1]. Another example includes research done by Sakowski et al, where they used LSTM to predict Bitcoin prices [2]. Therefore, LSTM is a well-researched tool within the world of financial time series forecasting.

## Problem Definition

Our interest is to see if machine learning can be used to predict the future price of stocks in the market. Because of the unpredictiability of the stock market, we will see if creating a neural network can predict whether the price of a stock will rise in the future. Using machine learning and LSTM recurrent neural networks, we hope to find a pattern that indicates whether the stock price will rise or fall in the future.

Based on the results from the Midterm part of the project, we wanted to analyze how our model did compared to other models, so we chose to create a linear regression and an SVR model to compare the models' respective performance. 

## Data Collection
For this LSTM, we collected the data using the yahoo finance API. We collected data on the stock SPY, which is an stock that tracks a large group of stocks in the S&P 500 index, and is generally regarded as a metric against which all other stocks' growth are measured. To do this, we used a package called yfinance (pulls from theYahoo Finance API) which can be installed in python, and then pulled the close price, open price, adjusted close price, and volume from yahoo finance for a specified date range. In order to make our data as recent as possible, we pulled data from 4/1/2017 to 4/1/2022, collecting these daily metrics for each day and downloading them into a pandas dataframe. For our LSTM, we are using adjusted close prices at different day ranges, so we started by cleaning the data to only include the adjusted close price column. Finally, we normalized the data so that all values were between 0 and 1, and began our forward feature selection.

For the linear regression and SVR models, the same data collection methods were used as indicated in the LSTM section. 

## Methods

The methods for building the model that will successfully be able to predict a certain stock’s future performance will use supervised learning because we will use the next day's price prediction as the label. Moreover, the feature engineering technique that we used to come up with the optimal number of features in our dataset was forward feature selection, wherein we kept adding a new feature column until our loss measure (RMSE) stabilized. By estimating this optimal measure,  the period of days over which the LSTM works can best predict the future price of the stock. To scale the data and calculate the r-squared and mean-squared error, we can use the sci-kit learn metrics package in python for these estimations. Once we run our LSTMs, we can use these calculations to pick the overall model that provides the best r-squared and has the least RMSE and use that optimized dataset for the selected LSTM. Once the dataset dimensionality is optimized, the supervised machine learning model used will be a Long Short-Term Memory (Liu) using the Keras LSTM package in tensor flow. We used a fully connected 3-layer neural network where the first two layers contained 30 and 60 LSTM neurons respectively, and the last layer was just one dense neuron that outputted the predicted index value. LSTMs are particularly useful because it stores relevant information that recently happened but begin to ignore further dated information (Moghar).
For the linear regression model, the data was split in the same 80/20 train/test split as the LSTM. The goal was to compare the performance of this model to that of the LSTM. For the number of features, we kept it fixed at 5 in order to keep the comparisons fair. Using this period of days, the sci-kit learn package was used to scale the data and to calculate the r-squared and MSE. Once the data was scaled, the linear regression model was fit to the data, and predictions were made on the test data.
For the SVR model, the data was also split in the same 80/20 train/test split as the LSTM and regression. Similar to the linear regression model, the goal was also to compare the performance of this supervised model to that of the LSTM. Again, we reused the optimized feature number of 5 in order to make fair comparisons. Using that period of days, the sci-kit learn package was used to scale the data to determine the goodness of fit of the data. Moreover, certain parameters inputs into the SVR function (like gamma for example) were tuned to obtain the best fit. 


## Results and Discussion

Our goal with this project is to use the LSTM model to predict stock prices with accuracy. Through repeated experimentation, we attempted to determine the optimal combination of parameters to produce the most accurate results. The first step was to determine the optimal number of previous days to use when predicting the next day's index as discussed in the Methods Section. A value of 10 features allowed for a small enough RMSE to produce accurate predictions. Using this value for the number of features, the results of training the LSTM model are depicted in the image below, where the "True" values (shown in blue) are the next day's index, and the "Predicted" values (shown in orange) are the model's output value for the next day based on learned relationships in the data.

![image](https://user-images.githubusercontent.com/73048076/161885571-16ca25b6-191e-4cc8-bd88-7a614fd6c450.png)


Because the stock market value is a continuous variable, an R^2 analaysis a good measurement of the goodness of fit between our prediction and the true label and the accuracy of the model. This value was 0.9340314843395934, meaning that variance of the next day's index is approximately 93.4% explained by the variance in the previous 10 days. In other words, our LSTM model was able to learn a function that tends to explain the next day's stock index, but not with perfect accuracy. This value should be as close to 1 as possible for a perfect model but our current model still yields a good score which is non negative. The normalized RMS value(normalized using standard deviation) is around 0.25 which is a good enough value and shows that there is no overfitting of the data. If you examine the graph, it is apparent that our prediction tends to undervalue the actual label for the next day. Moving forward with the project, it is our goal to continue experimentation to reduce the gap between the predicted values and the labels. However, thus far our LSTM model seems quite promising and well-suited for a time sensitive problem such as this.


The goal of the linear regression was similar in that it was used to predict the stock prices with accuracy and compare that to the results of the LSTM. The first step was to determine the optimal number of previous days to use when predicting the next day's index value, which was outlined in the previous "Methods" section using the forward feature selection algorithm. Using this value for the number of features, the result of the test data from the linear regression is seen below with the ground-truth values for the next day index in blue and the predicted values seen in orange. 


![Linear Regression](https://raw.githubusercontent.com/jrmartin11/CS-4641/main/linRegGraph.png?token=GHSAT0AAAAAABT7AGD4TWSTEX6GFEQVFOMAYTIORPQ)


R^2 was used as a measurement for the goodness of fit between the prediction and truth labels for the same reasons outlined above in the LSTM section. The value was 0.9430188149045843. Based on this, the regression model was also able to learn the trends of the stock market to accurately predict the next day's index price. It is also better that it is not 1 since that would likely mean that our data was overfitted. Similarly, the RMSE appears to show that the data is not overfitted with a value of 4.35023410428782. Although this a little high, it does not appear that there is overfitting of the data. 

When we perform SVR we use 3 parameters to get our fit. The kernel type selected is the radial bias kernel. An RBF allows us to get a better fit of data by projecting from 2D to a higher dimension. The parameter C is a measure of amount of misclassification we want to include in our hyperplane selection. Here C was chosen to be 1000 which is pretty high and leans towards a harder SVM. The gamma parameter defines how points close or far from the boundary line are considered in determining the hyperplane. We conducted analysis for different gamma values mainly 0.1 and 0.001. A lower gamma value means that points further from the boundary line are taken into consideration for the hyperplane placement. The lower gamma value resulted in a better performing model as shown in the graphs below and considered for further analysis. The rmse value for SVR is around 21.36981866149278. This reflects a poor performance compared to linear regression and the LSTM neural net models. For the SVR we again use R^2 as a measure of goodness of fit. Our value of R^2 is slightly negative equal to around -0.375 indicating that there might be a slight overfitting of the data. This proves that given unobserved data the SVR model would perform worse than the LSTM model.



![SVR](https://raw.githubusercontent.com/jrmartin11/CS-4641/main/SVRgraph.JPG?token=GHSAT0AAAAAABT7AGD5RQ3XAMZYG5PS5DXOYTIOTQQ)

SVR graph with gamma = 0.1

![image](https://user-images.githubusercontent.com/73048076/165398954-d0116a9e-86bf-4fc6-81bc-5b1faf74ba2f.png)

SVR graph with gamma = 0.01



##Conclusion


## References

Moghar, Adil, and Mhamed Hamiche. “Stock Market Prediction Using LSTM Recurrent Neural Network.” Science Direct, Procedia Computer Science, 2020, https://www.sciencedirect.com/science/article/pii/S1877050920304865. 

S. Liu, G. Liao and Y. Ding, "Stock transaction prediction modeling and analysis based on LSTM," 2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA), 2018, pp. 2787-2790, doi: 10.1109/ICIEA.2018.8398183. 

Ghorbani M, Chong EKP (2020) Stock price prediction using principal components. PLoS ONE 15(3): e0230124. https://doi.org/10.1371/journal.pone.0230124

Michanków, Jakub, et al. “LSTM in Algorithmic Investment Strategies on BTC and S&amp;P500 Index.” MDPI, 25 Jan. 2022, pp. 1–23., https://doi.org/https://www.google.com/url?sa=t&amp;rct=j&amp;q=&amp;esrc=s&amp;source=web&amp;cd=&amp;ved=2ahUKEwiGlfeSmpL2AhXtmmoFHfqkDsIQFnoECAgQAQ&amp;url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F22%2F3%2F917%2Fpdf&amp;usg=AOvVaw3xH-4IzHgmyDk5QGlFPSwJ.

K. Chen, Y. Zhou and F. Dai, "A LSTM-based method for stock returns prediction: A case study of China stock market," 2015 IEEE International Conference on Big Data (Big Data), 2015, pp. 2823-2824, doi: 10.1109/BigData.2015.7364089.
