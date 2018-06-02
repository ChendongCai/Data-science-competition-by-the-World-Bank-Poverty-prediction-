# Data-science-competition-by-the-World-Bank-Poverty-prediction-

In this data science competition, I need to make poverty status identification strategy for the World Bank based on the analysis and the built-up model.

It is very important to conduct feature engineering so that you can identify the strongest predictors of poverty to help the World Bank run surveys with fewer, more targeted questions.

Also, I predicted the probability of a given household being poor (achieved Mean Log Loss 0.154 and ranked 63 out of 2310 teams).


#In the data preprocessing step, I used a stupid way to combine the individual dataset and the household dataset which made the logic messy and the code complicated. You can easily reach to excatly the same result by performing one-hot encoding, then aggregating the data by household and calculating the average values.

#I will sepecify this in my next repository which is about building a logistic regression from scracth and using this dataset to test.
