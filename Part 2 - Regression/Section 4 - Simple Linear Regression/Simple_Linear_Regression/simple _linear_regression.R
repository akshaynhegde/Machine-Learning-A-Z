dataset = read.csv('Salary_Data.csv')

#splitting data into training and test sets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#fitting the Data set to the linear regressson model
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#predict the test test results
predict(regressor, newdata = test_set)

#plot the results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Savary vs Years of Experience') +
  xlab('Years of Experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Savary vs Years of Experience') +
  xlab('Years of Experience') +
  ylab('Salary')