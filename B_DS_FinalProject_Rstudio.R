library(tidyverse)
library(dplyr)
library(tidymodels)
library(readr)
library(ggplot2)
library(gapminder)
library(rsample)
library(margins)
library(caret)
library(stargazer)
library(lattice)


#____________________________________________________
# DATA WORK: 
#____________________________________________________ 


###################################################################################################
# Task one: Inspect and discuss variables of interest (for you) from your data, including the outcome
# and potentially relevant predictors. Offer tabular and customized visual summaries (at least one) 
# and highlight relevant aspects.
####################################################################################################


#_____________________________________________
# LOADING THE DATA

load("~/Desktop/Introduction to Data Science for Business and Social Applications/Final Exam/exam-2021.RData")

#_____________________________________________
# UID, WRANGLING AND JOINGING 

# I start of by finding out what is uniquely identifying, and creating a UID based on that (cik and year):
companies <- mutate(companies, uid = paste(cik, year, sep = "-")) 
reports <- mutate(reports, uid = paste(cik, year, sep = "-"))

# Then i compute a left_join to create a full data set of the two
names <- left_join(companies, reports, by = "uid")

head(names)

name <- names %>%
  select(cik.x, year.x, at, capx,ceq, ch, cogs, dltt, dpact, ni, ppegt, sale, wcap, uid, text, name_change)

head(name)

name <- name %>% 
  rename(cik=cik.x, year=year.x)


#____________________________________________
# SUMMARIZING 
dim(name)

summary(name)

# Table 1
str(name)





###################################################################################################
# Task two: Inspect and summarize the 8-K texts and link these to features of interest from the 
# companies data. Offer tabular and customized visual summaries (at least one) and highlight relevant
# aspects of the texts (1) and the texts in conjunction with company features (2).
###################################################################################################

#_____________________________________________
# WORKING WITH TEXT AS DATA 

library(tidytext)

name_text <- name %>% unnest_tokens(word, text)

#
name_text %>%
  slice_head(n = 12)

name_text <- name_text %>% anti_join(stop_words)

# most frequent word, no sentiment (Table 2)
name_text %>%
  count(word, sort = TRUE) %>%
  slice_head(n=30)

# adds sentiment - keeps only words with sentiment attached from bing
name_text_senti <- name_text %>% inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE)

# Figure 1 
name_text_senti %>%
  group_by(sentiment) %>%
  slice_max(n, n = 15) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>% 
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) + 
  facet_wrap(~sentiment, scales = "free_y") + 
  theme_minimal() +
  scale_fill_manual(values = c("#D16103", "#F4EDCA")) +
  labs(x = "Frequency of words", y = NULL,
       title = "Words that contribute to positive and negative sentiment within the 8-K filings",
       subtitle = "Based entirely on words available within the Bing sentiment dictionary")



#____________________________________________________
# MODELLING: 
#____________________________________________________ 


###################################################################################################
# Task one: Using company characteristics (not the texts), build, evaluate and interpret a model 
# suitable for predicting name changes.
####################################################################################################

# To make my logit model i start out by creating a proper data set only containing the relevant
# variables (sorting out uid, year, text and cik)
logit <- name %>%
  select(at, capx, ceq, ch, cogs, dltt, dpact, ni, ppegt, sale, wcap, name_change)

# then i convert the dependent variable into the right factor format:
logit$name_change <- as.factor(logit$name_change)


# and from here i am ready to start my modelling:
set.seed(200)

logit_split <- logit %>% rsample::initial_split(prop = 0.8)
logit_train <- training(logit_split )
logit_test <- testing(logit_split )

m1_logit <- glm(name_change ~ ., data = logit_train, family = binomial (link = "logit")) #specifies assumed distribution and link function

coef_m1_logit <- coef(m1_logit)

stargazer(coef_m1_logit, type = "text", title = "Table 3: Model Coefficients for Model 1 - Logistic regression (log odds ratio)", out = "coef_m1_logit.html")

summary(m1_logit)

# To be able to interpret the coefficients i have to calculate the average marginal effects for each predictor 
ame_m1_logit <- margins(m1_logit) #returns a dataset with predicted probabilities and individual marginal effects
est_ame_m1_logit <- as.data.frame(summary(ame_m1_logit)) # summarizes into actual AME's for each predictor

est_ame_m1_logit

# the Average Marinal effect for each predictor in a nice format (Table 3)
stargazer(est_ame_m1_logit, type = "text", title = "Table 5: The Average Marginal Effect for each predictor variable",summary = FALSE, out = "m1_logit_ame.html")


#__________________________________
# IN SAMPLE PREDICTED PROBABILITIES 
logit_train$pred_prob <- predict(m1_logit, type = "response")

logit_train$pred_class_correct <- ifelse(logit_train$pred_prob > median(logit_train$pred_prob), 1, 0)

confm_in <- confusionMatrix(factor(logit_train$name_change),
                            factor(logit_train$pred_class_correct))
confm_in
  

recall(factor(logit_train$name_change),
       factor(logit_train$pred_class_correct))

precision(factor(logit_train$name_change),
          factor(logit_train$pred_class_correct))

#______________________________________
# OUT OF SAMPLE PREDICTED PROBABILITIES 
logit_test$pred_prob <- predict(m1_logit, newdata = logit_test, type = "response")

logit_test$pred_class_correct <- ifelse(logit_test$pred_prob > median(logit_test$pred_prob), 1, 0)

confm_out <- confusionMatrix(factor(logit_test$name_change),
                             factor(logit_test$pred_class_correct))
confm_out

recall(factor(logit_test$name_change),
       factor(logit_test$pred_class_correct))

precision(factor(logit_test$name_change),
          factor(logit_test$pred_class_correct))




###################################################################################################
# Task two: Using the 8-K reports, build, evaluate and interpret a regularized model suitable for 
# predicting name changes. Make sure to document and discuss any text transformations you carry out
####################################################################################################

#___________________________
# REGUALIZED MODEL 

library("quanteda")
library(tidyverse)
library(dplyr)
install.packages("pacman")


pacman::p_load(tidyverse, quanteda, ggplot2, glmnet, caret, pROC, dplyr)

# Making a seperate dataset with only uid, text and name_change
filing <- name %>%
  select(uid, text, name_change)

# how many name changed or not
table(filing$name_change, useNA = "always")


# the name_change variable is already in a 1 or 0 format

# now we make a quanteda corpus which is the first step we need to to in order to work with the text data 
filing_corp <- corpus(filing$text, #which column contains the text are specified here
                  docvars = data.frame(name_change = filing$name_change, # here we specify the variables describing each text
                                       uid       = filing$uid))


# here we tokenize the corpus using unigrams - we are breaking up the 8-K filings so we dont have the full texts - we break them into unigrams, 
# we want to count the number of times each word appears in the filings. A uni-gram is a one word sequence - we want the words seperately
filing_dfm <- filing_corp %>%
  tokens() %>%
  dfm(remove = stopwords("en")) 

#then we save it as a df 
docvars(filing_dfm) <- data.frame(name_change = filing$name_change,
                              uid       = filing$uid)

# trim away very infrequent tokens - we specify that a token has to occur a minimum of two times to be included
filing_dfm <- dfm_trim(filing_dfm, min_docfreq = 2)
dim(filing_dfm)



filing_dfm[1:6, 1:6]

# we start out by setting a seed for reproducability 
set.seed(200)

# we already have a uid for all observations 
uid <- filing$uid

# draw a random 80 % of these ID's 
# save them in a vector
training <- sample(1:length(uid), 
                   floor(.80*length(uid)))

# take all the obs that are not in the training set 
# make a vector of those ID's
test <- (1:length(uid))[1:length(uid) %in% training == FALSE]


# make vectors contaiong the ID's for training and test set 
train_id <- uid[training]
test_id <- uid[test]

# now subset the dfm 
train_dfm <- dfm_subset(filing_dfm, uid %in% train_id)
test_dfm <- dfm_subset(filing_dfm, uid %in% test_id)


#______________________________________
#__ RIDGE FOR TEXT __#
#_____________________________________
# first we cross-validate 
ridge <- cv.glmnet(x = train_dfm, y = docvars(train_dfm, "name_change"), 
                   family = "binomial", alpha = 0, nfolds = 5, intercept = TRUE, #alpha = 0 -> ridge 
                   type.measure = "class")

# we reach into ridge and find out what is the lambda with the lowest classification error  
# that lamda puts a ridge penalization of 7,72 on our model - then we can estimate our ridge regression 
# using that lambda value 
ridge$lambda.min
# [1] 7,72

# then we estimate the model with the lowest cross-validated classification error 
ridge_fit <- glmnet(x=train_dfm, y = docvars(train_dfm, "name_change"), 
                    alpha = 0, lambda = ridge$lambda.min,
                    standardize = TRUE,
                    family = "binomial")

# we plot the cross validated classification errors with this command: (to see the MSE for the different lambda values)
plot(ridge)

# The plot displays the cross-validation error according to the log of lambda. The left dashed vertical line indicates that the log of the optimal value of lambda is approximately -5, which is the one that minimizes the prediction error. This lambda value will give the most accurate model. The exact value of lambda can be viewed as follow:



# extract the top predictive features for each regularization type:
#___________________________________________________________________

# we can look a which words that are most important for each results from each model, 
# which will tell us something about what the model found most important in the filings when classifying 
# we first need to extract the top ten most important words from the ridge regression, we do that by
# finding the ridge model with the best lambda:

best.lambda_ridge <- which(ridge$lambda == ridge$lambda.min)

beta_ridge <- ridge$glmnet.fit$beta[,best.lambda_ridge] 

sum(beta_ridge != 0)
# 20600 non zero betas - ridge can't remove so there is a lot more here
# there is a bit more than 20804 values in the full data set, why are the ridge not returning this value?
# there are more predictors than observations = the predictor disappears - if they are perfectly correlated (if two words always go together), 
# then the coefficient is undefined and the stat program just picks one


# we now create a df containing the ridge coefficients, the name/word that has been given weight, 
# then we want to arrange it to show the most important first:
word_ridge <- data.frame(ridge_est = as.numeric(beta_ridge),
                         ridge_choice = names(beta_ridge),
                         stingsAsFactors = FALSE) %>%
  arrange(desc(ridge_est)) %>%
  head(10)




# how do we choose the optimal threshold? -> ROC
# using training data: 
# to draw an ROC curve we use predict and get an object with predictions based on the rigde regression:
train_ridge <- predict(ridge_fit, newx = train_dfm, # newx = specify the training data, we have to do this when we work with glmnet
                       type = "response")
# second we add to this the actual manually coded classified ''impoliteness'' - the thing we are predicting:
# then we a predicted probability in the first column and the actual humanly classified impoliteness
train_ridge <- data.frame(train_ridge,
                          docvars(train_dfm, "name_change"))

#lets name the two: 
names(train_ridge) <- c("train_prob", "true_class")

# the way we draw our ROC curve is by running the ROC command: 
train_roc <- roc(train_ridge$true_class, train_ridge$train_prob)

# lets visualize the ROC curve for all the different thresholds along our predicted probabilities
# we can use the ROC curve to make a decision about the optimal threshold:
plot(train_roc)

# for that we reach in to the ROC curve object and in there we can get all the different thresholds, the sensitivities / TRUE POSITIVE RATE, and the TRUE NEGATIVE RATE under specificities
dec <- data.frame(thres = train_roc$thresholds, # thresholds
                  tp = train_roc$sensitivities, # true positive,
                  tf = train_roc$specificities) # true negative


# if we put equal weight on the true positive rate and the true negative rate, what we wanna do is choose the one
# with the threshold that minimizes the difference between the two, we calculate the absolute difference:
dec$dist <- abs(dec$tp - dec$tf)

# we can get the threshold associated with the minimum value like this: 
dec %>% 
  filter(dist == min(dist)) %>%
  dplyr::select(thres)
# [1] 1 0.2306192
# we can see that the threshold associated with the smallest difference between the TPR and the TNR has a threshold where 
# above 0.048 in predicted probability we classify the tweet as impolite, if under then we classify it as not impolite =  polite 

auc(train_ridge$true_class, train_ridge$train_prob)
# Area under the curve: 1 --> in sample AUC




#_____________________________________________________________________________
# Lets move OUT OF SAMPLE and use the threshold found above in our predictions FOR RIDGE
#_____________________________________________________________________________


pred_ridge <- predict(ridge_fit, newx = test_dfm, # newx = we now tell it that we move out of sample = test set 
                      type = "response") 
# we want to combine the human reading of the tweets with the predicted probabilities in the data set and name these appropriately: 
pred_ridge <- data.frame(pred_ridge,
                         docvars(test_dfm, "name_change"))
names(pred_ridge) <- c("pred_prob", "true_class")

# then we can get the ROC curve again (this time out of sample): 
p_roc <- roc(pred_ridge$true_class, pred_ridge$pred_prob)
plot(p_roc)
# as we see the ROC is much smaller out of sample than in sample 

# we can also get the out of sample AUC:
auc(pred_ridge$true_class, pred_ridge$pred_prob)
# Area under the curve: 0,95, we want AUC to be high - if the auc is smaller out of sample than in sample it might suggest overfitting 


# if we want to create a nicer looking ROC curve
roc_df <- data.frame (sens = p_roc$sensitivities,
                      spec = p_roc$specificities)
ggplot(roc_df, aes(x = 1-spec, y = sens)) +
  geom_line() +
  theme_classic() +
  geom_abline(intercept = 0, slope = 1) +
  annotate(geom = "text", x = 0.23, y = 0.5,
           label = "Performance of a/nrandom classifier") 

# next we can calculate accuracy, precision and recall, using the minimum/best threshold - the one with the lowest diff between TPR and FPR:
# first we classify all tweets as impolite if they have a predicted probability above 0.23:
pred_ridge$pred_name_change <- ifelse(pred_ridge$pred_prob > 0.23, 1, 0)

# next we compute the confusionmatrix, precision and recall: 
confusionMatrix(factor(pred_ridge$true_class), 
                factor(pred_ridge$pred_name_change))

# Reference
# Prediction  0  1
#          0 36  5
#          1  2 29

# Accuracy : 0.9028 

precision(factor(pred_ridge$true_class), 
          factor(pred_ridge$pred_name_change))
# [1] 0.8780 # high!

recall(factor(pred_ridge$true_class), 
       factor(pred_ridge$pred_name_change))
# [1] 0.9473 # high!







#______________________________________
#__ LASSO FOR TEXT __#
#_____________________________________

set.seed(200)

lasso <- cv.glmnet(x = train_dfm, y = docvars(train_dfm, "name_change"),
                   family = "binomial", alpha = 1, nfolds = 5, # for lasso we set alpha to be 1 
                   parallel = TRUE, intercept = TRUE,
                   type.measure = "class")

lasso_fit <- glmnet(x=train_dfm, y = docvars(train_dfm, "name_change"), 
                    alpha = 1, lambda = lasso$lambda.min, # alpha 1 here also bcus lasso
                    standardize = TRUE,
                    family = "binomial")

lasso$lambda.min
# [1] 0.04527

plot(lasso)

# hvad sker der her?
# we are trying to find the optimal lambda value,by finding the sweet spot between bias and variance = we add some 
# bias through lasso and ridge regression to do this

#ridge and lasso = we add some punishment to each predictor in our OLS model - we subtract something in each coefficient - can be a little and a lot
# we let the lambda value decide how much of the coefficients we want to subtract. With lasso we can make variables zero which excludes them
# ridge can not exclude, but only lower.
# we try to find which lambda value gives the best prediction 
# we select the lambda with the lowest error - in the plot each dot is a classification error - we choose the lowest one to be our lambda
# it happens with cross-validation - with 5 folds
# we do cross-validation to predict the best possible model - it helps us to split the data set so we don't overfit 
# we always exclude one part of the data in each of the five folds - therefore we don't contaminate the data - we always add data that has not been used/contaminated = avoid overfitting




# extract the top predictive features for each regularization type:
#___________________________________________________________________

# we can look a which words that are most important for each results from each model, 
# which will tell us something about what the model found most important in the tweets when classifying 
# we first need to extract the top ten most important words from the lasso regression, we do that by
# finding the lasso model with the best lambda:
best.lambda <- which(lasso$lambda == lasso$lambda.min)
# 47  --> the lambda value of 47

# the cross validated model saved all the coefficients under beta
beta <- lasso$glmnet.fit$beta[,best.lambda] # [, best.lambda] -> we need the 30th of our model estimations, that is the 30th run of our cross-validation 
# ^that gets us the best betas 

sum(beta != 0)
#20 non zero betas - lasso sorted ALOT of them out by setting them to zero (removing them)

# we now create a df containing the lasso coefficient, the name/word that has been given weight, 
# then we want to arrange it to show the most important first:
word_lasso <- data.frame(lasso_est = as.numeric(beta),
                         lasso_choice = names(beta),
                         stingsAsFactors = FALSE) %>%
  arrange(desc(lasso_est)) %>%
  head(15)

head(word_lasso, n = 20)
# lasso_est lasso_choice stingsAsFactors
# 1  0.80254333   detergents           FALSE
# 2  0.67436704   attraction           FALSE
# 3  0.17090085         clos           FALSE
# 4  0.16276885         turf           FALSE
# 5  0.08455913          box           FALSE
# 6  0.04097677   soliciting           FALSE
# 7  0.03153051        perox           FALSE
# 8  0.02745265 compensatory           FALSE
# 9  0.02712629   conference           FALSE
# 10 0.02360207   registrant           FALSE

stargazer(word_lasso, type = "text", title = "Table 8: The Top 15 most important words according to Model 2 - Lasso Regression", summary = FALSE, out = "word_lasso.html")



#___________________________
#ROC and AUC FOR LASSO
#___________________________
# how do we choose the optimal threshold? -> ROC
# using training data: 
# to draw an ROC curve we use predict and get an object with predictions based on the lasso regression:
train_lasso <- predict(lasso, newx = train_dfm, # newx = specify the training data, we have to do this when we work with glmnet
                       type = "response", s = lasso$lambda.min)
# second we add to this the actual manually coded classified ''impoliteness'' - the thing we are predicting:
# then we a predicted probability in the first column and the actual humanly classified impoliteness
train_lasso <- data.frame(train_lasso,
                          docvars(train_dfm, "name_change"))

#lets name the two: 
names(train_lasso) <- c("train_prob", "true_class")

# the way we draw our ROC curve is by running the ROC command: 
train_roc_lasso <- roc(train_lasso$true_class, train_lasso$train_prob)

# lets visualize the ROC curve for all the different thresholds along our predicted probabilities
# we can use the ROC curve to make a decision about the optimal threshold:
plot(train_roc_lasso)

# for that we reach in to the ROC curve object and in there we can get all the different thresholds, the sensitivities / TRUE POSITIVE, and the TRUE NEGATIVE RATE under specificities
dec_lasso <- data.frame(thres = train_roc_lasso$thresholds, # thresholds
                        tp = train_roc_lasso$sensitivities, # true positive,
                        tf = train_roc_lasso$specificities) # true negative


# if we put equal weight on the true positive rate and the true negative rate, what we wanna do is choose the one
# with the threshold that minimizes the difference between the two, we calculate the absolute difference:
dec_lasso$dist <- abs(dec_lasso$tp - dec_lasso$tf)

# we can get the threshold associated with the minimum value like this: 
dec_lasso %>% 
  filter(dist == min(dist)) %>%
  dplyr::select(thres)
# [1] 0.30243
# we can see that the threshold associated with the smallest difference between the TPR and the TNR has a threshold where 
# above 0.30243 in predicted probability we classify the company as having changes name, if under then we classify it as no name change  

auc(train_lasso$true_class, train_lasso$train_prob)
# Area under the curve: 0.9988 --> in sample AUC



#_____________________________________________________________________________
# Lets move OUT OF SAMPLE and use the threshold found above in our predictions FOR LASSO
#_____________________________________________________________________________


pred_lasso<- predict(lasso, newx = test_dfm, # newx = specify the test data, we have to do this when we work with glmnet
                     type = "response", s = lasso$lambda.min)

# we want to combine the human reading of the tweets with the predicted probabilities in the data set and name these appropriately: 
pred_lasso <- data.frame(pred_lasso,
                         docvars(test_dfm, "name_change"))

names(pred_lasso) <- c("pred_prob", "true_class")

# then we can get the ROC curve again (this time out of sample): 
p_roc_lasso <- roc(pred_lasso$true_class, pred_lasso$pred_prob)
plot(p_roc_lasso)
# as we see the ROC is smaller out of sample than in sample  - overfitting might have happened... or maybe not

# we can also get the out of sample AUC:
auc(pred_lasso$true_class, pred_lasso$pred_prob)
# Area under the curve: 0.9874, we want AUC to be high - if the auc is smaller out of sample than in sample it might suggest overfitting 


# next we can calculate accuracy, precision and recall, using the minimum/best threshold - the one with the lowest diff between TPR and FPR:
# first we classify all filings as have changed name if they have a predicted probability above 0.30:
pred_lasso$pred_name_change <- ifelse(pred_lasso$pred_prob > 0.30, 1, 0)

# next we compute the confusionmatrix, precision and recall: 
confusionMatrix(factor(pred_lasso$true_class), 
                factor(pred_lasso$pred_name_change))

# Confusion Matrix and Statistics

# Reference
# Prediction  0  1
#          0 38  3
#          1  2 29

# Accuracy : 0.9306 


precision(factor(pred_lasso$true_class), 
          factor(pred_lasso$pred_name_change))
# [1] 0.9268 # high!

recall(factor(pred_lasso$true_class), 
       factor(pred_lasso$pred_name_change))
# [1] 0.95 # high!













# CF's, Uncertainty and Booststrapping 

confint_m1_logit <- confint(m1_logit)

stargazer(confint_m1_logit, type = "text", title = "Table 4: Confidence intervals for Model 1 - Logistic Regression (log odds ratio)", out = "confint.html")







meanCH <- mean(logit$ch)
meanCH
# 239.1942

N <- nrow(logit)
varCH <- sum( (logit$ch - meanCH)^2 ) / N
sdCH <- sqrt(varCH)
sdCH
# 1430.252


sd_mean <- data.frame(Parameter = c("Mean", "Standard deviation"), 
           Value = c(meanCH, sdCH))

sd_mean

stargazer(sd_mean, type = "text", out = "sd_mean.html")





# EXTRA: 

library("tidyverse")
install.packages("texreg")
library("texreg")

names(logit)
set.seed(200)

m2 <- glm(name_change ~ ., family = binomial(link = "logit"),
          data = logit)

m2_uncertainty <- expand.grid(at = mean(logit$age, na.rm = TRUE),
                          capx = mean(logit$capx, na.rm = TRUE),
                          ceq = mean(logit$ceq, na.rm = TRUE),
                          ch = mean(logit$ch, na.rm = TRUE),
                          cogs = mean(logit$cogs, na.rm = TRUE),
                          dltt = mean(logit$dltt, na.rm = TRUE),
                          dpact = mean(logit$dpact, na.rm = TRUE),
                          ni = mean(logit$ni, na.rm = TRUE),
                          ppegt = mean(logit$ppegt, na.rm = TRUE),
                          sale = mean(logit$sale, na.rm = TRUE),
                          wcap = mean(logit$wcap, na.rm = TRUE))

preds <- predict(m2, m2_uncertainty, se.fit = TRUE, type = "response",
                 interval = "prediction")

preds

  