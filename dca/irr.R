#install.packages("irr")
#install.packages("tidyverse")
library(irr)
library(icr)
library(tidyverse)

# Complex Model AUC - Model WU
men_lab <- read.csv2('./results (model auc)/Complex/complex-meniscus-label.csv', sep = ',')
men_pred <- read.csv2('./results (model auc)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model auc)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model auc)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model auc)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model auc)/Complex/complex-acl-prediction.csv', sep = ',')

meniscus_lab_wu <- read.csv2('./results (model wu)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred_wu <- read.csv2('./results (model wu)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab_wu <- read.csv2('./results (model wu)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred_wu <- read.csv2('./results (model wu)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab_wu <- read.csv2('./results (model wu)/Complex/complex-acl-label.csv', sep = ',')
acl_pred_wu <- read.csv2('./results (model wu)/Complex/complex-acl-prediction.csv', sep = ',')


men_wu <- merge(meniscus_lab_wu, meniscus_pred_wu, by = "Case", all = TRUE)
abn_wu <- merge(abn_lab_wu, abn_pred_wu, by = "Case", all = TRUE)
acl_wu <- merge(acl_lab_wu, acl_pred_wu, by = "Case", all = TRUE)

men <- merge(men_lab, men_pred, by="Case", all=TRUE)
abn <- merge(abn_lab, abn_pred, by="Case", all=TRUE)
acl <- merge(acl_lab, acl_pred, by="Case", all=TRUE)

men_ratings <- men %>% select(Score.x, Score.y)
abn_ratings <- abn %>% select(Score.x, Score.y)
acl_ratings <- acl %>% select(Score.x, Score.y)

men_ratings_wu <- men_wu %>% select(Score.x, Score.y)
abn_ratings_wu <- abn_wu %>% select(Score.x, Score.y)
acl_ratings_wu <- acl_wu %>% select(Score.x, Score.y)


# Overall agreement
print("Model Blue Overall Agreement - Complex Case")
agree(men_ratings)
agree(abn_ratings)
agree(acl_ratings)

print("Model Green Overall Agreement - Complex Case")
agree(men_ratings_wu)
agree(abn_ratings_wu)
agree(acl_ratings_wu)


# Cohen's Kappa
print("Model Blue Cohen's Kappa - Complex Case")
kappa2(men_ratings)
kappa2(abn_ratings)
kappa2(acl_ratings)

print("Model Green Cohen's Kappa - Complex Case")
kappa2(men_ratings_wu)
kappa2(abn_ratings_wu)
kappa2(acl_ratings_wu)


# Krippendorff's alpha 
print("Model Blue Krippendorff's Alpha - Complex Case")
krippalpha(men, metric = "ordinal")
krippalpha(abn, metric = "ordinal")
krippalpha(acl, metric = "ordinal")
print("Model Green Krippendorff's Alpha - Complex Case")
krippalpha(men_wu, metric = "ordinal")
krippalpha(abn_wu, metric = "ordinal")
krippalpha(acl_wu, metric = "ordinal")

# Validation Model AUC - WU

men_test_lab <- read.csv2('./results (model auc)/Test Set/meniscus-label.csv', sep = ',')
men_test_pred <- read.csv2('./results (model auc)/Test Set/meniscus-prediction.csv', sep = ',')

abn_test_lab <- read.csv2('./results (model auc)/Test Set/abnormal-label.csv', sep = ',')
abn_test_pred <- read.csv2('./results (model auc)/Test Set/abnormal-prediction.csv', sep = ',')

acl_test_lab <- read.csv2('./results (model auc)/Test Set/acl-label.csv', sep = ',')
acl_test_pred <- read.csv2('./results (model auc)/Test Set/acl-prediction.csv', sep = ',')


men_test <- merge(men_test_lab, men_test_pred, by="Case", all=TRUE)
abn_test <- merge(abn_test_lab, abn_test_pred, by = "Case", all=TRUE)
acl_test <- merge(acl_test_lab, acl_test_pred, by = "Case", all=TRUE)


men_test_lab_wu <- read.csv2('./results (model wu)/Test Set/meniscus-label.csv', sep = ',')
men_test_pred_wu <- read.csv2('./results (model wu)/Test Set/meniscus-prediction.csv', sep = ',')

abn_test_lab_wu <- read.csv2('./results (model wu)/Test Set/abnormal-label.csv', sep = ',')
abn_test_pred_wu <- read.csv2('./results (model wu)/Test Set/abnormal-prediction.csv', sep = ',')

acl_test_lab_wu <- read.csv2('./results (model wu)/Test Set/acl-label.csv', sep = ',')
acl_test_pred_wu <- read.csv2('./results (model wu)/Test Set/acl-prediction.csv', sep = ',')


men_test_wu <- merge(men_test_lab_wu, men_test_pred_wu, by="Case", all=TRUE)
abn_test_wu <- merge(abn_test_lab_wu, abn_test_pred_wu, by = "Case", all=TRUE)
acl_test_wu <- merge(acl_test_lab_wu, acl_test_pred_wu, by = "Case", all=TRUE)


men_ratings_test <- men_test %>% select(Score.x, Score.y)
abn_ratings_test <- abn_test %>% select(Score.x, Score.y)
acl_ratings_test <- acl_test %>% select(Score.x, Score.y)

men_ratings_test_wu <- men_test_wu %>% select(Score.x, Score.y)
abn_ratings_test_wu <- abn_test_wu %>% select(Score.x, Score.y)
acl_ratings_test_wu <- acl_test_wu %>% select(Score.x, Score.y)


# Overall agreement
print("Model Blue Overall Agreement - Validation Set")

agree(men_ratings_test)
agree(abn_ratings_test)
agree(acl_ratings_test)

print("Model Green Overall Agreement - Validation Set")
agree(men_ratings_test_wu)
agree(abn_ratings_test_wu)
agree(acl_ratings_test_wu)


# Cohen's Kappa
print("Model Blue Cohen's Kappa - Validation Set")
kappa2(men_ratings_test)
kappa2(abn_ratings_test)
kappa2(acl_ratings_test)
print("Model Green Cohen's Kappa - Validation Set")
kappa2(men_ratings_test_wu)
kappa2(abn_ratings_test_wu)
kappa2(acl_ratings_test_wu)

# Krippendorff

print("Model Blue Krippendorff's Alpha - Validation Set")
krippalpha(men_test, metric = "ordinal")
krippalpha(abn_test, metric = "ordinal")
krippalpha(acl_test, metric = "ordinal")

print("Model Green Krippendorff's Alpha - Validation Set")
krippalpha(men_test_wu, metric = "ordinal")
krippalpha(abn_test_wu, metric = "ordinal")
krippalpha(acl_test_wu, metric = "ordinal")



