#install.packages("riskRegression")
#install.packages("prodlim")
library(ggplot2)
library(riskRegression)
library(prodlim)


# Complex Model AUC
meniscus_lab <- read.csv2('./results (model auc)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model auc)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model auc)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model auc)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model auc)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model auc)/Complex/complex-acl-prediction.csv', sep = ',')

# Merge label-pred score 
men_acc <- merge(meniscus_lab, meniscus_pred, by="Case", all=TRUE)
abn_acc <- merge(abn_lab, abn_pred, by = "Case", all=TRUE)
acl_acc <- merge(acl_lab, acl_pred, by = "Case", all=TRUE)


# mod <- lm(Score.x ~ Score.y, data = meniscus)
# intercept <- mod$coefficients[1]
# slope <- mod$coefficients[2]


meniscus_wu_lab <- read.csv2('./results (model wu)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_wu_pred <- read.csv2('./results (model wu)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_wu_lab <- read.csv2('./results (model wu)/Complex/complex-abnormal-label.csv', sep = ',')
abn_wu_pred <- read.csv2('./results (model wu)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_wu_lab <- read.csv2('./results (model wu)/Complex/complex-acl-label.csv', sep = ',')
acl_wu_pred <- read.csv2('./results (model wu)/Complex/complex-acl-prediction.csv', sep = ',')


men_wu <- merge(meniscus_lab, meniscus_pred, by = "Case", all = TRUE)
abn_wu <- merge(abn_lab, abn_pred, by = "Case", all = TRUE)
acl_wu <- merge(acl_lab, acl_pred, by = "Case", all = TRUE)


fb1=glm(abn_acc$Score.x~abn_acc$Score.y,data=abn_acc,family ="binomial")
fb2=glm(abn_wu$Score.x~abn_wu$Score.y,data=abn_wu,family="binomial")
xb=Score(list(model1=fb1,model2=fb2),Score.y~Score.x,data = abn_wu, split.method="loob",
         plots="cal")
plotCalibration(xb,brier.in.legend=TRUE)
title(main = "Calibration Curve Model Blue and Model Green - Class Abnormal (Complex Case)")


fb1=glm(Score.x~Score.y,data=acl_acc,family="binomial")
fb2=glm(Score.x~Score.y,data=acl_wu,family="binomial")
xb=Score(list(model1=fb1,model2=fb2),Score.y~Score.x,data = acl_wu,
         plots="cal")
plotCalibration(xb,brier.in.legend=TRUE)
title(main = "Calibration Curve Model Blue and Model Green - Class ACL (Complex Case)")

# Validation set

meniscus_lab <- read.csv2('./results (model auc)/Test Set/meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model auc)/Test Set/meniscus-prediction.csv', sep = ',')
meniscus <- merge(meniscus_lab, meniscus_pred, by="Case", all=TRUE)


meniscus_lab_wu <- read.csv2('./results (model wu)/Test Set/meniscus-label.csv', sep = ',')
meniscus_pred_wu <- read.csv2('./results (model wu)/Test Set/meniscus-prediction.csv', sep = ',')
meniscus_wu <- merge(meniscus_lab_wu, meniscus_pred_wu, by = "Case", all = TRUE)

fb1=glm(meniscus$Score.x~meniscus$Score.y,data=meniscus,family ="binomial")
fb2=glm(meniscus_wu$Score.x~meniscus_wu$Score.y,data=meniscus_wu,family="binomial")
xb=Score(list(model1=fb1,model2=fb2),Score.y~Score.x,data = meniscus_wu,
         plots="cal")
plotCalibration(xb,brier.in.legend=TRUE)
title(main = "Calibration Curve Model Blue and Model Green - Class Abnormal (Validation Set)")


