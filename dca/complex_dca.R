# Import library
library(rmda)


# Complex Model AUC
meniscus_lab <- read.csv2('./results (model auc)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model auc)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model auc)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model auc)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model auc)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model auc)/Complex/complex-acl-prediction.csv', sep = ',')


meniscus <- merge(meniscus_lab, meniscus_pred, by="Case", all=TRUE)
abn <- merge(abn_lab, abn_pred, by = "Case", all=TRUE)
acl <- merge(acl_lab, abn_pred, by = "Case", all=TRUE)

head(meniscus)
head(abn)
head(acl)

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = meniscus,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Meniscus class (Model AUC) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Abnormal class (Model AUC) ", col = c('orange', 'blue'),  )

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - ACL class (Model AUC) ")

# Complex Model WU

meniscus_lab <- read.csv2('./results (model wu)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model wu)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model wu)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model wu)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model wu)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model wu)/Complex/complex-acl-prediction.csv', sep = ',')


meniscus <- merge(meniscus_lab, meniscus_pred, by = "Case", all = TRUE)
abn <- merge(abn_lab, abn_pred, by = "Case", all = TRUE)
acl <- merge(acl_lab, abn_pred, by = "Case", all = TRUE)

head(meniscus)
head(abn)
head(acl)

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = meniscus,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Meniscus class (Model WU) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Abnormal class (Model WU) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - ACL class (Model Wu) ")


