# Import library
library(rmda)
library(ROCit)

# Test set Model AUC
meniscus_lab <- read.csv2('./results (model auc)/Test Set/meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model auc)/Test Set/meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model auc)/Test Set/abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model auc)/Test Set/abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model auc)/Test Set/acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model auc)/Test Set/acl-prediction.csv', sep = ',')


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

plot_decision_curve(baseline.model,  curve.names = "Test set cases - Meniscus class (Model AUC) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Test set cases - Abnormal class (Model AUC) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Test set cases - ACL class (Model AUC) ")

# Test set Model WU

meniscus_lab <- read.csv2('./results (model wu)/Test Set/meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model wu)/Test Set/meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model wu)/Test Set/abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model wu)/Test Set/abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model wu)/Test Set/acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model wu)/Test Set/acl-prediction.csv', sep = ',')


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

plot_decision_curve(baseline.model,  curve.names = "Test set cases - Meniscus class (Model WU) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Test set cases - Abnormal class (Model WU) ")

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Test set cases - ACL class (Model Wu) ")


