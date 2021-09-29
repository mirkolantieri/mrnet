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

men_roc <- rocit(score= meniscus$Score.y,class= meniscus$Score.x, method = "bin")
abn_roc <- rocit(score= abn$Score.y,class= abn$Score.x, method = "bin")
acl_roc <- rocit(score= acl$Score.y,class= acl$Score.x, method = "empirical")

plot(men_roc, legend=FALSE, col = c("blue", "orange", "red"))
title(main="ROC Curve (Validation Set) - Model Blue")
lines(abn_roc$TPR~abn_roc$FPR, col = "red" )
lines(acl_roc$TPR~acl_roc$FPR, col = "green" )
legend("bottomright", col = c("blue", "red", "green"),
       c("Meniscus", "Abnormal",
         "ACL"), lwd = 2)

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

men_roc <- rocit(score= meniscus$Score.y,class= meniscus$Score.x, method = "bin")
abn_roc <- rocit(score= abn$Score.y,class= abn$Score.x, method = "bin")
acl_roc <- rocit(score= acl$Score.y,class= acl$Score.x, method = "empirical")

plot(men_roc, legend=FALSE, col = c("blue", "orange", "red"))
title(main="ROC Curve (Validation Set) - Model Green")
lines(abn_roc$TPR~abn_roc$FPR, col = "red" )
lines(acl_roc$TPR~acl_roc$FPR, col = "green" )
legend("bottomright", col = c("blue", "red", "green"),
       c("Meniscus", "Abnormal",
         "ACL"), lwd = 2)

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


