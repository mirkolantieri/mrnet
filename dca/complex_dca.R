# Import library
library(rmda)
library(ROCit)
library(ggplot2)


# Complex Model AUC
meniscus_lab <- read.csv2('./results (model auc)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model auc)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model auc)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model auc)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model auc)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model auc)/Complex/complex-acl-prediction.csv', sep = ',')

# Merge label-pred score 
meniscus <- merge(meniscus_lab, meniscus_pred, by="Case", all=TRUE)
abn <- merge(abn_lab, abn_pred, by = "Case", all=TRUE)
acl <- merge(acl_lab, acl_pred, by = "Case", all=TRUE)

# Create ROC Curve 
men_roc <- rocit(score= meniscus$Score.y,class= meniscus$Score.x, method = "bin")
abn_roc <- rocit(score= abn$Score.y,class= abn$Score.x, method = "bin")
acl_roc <- rocit(score= acl$Score.y,class= acl$Score.x, method = "bin")

plot(men_roc, legend=FALSE, col = c("blue", "orange", "red"))
title(main="ROC Curve of Complex Case - Model Blue")
lines(abn_roc$TPR~abn_roc$FPR, col = "red" )
lines(acl_roc$TPR~acl_roc$FPR, col = "green" )
legend("bottomright", col = c("blue", "red", "green"),
       c("Meniscus", "Abnormal",
         "ACL"), lwd = 2)




baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = meniscus,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Meniscus class (Model Blue)", 
                    col = c("darkblue", "orange"))

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Abnormal class (Model Blue) ", 
                    col = c('red', 'green'),  )

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - ACL class (Model Blue)",
                    col = c("darkorange", "navy"))

# Complex Model WU

meniscus_lab <- read.csv2('./results (model wu)/Complex/complex-meniscus-label.csv', sep = ',')
meniscus_pred <- read.csv2('./results (model wu)/Complex/complex-meniscus-prediction.csv', sep = ',')

abn_lab <- read.csv2('./results (model wu)/Complex/complex-abnormal-label.csv', sep = ',')
abn_pred <- read.csv2('./results (model wu)/Complex/complex-abnormal-prediction.csv', sep = ',')

acl_lab <- read.csv2('./results (model wu)/Complex/complex-acl-label.csv', sep = ',')
acl_pred <- read.csv2('./results (model wu)/Complex/complex-acl-prediction.csv', sep = ',')


meniscus <- merge(meniscus_lab, meniscus_pred, by = "Case", all = TRUE)
abn <- merge(abn_lab, abn_pred, by = "Case", all = TRUE)
acl <- merge(acl_lab, acl_pred, by = "Case", all = TRUE)

# Create ROC Curve 
men_roc <- rocit(score= meniscus$Score.y,class= meniscus$Score.x, method = "bin")
abn_roc <- rocit(score= abn$Score.y,class= abn$Score.x, method = "bin")
acl_roc <- rocit(score= acl$Score.y,class= acl$Score.x, method = "bin")

plot(men_roc, legend=FALSE, col = c("blue", "orange", "red"))
title(main="ROC Curve of Complex Case - Model Green")
lines(abn_roc$TPR~abn_roc$FPR, col = "red" )
lines(acl_roc$TPR~acl_roc$FPR, col = "green" )
legend("bottomright", col = c("blue", "red", "green"),
       c("Meniscus", "Abnormal",
         "ACL"), lwd = 2)


baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = meniscus,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Meniscus class (Model Green)",
                    col = c("blue", "red"))

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = abn,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - Abnormal class (Model Green) ",
                    col = c("orange", "black"))

baseline.model <- decision_curve(Score.x ~ Score.y,
                                 data = acl,
                                 policy = "opt-in",
                                 bootstraps = 50)

plot_decision_curve(baseline.model,  curve.names = "Complex cases - ACL class (Model Green) ",
                    col = c("green", "navy"))

# ROC CURVE WITH PRECISION - RECALL
#library(precrec)
#precrec_obj <- evalmod(scores = meniscus$Score.y, labels = meniscus$Score.x, mode="rocprc")
#plot(precrec_obj)



#ROCit_obj <- rocit(score= meniscus$Score.y,class= meniscus$Score.x, method = "empirical")
#plot(ROCit_obj)
