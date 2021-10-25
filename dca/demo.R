install.packages("rmda")

library(rmda)


data(dcaData)
head(dcaData)

set.seed(123)
#first use rmda with the default settings (set bootstraps = 50 here to reduce computation time). 
baseline.model <- decision_curve(Cancer~Age + Female + Smokes, #fitting a logistic model
                                 data = dcaData, 
                                 study.design = "cohort", 
                                 policy = "opt-in",  #default 
                                 bootstraps = 50)

#plot the curve
plot_decision_curve(baseline.model,  curve.names = "baseline model")


opt.out.dc <- decision_curve(Cancer~Age + Female + Smokes + Marker1 + Marker2,
                             data = dcaData, 
                             bootstraps = 50, 
                             policy = 'opt-out') #set policy = 'opt-out' (default is 'opt-in')

plot_decision_curve( opt.out.dc,  xlim = c(0, 1), ylim = c(-.2,2), 
                     standardize = FALSE, 
                     curve.names = "model", legend = 'bottomright') 