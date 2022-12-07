library(opera)
setwd('/home/riad/EDF/opera_py/')
targets <- read.csv('./data/targets.csv')['x']
experts <- read.csv('./data/experts.csv')
awake <-  t(replicate(nrow(experts), c(1,0,1)))
MLpol0 <- mixture(Y = targets, experts = experts, model = "BOA", loss.type = "square", loss.gradient = FALSE)
MLpol0$weights

plot(MLpol0, dynamic=TRUE)
typeof(targets)
targets
experts
nrow(experts) == length(Y)
