library(opera)
targets <- read.csv('./data/targets.csv')
targets <- targets$x
experts <- read.csv('./data/experts.csv')
# awake <-  t(replicate(nrow(experts), c(1,1,1)))
MLpol0 <- mixture(Y = targets, experts = experts, model = "BOA", loss.type = "square", loss.gradient = FALSE)
MLpol0$weights

plot(MLpol0, dynamic=TRUE)
