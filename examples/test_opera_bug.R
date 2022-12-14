library(opera)
targets <- read.csv('./data/targets.csv')
targets <- targets$x
experts <- read.csv('./data/experts.csv')

# on désactive l'expert 2
awake <-  t(replicate(nrow(experts), c(1, 0, 1)))

mod_1 <- mixture(Y = targets[1:100], 
                 experts = experts[1:100, ], 
                 awake = awake[1:100, ], 
                 model = "BOA", 
                 loss.type = "square", 
                 loss.gradient = FALSE)


mod_1$weights
mod_1$prediction
mod_1$coefficients

# mod_1$coefficients !!!
# [1] 8.079298e-14 1.000000e+00 1.136944e-15

# TO CHECK : R prev without awake
predict(mod_1, 
        newexperts = experts[-c(1:100), ], 
        awake = awake[-c(1:100), ],
        online = FALSE, type = "response")

predict(mod_1, 
        newexperts = experts[-c(1:100), ], 
        online = FALSE, type = "response")
as.matrix(experts[-c(1:100), ]) %*% t(t(mod_1$coefficients))

plot(mod_1, dynamic=TRUE)
