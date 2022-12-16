library(opera)
targets <- read.csv('./data/targets.csv')
targets <- targets$x
experts <- read.csv('./data/experts.csv')

# model = "BOA"
# model = "MLpol"
model = "MLprod"

set_awake = TRUE

if(set_awake){
  awake <-  t(replicate(nrow(experts), c(1, 0, 1)))
} else {
  awake <-  t(replicate(nrow(experts), c(1, 1, 1)))
}

mod_1 <- mixture(Y = targets[1:100], 
                 experts = experts[1:100, ], 
                 awake = awake[1:100, ], 
                 model = model, 
                 loss.type = "square", 
                 loss.gradient = FALSE)


mod_1$weights
mod_1$prediction
mod_1$coefficients

# TO CHECK : R prev without awake
predict(mod_1, 
        newexperts = experts[-c(1:100), ], 
        awake = awake[-c(1:100), ],
        online = FALSE, type = "response")

# plot(mod_1, dynamic=TRUE)
