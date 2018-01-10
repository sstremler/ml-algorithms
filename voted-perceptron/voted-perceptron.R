rm(list=ls())
library(ggplot2)
library(animation)
library(gganimate)
library(R6)

lineFunction <- function(x, w , b){
  return((-b - x * w[1]) / w[2])
}

lineFunctionVec <- function(x, w , b){
  return((-b - x * w[,1]) / w[,2])
}

VotedPerceptron <- R6Class("VotedPerceptron",
  public = list(
    weightSteps = NULL,
    cSteps = NULL,
    X = NULL,
    Y = NULL,
    
    train = function(X, Y, epoch = 2, eta = 1){
      X <- data.matrix(data.frame(b = 1,X))
      weightSteps <- matrix(0, ncol = dim(X)[2])
      cSteps <- c(0)
      self$X <- X
      self$Y <- Y
      k <- 1
      
      while(epoch > 0){
        for(i in seq(1,dim(X)[1])){
          if(sum(weightSteps[k,] * X[i,]) * Y[i] <= 0){
            weightSteps <- rbind(weightSteps, weightSteps[k,] + eta * Y[i] * X[i,])
            cSteps <- c(cSteps, 1)
            k <- k + 1
          } else {
            cSteps[k] <- cSteps[k] + 1
          }
        }
        
        epoch <- epoch - 1
      }
      
      self$weightSteps <- weightSteps
      self$cSteps <- cSteps
      
      return(list(weightSteps = weightSteps, cSteps = cSteps))
    },
    
    predict = function(X){
      return(private$predict_(X, self$weightSteps))
    }
    
  ),
  
  private = list(
    predict_ = function(X, w){
      return(sign(sum(self$cSteps[1:dim(matrix(w, ncol=3))[1]] * sign(c(1,unlist(X)) %*% t(w)))))
    }  
  )
)

set.seed(0)

N <- 100
X <- data.frame(matrix(rnorm(2 * N, mean=1.2, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')
X$y = c(rep(-1, N), rep(1, N))
# ggplot(X, aes(x = x1, y = x2, color = y)) + geom_point()

vPctr <- VotedPerceptron$new()
weight <- vPctr$train(X[,1:2], X[,3], epoch = 5, eta = 0.5)
print(weight)

# vPctr$plotSteps()
# prc$plotAnimation()

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- vPctr$predict(X[i,1:2])
  res <- c(res, pred)
}

trDf = data.frame(Y = X[,3], Pred = res)
trScore = sum(trDf$Y == trDf$Pred) / dim(trDf)[1] *100
print(paste0('Training score: ', trScore, '%'))
