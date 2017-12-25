rm(list=ls())
library(ggplot2)
library(animation)
library(gganimate)
library(R6)

line_func <- function(x, w , b){
  return((-b - x * w[1]) / w[2])
}

line_func_vec <- function(x, w , b){
  return((-b - x * w[,1]) / w[,2])
}

gaussianKernel <- function(x1, x2, sigma = 1){
  diff <- sweep(data.matrix(x1),2,data.matrix(x2))
  x <- apply(diff, 1, norm, type = "2")

  return(exp(-(x^2)/(2*sigma^2)))
}

polynomialKernel <- function(x1, x2, d = 2){
  return((data.matrix(x1) %*% data.matrix(x2) + 1) ^ d)
}

KernelPerceptron <- R6Class("KernelPerceptron",
  public = list(
    X = NULL,
    Y = NULL,
    alpha = NULL,
    alphaSteps = NULL,
    
    train2 = function(X, Y){
      X <- data.matrix(data.frame(b = 1,X))
      self$X = X
      self$Y = Y
      alpha <- rep(0,dim(X)[1])
      alphaSteps <- matrix(0, ncol = dim(X)[1])[-1,]
      
      repeat {
        error <- FALSE
        for(i in seq(1,dim(X)[1])){
          if(Y[i] * ((alpha * Y) %*% (polynomialKernel(X, X[i,]))) <= 0 ){
            alpha[i] <- alpha[i] + 1
            alphaSteps <- rbind(alphaSteps, alpha)
            error <- TRUE
          }
        }
        
        if(!error){
          break
        }
      }
      
      print(alphaSteps)
      self$alpha <- alpha #  every alpha_i > 0 is a support vector
      self$alphaSteps <- alphaSteps
      return(list(alpha = alpha))
    },
    
    plotSteps = function(){
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      
      for(i in seq(dim(self$alphaSteps)[1])){
        
        rangeX = seq(x1Min, x1Max, by = (x1Max - x1Min) / 200)
        rangeY = seq(x2Min, x2Max, by = (x2Max - x2Min) / 200)
        gg <- expand.grid(x=rangeX,y=rangeY)
        gg$z <- apply(gg[,1:2], 1, self$predict, self$alphaSteps[i,])
        
        dfX = data.frame(self$X, self$Y)
        colnames(dfX) <- c("b", "x1", "x2", "y")
        
        p <- ggplot(dfX, aes(x=x1, y=x2)) +
          geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
          geom_raster(data = gg, aes(x = x, y = y, fill = factor(z)), alpha = 1/5) +
          coord_cartesian(ylim=c(x2Min, x2Max)) +
          scale_color_manual(values = c("#DC0026", "#457CB6")) + 
          scale_fill_manual(values = c("#DC0026", "#457CB6"))
        
        print(p)
      }
    },
    
    predict = function(X, alpha = self$alpha){
      X <- data.matrix(c(1,unlist(X)))
      
      return(sign((alpha * self$Y) %*% polynomialKernel(self$X, X)))
    }
    
  )
)

set.seed(0)

N <- 10
X <- data.frame(matrix(rnorm(2 * N, mean=0, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=2, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')
X$y = c(rep(-1, N), rep(1, N))

kPctr <- KernelPerceptron$new()
bb <- kPctr$train2(X[,1:2], X[,3])
print(bb)

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- kPctr$predict(X[i,1:2])
  res <- c(res, pred)
}

result <- data.frame(Y = X[,3], Pred = res)

kPctr$plotSteps()