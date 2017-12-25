rm(list=ls())
library(ggplot2)
library(animation)
library(gganimate)
library(R6)

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
    kernel = NULL,
    kernelParam = NULL,
    
    train = function(X, Y, kernel = polynomialKernel, kernelParam = 1){
      X <- data.matrix(data.frame(b = 1,X))
      self$X = X
      self$Y = Y
      self$kernel = kernel
      self$kernelParam = kernelParam
      alpha <- rep(0,dim(X)[1])
      alphaSteps <- matrix(0, ncol = dim(X)[1])[-1,]
      
      repeat {
        error <- FALSE
        for(i in seq(1,dim(X)[1])){
          if(Y[i] * ((alpha * Y) %*% (self$kernel(X, X[i,], self$kernelParam))) <= 0 ){
            alpha[i] <- alpha[i] + 1
            alphaSteps <- rbind(alphaSteps, alpha)
            error <- TRUE
          }
        }
        
        if(!error){
          break
        }
      }
      
      self$alpha <- alpha #  every alpha_i > 0 is a support vector
      self$alphaSteps <- alphaSteps
      return(alpha)
    },
    
    plotSteps = function(){
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      
      for(i in seq(dim(self$alphaSteps)[1])){
        
        rangeX = seq(x1Min, x1Max, by = (x1Max - x1Min) / 200)
        rangeY = seq(x2Min, x2Max, by = (x2Max - x2Min) / 200)
        contour <- expand.grid(x = rangeX, y = rangeY)
        contour$z <- apply(contour[,1:2], 1, self$predict, self$alphaSteps[i,])
        
        dfX = data.frame(self$X, self$Y)
        colnames(dfX) <- c("b", "x1", "x2", "y")
        
        p <- ggplot(dfX, aes(x = x1, y = x2)) +
          geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
          geom_raster(data = contour, aes(x = x, y = y, fill = factor(z)), alpha = 1/5) +
          coord_cartesian(ylim=c(x2Min, x2Max)) +
          scale_color_manual(values = c("#DC0026", "#457CB6")) + 
          scale_fill_manual(values = c("#DC0026", "#457CB6"))
        
        print(p)
      }
    },
    
    plotAnimation = function(){
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      rangeX = seq(x1Min, x1Max, by = (x1Max - x1Min) / 200)
      rangeY = seq(x2Min, x2Max, by = (x2Max - x2Min) / 200)
      
      dfX = data.frame(self$X, self$Y)
      colnames(dfX) <- c("b", "x1", "x2", "y")
      contour <- data.frame()
      contourTemp <- expand.grid(x = rangeX, y = rangeY)
      
      for(i in seq(dim(self$alphaSteps)[1])){
        contourTemp$z <- apply(contourTemp[,1:2], 1, self$predict, self$alphaSteps[i,])
        contourTemp$time <- i
        contour <- rbind(contour, contourTemp)
      }
      
      p <- ggplot(dfX, aes(x = x1, y = x2)) +
        geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
        geom_raster(data = contour, aes(x = x, y = y, fill = factor(z), frame = time), alpha = 1/5) +
        coord_cartesian(ylim=c(x2Min, x2Max)) +
        scale_color_manual(values = c("#DC0026", "#457CB6")) + 
        scale_fill_manual(values = c("#DC0026", "#457CB6"))
      
      gganimate(p)
    },
    
    predict = function(X, alpha = self$alpha){
      X <- data.matrix(c(1,unlist(X)))
      
      return(sign((alpha * self$Y) %*% self$kernel(self$X, X, self$kernelParam)))
    }
    
  )
)

set.seed(0)

N <- 100
X <- data.frame(matrix(rnorm(2 * N, mean=0, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')

# generate classes
# let the class be -1 under the mirror of the hyperbolic cosine function
# and +1 above
X$y <- X[,2] <= -cosh(X[,1]) + 1.5
X$y[X$y == TRUE] <- -1
X$y[X$y == FALSE] <- 1

X$x2[X$y == 1] = X$x2[X$y == 1] + 0.5
# ggplot(X, aes(x = x1, y = x2, color = y)) + geom_point()

kPctr <- KernelPerceptron$new()
alpha <- kPctr$train(X[,1:2], X[,3], kernel = polynomialKernel, kernelParam = 2)
print(alpha)

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- kPctr$predict(X[i,1:2])
  res <- c(res, pred)
}

kPctr$plotSteps()