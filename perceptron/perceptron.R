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

Perceptron <- R6Class("Perceptron",
  public = list(
    weight = NULL,
    weightSteps = NULL,
    X = NULL,
    Y = NULL,
    
    train = function(X, Y){
      eta <- 0.05
      w <- c(0, 1, 1)
      X <- data.matrix(data.frame(b = 1,X))
      self$X = X
      self$Y = Y
      error = dim(X)[1]
      
      while(error > 0){
        error = dim(X)[1]
        
        for(i in seq(1,dim(X)[1])){
          if(sum(X[i,] * w) * Y[i] <= 0){
            w = c(unlist(w + Y[i]*eta*X[i,]))
            self$weightSteps <- rbind(self$weightSteps, w)
          } else {
            error = error - 1
          }
        }
        
      }
      
      self$weight <- w
      
      return(w)
    },
    
    predict = function(X){
      return(sign(sum(c(1,unlist(X)) * self$weight)))
    },
    
    plotSteps = function(){
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      
      for(i in seq(dim(self$weightSteps)[1])){
        w <- self$weightSteps[i,]
        
        dfLine <- data.frame(x1 = c(x1Min, x1Max),
                             x2 = c(lineFunction(x1Min, w = w[2:3], b = w[1]), lineFunction(x1Max, w = w[2:3], b = w[1])))
        
        dfX = data.frame(self$X, self$Y)
        colnames(dfX) <- c("b", "x1", "x2", "y")
        
        p <- ggplot(dfX, aes(x=x1, y=x2)) +
          geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
          geom_line(data = dfLine, mapping = aes(x = x1, y = x2), inherit.aes = FALSE) +
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
      
      w <- self$weightSteps
      rowNum <- dim(w)[1]
      
      dfLine <- data.frame(x1 = c(rep(x1Min, rowNum), rep(x1Max, rowNum)),
                           x2 = c(lineFunctionVec(rep(x1Min, rowNum), w = w[,2:3], b = w[,1]),
                                  lineFunctionVec(rep(x1Max, rowNum), w = w[,2:3], b = w[,1])),
                           time = seq(dim(w)[1]))
      
      dfX = data.frame(self$X, self$Y)
      colnames(dfX) <- c("b", "x1", "x2", "y")
      
      p <- ggplot(dfX, aes(x=x1, y=x2)) +
        geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
        geom_line(data = dfLine, mapping = aes(x = x1, y = x2, frame = time), inherit.aes = FALSE) +
        coord_cartesian(ylim=c(x2Min, x2Max)) +
        scale_color_manual(values = c("#DC0026", "#457CB6")) + 
        scale_fill_manual(values = c("#DC0026", "#457CB6"))
      
      gganimate(p)
    }
    
  )
)

set.seed(0)

N <- 10
X <- data.frame(matrix(rnorm(2 * N, mean=0, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=2, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')
X$y = c(rep(-1, N), rep(1, N))

pctr <- Perceptron$new()
weight <- pctr$train(X[,1:2], X[,3])
print(weight)

pctr$plotSteps()
# prc$plotAnimation()

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- pctr$predict(X[i,1:2])
  res <- c(res, pred)
}

result <- data.frame(Y = X[,3], Pred = res)
