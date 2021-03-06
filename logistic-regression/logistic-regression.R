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

logisticFunction <- function(x){
  return(1 / (1 + exp(-x)))
}

LogisticRegression <- R6Class("LogisticRegression",
  public = list(
    X = NULL,
    Y = NULL,
    weightSteps = NULL,
    w = NULL,
    
    train = function(X, Y, maxit = 10, eta = 0.01){
      X <- data.matrix(data.frame(b = 1,X))
      self$weightSteps <- matrix(0, ncol = dim(X)[2])
      self$X <- X
      self$Y <- Y
      w <- rep(0, dim(X)[2])
      
      for(i in seq(maxit)){
        w <- w + eta * (Y - w %*% t(X)) %*% X
        self$weightSteps <- rbind(self$weightSteps, w)
      }
      
      self$w <- w
      return(w)
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
        dfLine[is.na(dfLine)] <- 0
        
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
      dfLine[is.na(dfLine)] <- 0
      
      dfX = data.frame(self$X, self$Y)
      colnames(dfX) <- c("b", "x1", "x2", "y")
      
      p <- ggplot(dfX, aes(x=x1, y=x2)) +
        geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
        geom_line(data = dfLine, mapping = aes(x = x1, y = x2, frame = time), inherit.aes = FALSE) +
        coord_cartesian(ylim=c(x2Min, x2Max)) +
        scale_color_manual(values = c("#DC0026", "#457CB6")) + 
        scale_fill_manual(values = c("#DC0026", "#457CB6"))
      
      gganimate(p)
    },
    
    plotLogisticFunctionStep = function(){
      xMin <- min(self$X[,2])
      xMax <- max(self$X[,2])
      xDiff <- xMax - xMin
      rangeX = seq(xMin-xDiff*0.1, xMax+xDiff*0.1, by = xDiff / 200)
      logisticDf <- data.frame(x = rangeX)
      
      for(i in seq(dim(self$weightSteps)[1])){
        w <- self$weightSteps[i,]

        probDf <- data.frame(x = self$X[,2], y = self$Y)
        probDf$prob <- apply(self$X, 1, self$predictProbability, w)
        probDf$prob[probDf$prob > 0.5] <- 1
        probDf$prob[probDf$prob <= 0.5] <- 0
        
        p <- ggplot(logisticDf, aes(x)) +
          stat_function(fun = logisticFunction) +
          geom_point(data = probDf, aes(x=x, y=prob, color=factor(y), shape=factor(y)), inherit.aes = FALSE)+
          theme(legend.position="none") +
          scale_color_manual(values = c("#DC0026", "#457CB6")) + 
          scale_fill_manual(values = c("#DC0026", "#457CB6"))
        
        print(p)
      }
    },
    
    plotLogisticFunctionAnimation = function(){
      xMin <- min(self$X[,2])
      xMax <- max(self$X[,2])
      xDiff <- xMax - xMin
      rangeX = seq(xMin-xDiff*0.1, xMax+xDiff*0.1, by = xDiff / 200)
      logisticDf <- data.frame(x = rangeX)
      
      probDf <- data.frame(matrix(0, ncol = 4))[-1]
      
      for(i in seq(dim(self$weightSteps)[1])){
        w <- self$weightSteps[i,]
        
        probDfTemp <- data.frame(x = self$X[,2], y = self$Y, time = i)
        probDfTemp$prob <- apply(self$X, 1, self$predictProbability, w)
        probDfTemp$prob[probDfTemp$prob > 0.5] <- 1
        probDfTemp$prob[probDfTemp$prob <= 0.5] <- 0
        
        probDf <- rbind(probDf, probDfTemp)
      }
      
      p <- ggplot(logisticDf, aes(x)) +
        stat_function(fun = logisticFunction) +
        geom_point(data = probDf, aes(x=x, y=prob, color=factor(y), shape=factor(y), frame=time), inherit.aes = FALSE)+
        theme(legend.position="none") +
        scale_color_manual(values = c("#DC0026", "#457CB6")) + 
        scale_fill_manual(values = c("#DC0026", "#457CB6"))
      
      gganimate(p)
    },
    
    predictProbability = function(x, w){
      return(1 / (1 + exp(-sum(w * x))))
    },
    
    predictClass = function(x, w){
      return(sign(sum(w * unlist(c(1,x)))))
    }
  )
)

set.seed(0)

N <- 100
X <- data.frame(matrix(rnorm(2 * N, mean=1.8, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')
X$y = c(rep(1, N), rep(-1, N))

logReg <- LogisticRegression$new()
weight <- logReg$train(X[,1:2], X[,3], maxit = 10, eta = 0.001)
print(weight)

# logReg$plotSteps()
logReg$plotLogisticFunction()

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- logReg$predictClass(X[i,1:2], weight)
  res <- c(res, pred)
}

trDf = data.frame(Y = X[,3], Pred = res)
trScore = sum(trDf$Y == trDf$Pred) / dim(trDf)[1] *100
print(paste0('Training score: ', trScore, '%'))
