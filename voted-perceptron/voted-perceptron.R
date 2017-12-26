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
      return(sign(sum(vPctr$cSteps * sign(c(1,unlist(X)) %*% t(vPctr$weightSteps)))))
    },
    
    plotSteps = function(){
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      
      for(i in seq(2,dim(self$weightSteps)[1])){
        w <- colSums(vPctr$cSteps[1:i] * vPctr$weightSteps[1:i,]) / sum(vPctr$cSteps[1:i])
        
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
      
      w <- matrix(0, ncol = dim(self$X)[2])
      
      for(i in seq(2,dim(self$weightSteps)[1])){
        w <- rbind(w, colSums(vPctr$cSteps[1:i] * vPctr$weightSteps[1:i,]) / sum(vPctr$cSteps[1:i]))
      }
      
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

N <- 100
X <- data.frame(matrix(rnorm(2 * N, mean=1.8, sd=1), N, 2))
X[seq(N+1,2*N),] <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
colnames(X) <- c('x1', 'x2')
X$y = c(rep(-1, N), rep(1, N))
# ggplot(X, aes(x = x1, y = x2, color = y)) + geom_point()

vPctr <- VotedPerceptron$new()
weight <- vPctr$train(X[,1:2], X[,3], epoch = 3, eta = 1)
print(weight)

vPctr$plotSteps()
# prc$plotAnimation()

res <- c()
for(i in seq(dim(X[,1:2])[1])){
  pred <- vPctr$predict(X[i,1:2])
  res <- c(res, pred)
}

trDf = data.frame(Y = X[,3], Pred = res)
trScore = sum(trDf$Y == trDf$Pred) / dim(trDf)[1] *100
print(paste0('Training score: ', trScore, '%'))
