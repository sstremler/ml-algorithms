rm(list=ls())
library(ggplot2)
library(animation)
library(gganimate)
library(R6)
library(dummies)

NNClassifier <- R6Class("NNClassifier",
  public = list(
    X = NULL,
    y = NULL,
    yHot = NULL,
    eta = NULL,
    layers = NULL,
    w = NULL,
    wSteps = NULL,
    errorSteps = NULL,
    scoreSteps = NULL,
    epoch = NULL,
    
    train = function(X, y, layers, epoch = 10, eta = 0.5){
      xB <- cbind(rep(1, dim(X)[1]), X)
      self$yHot <- as.matrix(dummy.data.frame(data.frame(y=y), names=c("y"), sep=""))
      self$y <- y
      # the number of neurons in the output layer is equal to the unique class labels in y
      layers <- c(dim(X)[2], layers, dim(self$yHot)[2]) 
      w <- private$initializeWeights(layers)
      self$eta <- eta
      self$X <- xB
      self$layers <- layers
      self$wSteps = list()
      self$errorSteps <- c()
      self$scoreSteps <- c()
      self$epoch <- epoch
      
      wTemp <- list()
      
      for(i in 1:epoch){ # iterate over epochs
        set.seed(i)
        
        for(k in sample(dim(xB)[1])){ # iterate over points randomly
          # feedforward
          
          layerSum <- list()
          layerOutput <- list()
          layerOutput[[1]] <- xB[k,]
          
          for(j in seq(length(layers) - 2)){ # only for hidden layers
            layerHSum <- t(t(layerOutput[[j]]) %*% w[[j]])
            layerSum[[j]] <- layerHSum
            layerHOutput <- private$sigmoid(layerHSum)
            layerOutput[[j + 1]] <- matrix(c(1,layerHOutput)) 
          }
          
          # output layer
          lastLayerOutput <- layerOutput[[length(layerOutput)]] # (n-1)st layer
          layerOSum <- t(t(lastLayerOutput) %*% w[[length(w)]]) # output layer sum
          layerOOutput <- private$softmax(layerOSum) # output of the output layer
          # end output layer
          
          # calculate error
          layerOOutputDifference <- self$yHot[k,] - layerOOutput
          layerOOutputErrorTotal <- -1*sum(self$yHot[k,]*log(layerOOutput)) # cross entropy
          # cat("Individual error: ",-layerOOutputDifference, "  CE: ", layerOOutputErrorTotal, "\n")
          # end calculate error
          # end feedforward
          
          # backpropagation
          # output layer
          lastLayerOutputWoB <- matrix(lastLayerOutput[-1]) # output of the (n-1)st layer without bias
          deltaOutput <- (-layerOOutputDifference) # local gradient of the output layer
          gradWOutput <- deltaOutput %*% t(lastLayerOutputWoB)
          gradWOutput <- rbind(t(deltaOutput), t(gradWOutput))
          
          wTemp[[length(w)]] <- w[[length(w)]] - eta*gradWOutput
          # end output layer
          
          # hidden layer
          for(j in (length(layers)-2):1){
            deltaH <- private$sigmoidDer(layerSum[[j]]) * t(t(deltaOutput) %*% t(w[[j + 1]][-1,]))
            gradWH <- deltaH %*% t(matrix(layerOutput[[j]][-1]))
            gradWH <- rbind(t(deltaH), t(gradWH))
            wTemp[[j]] <- w[[j]] - eta*gradWH
            deltaOutput <- deltaH
          }
          # hl end
          # end backpropagation
          
          w <- wTemp
        }
        
        self$wSteps[[i]] <- w
        pred <- self$predictProba(matrix(self$X[,-1], ncol = 2), w)
        self$errorSteps <- c(self$errorSteps, -1*sum(self$yHot*log(pred)))
        
        df = data.frame(y = self$y, pred = matrix(apply(pred, 1, which.max)))
        self$scoreSteps <- c(self$scoreSteps, sum(df$y == df$pred) / dim(df)[1] * 100)
      }
      
      self$w <- w
      
      return(w)
    },
    
    plotSteps = function(step){ # the function plot every stepth epoch
      x1Min <- min(self$X[,2])
      x1Max <- max(self$X[,2])
      x2Min <- min(self$X[,3])
      x2Max <- max(self$X[,3])
      rangeX = seq(x1Min, x1Max, by = (x1Max - x1Min) / 200)
      rangeY = seq(x2Min, x2Max, by = (x2Max - x2Min) / 200)
      
      for(i in seq(1, length(nn$wSteps), step)){
        contour <- data.matrix(expand.grid(x = rangeX, y = rangeY))
        contour <- cbind(contour,self$predict(contour[,1:2], self$wSteps[[i]]))
  
        dfX = data.frame(self$X, self$y)
        colnames(dfX) <- c("b", "x1", "x2", "y")
        contour <- data.frame(x = contour[,1], y = contour[,2], z = contour[,3])
        
        p <- ggplot(dfX, aes(x = x1, y = x2)) +
          geom_point(aes(color = factor(y), shape = factor(y)), size = 2) + theme(legend.position="none") +
          geom_raster(data = contour, aes(x = x, y = y, fill = factor(z)), alpha = 1/5) +
          coord_cartesian(ylim=c(x2Min, x2Max)) +
          scale_color_manual(values = c("#DC0026", "#457CB6", "#FF9900")) + 
          scale_fill_manual(values = c("#DC0026", "#457CB6", "#FF9900"))
        
        print(p)
      }
    },
    
    plotError = function(){
      errDf <- data.frame(sse = nn$errorSteps, epoch = 1:self$epoch)
      print(ggplot(errDf, aes(x = epoch, y = sse)) + geom_line())
    },
    
    plotScore = function(){
      scoreDf <- data.frame(score = nn$scoreSteps, epoch = 1:self$epoch)
      print(ggplot(scoreDf, aes(x = epoch, y = score)) + geom_line())
    },
    
    predictProba = function(X, w = self$w){
      xB <- cbind(rep(1, dim(X)[1]), X)
      hiddenLayerNo <- length(self$layers) - 2
      for(i in 1:hiddenLayerNo){
        xB <- cbind(1, private$sigmoid(xB %*% w[[i]]))
      }
      
      return(t(apply(xB %*% w[[hiddenLayerNo + 1]], 1, private$softmax)))
    },
    
    predict = function(X, w = self$w){
      return(matrix(apply(self$predictProba(X, w), 1, which.max)))
    },
    
    score = function(X, y){
      df = data.frame(y = y, pred = self$predict(X))
      score = sum(df$y == df$pred) / dim(df)[1] * 100
      
      return(score)
    },
    
    crossEntropyError = function(X, y){
      pred <- self$predictProba(X)
      yHot <- as.matrix(dummy.data.frame(data.frame(y=y), names=c("y"), sep=""))
      return(-1*sum(yHot * log(pred)))
    }
  ),
  
  private = list(
    initializeWeights = function(layers){
      w <- list()
      
      for(i in seq(length(layers) - 1)){
        wNo <- (layers[i] + 1) * layers[i + 1] # number of weights between the i-th and i+1-th layer
        nNo <- (layers[i] + 1) + layers[i + 1] # number of neurons in the i-th and i+1-th layer
        wTemp <- matrix(runif(wNo, min = -sqrt(6/nNo), max = sqrt(6/nNo)), ncol = layers[i + 1]) # Glorot initialization of weights
        w[[i]] <- wTemp
      }
      
      return(w)
    },
    
    sigmoid = function(x){
      return(1/(1+exp(-x)))
    },
    
    sigmoidDer = function(x){
      return(private$sigmoid(x)*(1-private$sigmoid(x)))
    },
    
    softmax = function(x){
      summa <- sum(exp(x))
      
      return(exp(x) / summa)
    }
  )
)

#############
###### Test 1
#############
# set.seed(0)
# N <- 100
# X <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
# X <- rbind(X, matrix(rnorm(2 * N, mean=0, sd=1), N, 2))
# 
# # generate classes
# # let the class be -1 under the mirror of the hyperbolic cosine function
# # and +1 above
# y <- X[,2] <= -cosh(X[,1]) + 1.5
# y[y == TRUE] <- 1
# y[y == FALSE] <- 2
# yHot <- as.matrix(dummy.data.frame(data.frame(y=y), names=c("y"), sep=""))
# 
# X[,2][y == 2] = X[,2][y == 2] + 0.5
# 
# # df <- data.frame(x1 = X[,1], x2 = X[,2], y = y)
# # ggplot(df, aes(x = x1, y = x2, color = y)) + geom_point()
# 
# layers <- c(25)
# 
# nn <- NNClassifier$new()
# weights <- nn$train(X, yHot, layers, epoch = 1001, eta = 0.005)
# # print(weights)
# 
# print(paste("Training score: ", nn$score(X,y), "%", sep = ""))
# print(paste("Cross Entropy Error: ", nn$crossEntropyError(X,y), sep = ""))
# 
# nn$plotSteps(100)
# nn$plotError()

#############
###### Test 1 end
#############

#############
###### Test 2
#############
set.seed(0)
N <- 100
X <- matrix(rnorm(2 * N, mean=0, sd=1), N, 2)
X <- rbind(X, matrix(rnorm(2 * N, mean=0, sd=1), N, 2))

# generate classes
# let the class be -1 under the mirror of the hyperbolic cosine function
# and +1 above
y <- X[,2] <= -cosh(X[,1]) + 1.5
y[y == TRUE] <- 1
y[y == FALSE] <- 2

# let class 1 x>0 be class 3
y3 <- X[y == 2,1] > 0
y3[y3 == TRUE] <- 3
y3[y3 == FALSE] <- 2
y[y == 2] <- y3

X[,2][y == 2] = X[,2][y == 2] + 0.25
X[,1][y == 3] = X[,1][y == 3] + 0.25

# df <- data.frame(x1 = X[,1], x2 = X[,2], y = y)
# ggplot(df, aes(x = x1, y = x2, color = y)) + geom_point()

layers <- c(25)

nn <- NNClassifier$new()
weights <- nn$train(X, y, layers, epoch = 1001, eta = 0.005)
# print(weights)

print(paste("Training score: ", nn$score(X,y), "%", sep = ""))
print(paste("Cross Entropy Error: ", nn$crossEntropyError(X,y), sep = ""))

nn$plotSteps(100)
nn$plotError()
nn$plotScore()
#############
###### Test 2 end
#############


