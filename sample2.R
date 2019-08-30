
rm(list=ls())
library('keras')
library('tensorflow')

##########

kcol <- 1000
krow <- 10000
khid <- 3
X <- lapply(1:khid,function(i){
  x <- runif(krow,1,2)
  x <- cbind(x,log(x),x^2,sqrt(x),exp(x))
  x
})
X <- do.call(cbind,X)
X <- X + rnorm(length(X))/2
M <- matrix(rnorm(ncol(X)*krow),ncol(X),kcol)
M[abs(M)<0.25] <- 0
Z <- X  %*% M
Z <- Z + rnorm(length(Z))/2
Z <- (Z-min(Z))/(max(Z)-min(Z))

Z.train <- Z[1:9000,]
Z.test <- Z[-1:-9000,]

##########

model <- keras_model_sequential()
model %>%
  layer_dense(units = (ncol(Z)), activation = "tanh", input_shape = c(ncol(Z))) %>%
  layer_dense(units = (floor(ncol(Z)/10)), activation = "tanh", name = "latent") %>%
  layer_dense(units = ncol(Z), activation = "sigmoid", input_shape = (floor(ncol(Z)/10))) %>%
  layer_dense(units = ncol(Z), name='output')
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)
model %>% fit(
  x = Z.train, 
  y = Z.train, 
  epochs = 100
)
encoder <- keras_model(inputs=model$input,outputs=get_layer(model,'latent')$output)

##########

cor(Z.train,model %>% predict(Z.train)) %>% diag %>% summary
cor(Z.test,model %>% predict(Z.test)) %>% diag %>% summary
mse <- function(x,y){
  score1 <- mean((x-y)^2)
  score2 <- mean(apply(y,2,function(x){
    (x-mean(x))^2
  }))
  score1/score2
}
mse(model %>% predict(Z.train),Z.train)
mse(model %>% predict(Z.test),Z.test)

