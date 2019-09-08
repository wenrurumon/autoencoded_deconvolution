rm(list=ls())
library(keras)

#VAE

vae <- function(X,intermediate_dim=ncol(X),latent_dim=floor(ncol(X)/10),epsilon_std=1,epochs=50,batch_size=100,klpen=0.1){
  original_dim <- ncol(X)
  x <- layer_input(shape = c(original_dim))
  h <- layer_dense(x, intermediate_dim, activation = "tanh")
  z_mean <- layer_dense(h, latent_dim)
  z_log_var <- layer_dense(h, latent_dim)
  sampling <- function(arg){
    z_mean <- arg[, 1:(latent_dim)]
    z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
    epsilon <- k_random_normal(
      shape = c(k_shape(z_mean)[[1]]), 
      mean=0.,
      stddev=epsilon_std
    )
    z_mean + k_exp(z_log_var/2)*epsilon
  }
  z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
    layer_lambda(sampling)
  decoder_h <- layer_dense(units = intermediate_dim, activation = "tanh")
  decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
  h_decoded <- decoder_h(z)
  x_decoded_mean <- decoder_mean(h_decoded)
  vae <- keras_model(x, x_decoded_mean)
  encoder <- keras_model(x, z_mean)
  decoder_input <- layer_input(shape = latent_dim)
  h_decoded_2 <- decoder_h(decoder_input)
  x_decoded_mean_2 <- decoder_mean(h_decoded_2)
  generator <- keras_model(decoder_input, x_decoded_mean_2)
  vae_loss <- function(x, x_decoded_mean){
    xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    xent_loss + 0.1*kl_loss
  }
  vae %>% compile(optimizer = "rmsprop", loss = vae_loss, metrics = c('accuracy') )
  history <- vae %>% fit(
    X,X, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size 
  )
  return(list(score=(encoder%>%predict(X)),
              encoder=encoder,model=vae,history=history))
}

#AE

ae <- function(Z,intermediate_dim=ncol(Z),latent_dim=(floor(ncol(Z)/10)),epochs=50,batch_size=100){
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = intermediate_dim, activation = "tanh", input_shape = c(ncol(Z))) %>%
    layer_dense(units = latent_dim, activation = "tanh", name = "latent") %>%
    layer_dense(units = intermediate_dim, activation = "tanh", input_shape = (floor(ncol(Z)/10))) %>%
    layer_dense(units = intermediate_dim, activation = "sigmoid", name='output')
  model %>% compile(
    loss = "mean_squared_error", 
    optimizer = "adam",
    metrics = c('accuracy')
  )
  history <- model %>% fit(
    x = Z, 
    y = Z,
    epochs = epochs,
    batch_size = batch_size
  )
  encoder <- keras_model(inputs=model$input,
                         outputs=get_layer(model,'latent')$output)
  return(list(score=(encoder%>%predict(Z)),
              encoder=encoder,model=model,history=history))
}

# QPCA

pca <- function(A,rank=0,ifscale=TRUE){
  if(ifscale){A <- scale(as.matrix(A))[,]}
  A.svd <- svd(A)
  if(rank==0|rank==ncol(A)){
    d <- A.svd$d
  } else {
    d <- A.svd$d-A.svd$d[min(rank+1,nrow(A),ncol(A))]
    d <- d[d > 1e-8]
  }
  r <- length(d)
  prop <- d^2; info <- sum(prop)/sum(A.svd$d^2);prop <- cumsum(prop/sum(prop))
  d <- diag(d,length(d),length(d))
  u <- A.svd$u[,1:r,drop=F]
  v <- A.svd$v[,1:r,drop=F]
  x <- u%*%sqrt(d)
  y <- sqrt(d)%*%t(v)
  z <- x %*% y
  rlt <- list(rank=r,X=x,Y=y,Z=x%*%y,prop=prop,info=info)
  return(rlt)

  }
qpca <- function(A,prop=0.99,ifscale=TRUE){
  x <- pca(A,rank=0,ifscale=TRUE)
  x <- pca(A,rank=which(x$prop>prop)[1],ifscale=TRUE)
  x
}
vnorm <- function(x){
  apply(x,2,function(x){
    (x-min(x))/(max(x)-min(x))
  })
}

# Test data 
mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")

system.time(model.vae1 <- vae(x_train,epochs=100))
system.time(model.vae2 <- vae(x_train,epochs=100,klpen=0.2))
system.time(model.vae3 <- vae(x_train,epochs=100,klpen=0.3))
system.time(model.ae <- ae(x_train,epochs=100))
system.time(model.pca <- pca(x_test,rank=100,ifscale=F))

fit.vae1 <- model.vae1$model %>% predict(x_test)
fit.vae2 <- model.vae2$model %>% predict(x_test)
fit.vae3 <- model.vae3$model %>% predict(x_test)
fit.ae <- model.ae$model %>% predict(x_test)
mse <- function(x,y){mean((x-y)^2)}

mse(fit.vae1,x_test)
mse(fit.vae2,x_test)
mse(fit.vae3,x_test)
mse(fit.ae,x_test)
mse(model.pca$Z,x_test)


