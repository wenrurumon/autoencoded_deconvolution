
rm(list=ls())
library(keras)

#VAE

vae <- function(X,intermediate_dim,latent_dim,epsilon_std=1,epochs=50,batch_size=100){
  original_dim <- ncol(X)
  x <- layer_input(shape = c(original_dim))
  h <- layer_dense(x, intermediate_dim, activation = "relu")
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
  decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
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
    xent_loss + kl_loss
  }
  vae %>% compile(optimizer = "rmsprop", loss = vae_loss, metrics = c('accuracy') )
  history <- vae %>% fit(
    X,X, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size 
  )
  return(list(score=(encoder%>%predict(X)),
    encoder=encoder,decoder=generator,vae=vae,history=history))
}

#Test

init <- function(X){
  X <- (X-min(X))/(max(X)-min(X))
  X <- array_reshape(X,c(nrow(X),length(X)/nrow(X)))
  X
}
mnist <- dataset_mnist()
X <- init(mnist$train$x)
model <- vae(X,256,10)

X.feature <- model$encoder %>% predict(X)
X.fit <- model$decoder %>% predict(X.feature)
X.fit2 <- model$vae %>% predict(X)

#
