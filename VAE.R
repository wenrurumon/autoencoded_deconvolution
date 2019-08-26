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

# Dummy data

kcol <- 100
krow <- 1000
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

# Test with QPCA

rank <- which((Z.qpca <- qpca(Z))$prop>0.99)[1]
Z.vae <- vae(Z,intermediate_dim=ncol(Z),latent_dim=rank)

