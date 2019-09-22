
rm(list=ls())
library(keras)

#VAE

vae <- function(X,intermediate_dim=ncol(X),latent_dim=floor(ncol(X)/10),epsilon_std=1,epochs=50,batch_size=100,klpen=0.1){
  original_dim <- ncol(X)
  x <- layer_input(shape = c(original_dim))
  h <- layer_dense(x, intermediate_dim, activation = "sigmoid")
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
  decoder_h <- layer_dense(units = intermediate_dim, activation = "sigmoid")
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
    batch_size = batch_size,
    verbose = 0
  )
  return(list(score=(encoder%>%predict(X)),
              encoder=encoder,decoder=generator,model=vae,history=history))
}

#VAE fit

vae <- function(X,intermediate_dim=ncol(X),latent_dim=floor(ncol(X)/10),epsilon_std=1,epochs=50,batch_size=100,klpen=0.1){
  original_dim <- ncol(X)
  x <- layer_input(shape = c(original_dim))
  h <- layer_dense(x, intermediate_dim, activation = "sigmoid")
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
  decoder_h <- layer_dense(units = intermediate_dim, activation = "sigmoid")
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
    batch_size = batch_size,
    verbose = 0
  )
  return(list(score=(encoder%>%predict(X)),
              encoder=encoder,decoder=generator,model=vae,history=history))
}

#ae2

ae <- function(Z,intermediate_dim=ncol(Z),latent_dim=(floor(ncol(Z)/10)),epochs=50,batch_size=100){
  e.input <- layer_input(shape=ncol(Z))
  e.layer <- layer_dense(e.input,intermediate_dim,activation='sigmoid')
  l.layer <- layer_dense(e.layer,latent_dim,activation='sigmoid')
  d.layer <- layer_dense(units=intermediate_dim,activation='sigmoid')
  d.output <- layer_dense(units=ncol(Z),activation='sigmoid')
  d.input <- layer_input(shape=latent_dim)
  model <- keras_model(e.input,d.output(d.layer(l.layer)))
  encoder <- keras_model(e.input,l.layer)
  decoder <- keras_model(d.input,d.output(d.layer(d.input)))
  model %>% compile(
    loss = "mean_squared_error", 
    optimizer = "adam",
    metrics = c('accuracy')
  )
  history <- model %>% fit(
    x = Z, 
    y = Z,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 0
  )
  return(list(score=(encoder%>%predict(Z)),
              encoder=encoder,decoder=decoder,model=model,history=history))
}

# QPCA

qpca <- function(A,rank=0,ifscale=TRUE){
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
pca <- function(X,prop=1,ifscale=TRUE){
  if(ifscale){X <- scale(X)[,]}
  m = nrow(X)
  n = ncol(X)
  Xeigen <- svd(X)
  value <- (Xeigen$d)^2/m
  value <- cumsum(value/sum(value))
  score <- X %*% Xeigen$v
  list(score=score,value=value,mat=Xeigen$v)
}
vnorm <- function(x){
  apply(x,2,function(x){
    (x-min(x))/(max(x)-min(x))
  })
}

mse <- function(x,y){mean((x-y)^2)/mean((y-mean(y))^2)}

#

library(e1071)
library(parallel)
library(preprocessCore)

CoreAlg <- function(X, y){
  
  res <- function(i){
    nus <- 0.25 * i
    # if(i==1){nus <- 0.25}
    # if(i==2){nus <- 0.5}
    # if(i==3){nus <- 0.75}
    model<-svm(X,y,type="nu-regression",kernel="linear",nu=nus,scale=F)
    model
  }
  
  svn_itor <- 3
  out <- lapply(1:svn_itor,res) # if mclapply doesnt work
  
  nusvm <- rep(0,svn_itor)
  corrv <- rep(0,svn_itor)
  
  #do cibersort
  t <- 1
  while(t <= svn_itor) {
    weights = t(out[[t]]$coefs) %*% out[[t]]$SV
    weights[which(weights<0)]<-0
    w<-weights/sum(weights)
    u <- sweep(X,MARGIN=2,w,'*')
    k <- apply(u, 1, sum)
    nusvm[t] <- sqrt((mean((k - y)^2)))
    corrv[t] <- cor(k, y)
    t <- t + 1
  }
  
  #pick best model
  rmses <- nusvm
  mn <- which.min(rmses)
  if(length(mn)>0){
    model <- out[[mn]]
    q <- t(model$coefs) %*% model$SV
    q[which(q<0)]<-0
    w <- (q/sum(q))
    mix_rmse <- rmses[mn]
    mix_r <- corrv[mn]
  } else {
    print('bug')
    w <- rep(1,ncol(X))/ncol(X)
    # w <- lmpos(y,X)
  }
  
  # newList <- list("w" = w, "mix_rmse" = mix_rmse, "mix_r" = mix_r)
  newList <- list("w" = w)
  newList
  
}

deconv <- function(x,y,QN=TRUE,ifprint=F){
  #y is the matrix of sample and result, x is the matrix of x and action
  #Input Setup
  rawX <- X <- data.matrix(x)
  rawY <- Y <- data.matrix(y)
  #Process Y Matrix
  Y <- scale(Y)
  if(max(Y) < 50) {Y <- 2^Y}
  dnmY <- dimnames(Y)
  if(QN == TRUE){Y <- normalize.quantiles(Y);dimnames(Y) <- dnmY}
  #Process X matrix
  X <- scale(X)
  #Run SVM
  eP <- sapply(1:ncol(Y),function(i){
    if(ifprint){print(i)}
    CoreAlg(X,Y[,i])$w})
  dimnames(eP) <- list(colnames(x),colnames(y))
  
  
  eY <- sapply(1:ncol(Y),function(i){
    x %*% (cbind(sum(y[,i]) * eP[,i])/colSums(x))
  })
  # eY[is.na(eY)] <- 0
  # eY <- rawX %*% eP
  # rsquare <- sapply(1:ncol(Y),function(i){summary(lm(rawY[,i]~eY[,i]-1))$r.square})
  # names(rsquare) <- colnames(Y)
  return(
    list(coef = rbind(eP),fit=eY
         # ,rsquare=rsquare
         )
  )
}

lmpos <- function(y,x){
  y <- scale(y)
  x <- scale(x)
  sel <- 1:ncol(x)
  
  model <- lm(y~x[,sel,drop=F]-1)
  summary(model)
  model.tstat <- coef(summary(model))[,3]
  model.kick <- sel[which.min(model.tstat)]
  while(length(model.tstat)>1 & model.tstat[which(sel==model.kick)]<0){
    sel <- sel[!sel%in%model.kick]
    model <- lm(y~x[,sel,drop=F]-1)
    model.tstat <- coef(summary(model))[,3]
    model.kick <- sel[which.min(model.tstat)]
  }

  out <- rep(0,ncol(x))
  out[sel] <- 1
  out
}

minmax <- function(x){
  (x-min(x))/(max(x)-min(x))
}


setwd("/Users/wenrurumon/Documents/fun/tensorflow")

##############################################

fun3 <- function(i){
  #set seed
  gc()
  set.seed(i)
  print(paste(i,Sys.time()))
  #Generate the 300 sample * 5 cell type contribution matrix
  contri <- sapply(1:5,function(i){
    x <- rnorm(300)
    x
  })
  contri[contri<0] <- 0
  for(j in which(rowSums(contri)==0,)){
    contri[j,sample(5)[1:2]] <- runif(2,0,100)
  }
  contri <- contri / rowSums(contri)
  #Generate the 5 cell type * 100 gene reference matrix
  rawref <- sapply(1:5,function(i){
    x <- rnorm(100)
    x-min(x)+1
  })
  ref <- apply(rawref,2,function(x){
    x1 <- sin(x*100)
    x2 <- tan(x*100)
    x3 <- log(x)
    x4 <- exp(x)
    x <- apply(cbind(x1,x2,x3,x4),2,function(x){(x-min(x))/(max(x)-min(x))})
    x <- x + runif(length(x),0,0.2)
    x <- apply(x,2,function(x){(x-min(x))/(max(x)-min(x))})
    rowSums(x)
  })
  #Contribution * reference: 300*100 bluk
  bulk <- contri %*% t(ref)
  bulk <- apply(bulk,2,function(x){(x-min(x))/(max(x)-min(x))})
  #deconvolution raw
  deconv.raw <- deconv(rawref,t(bulk))
  deconv.ref <- deconv(ref,t(bulk))
  #deconvolution autoencoder
  system.time(model.ae <- ae(bulk,latent_dim=20,epochs=2000))
  bulk.ae <- model.ae$encoder %>% predict(bulk)
  ref.ae <- model.ae$encoder %>% predict(t(rawref))
  deconv.ae <- deconv(t(ref.ae),t(bulk.ae))
  fit.ae <- model.ae$decoder %>% predict(t(deconv.ae$fit))
  #deconvolution pca
  system.time(model.pca <- pca(bulk))
  bulk.pca <- minmax(bulk %*% model.pca$mat[,1:20])
  ref.pca <- minmax(t(rawref) %*% model.pca$mat[,1:20])
  deconv.pca <- deconv(t(ref.pca),t(bulk.pca))
  fit.pca <- cbind(1,t(deconv.pca$fit)) %*% coef(lm(bulk~model.pca$score[,1:20]))
  #Result
  rlt <- c(rawfit=mse(deconv.raw$fit,t(bulk)),
    reffit=mse(deconv.ref$fit,t(bulk)),
    aefit=mse(fit.ae,bulk),
    pcafit=mse(fit.pca,(bulk)),
    rawcoef=mse(deconv.raw$coef,t(contri)),
    refcoef=mse(deconv.ref$coef,t(contri)),
    aecoef=mse(deconv.ae$coef,t(contri)),
    pcacoef=mse(deconv.pca$coef,t(contri))
  )
  print(rlt[1:4])
  print(rlt[5:8])
  return(rlt)
}

# system.time(test <- t(sapply(401:500,fun3)))
# save(test,file='simulation_401_500.rda')

##############################################

rm(list=ls())
load("/Users/wenrurumon/Documents/fun/tensorflow/simulation_fun3.rda")
test <- apply(rlt,2,function(x){rep(x,4)})
par(mfrow=c(2,2))
for(i in 1:8){
if(i%%4!=0){
  hist(test[,i],main=colnames(test)[i],breaks=c(0:26)/20)
} else {
  hist(test[,i],main=colnames(test)[i],breaks=20)
}
}


