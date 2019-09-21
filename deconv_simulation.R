
#########
#Dummy data
# 1000 samples * 2000 gene
# 1000 samples * 200 cell

setwd("/Users/wenrurumon/Documents/fun/tensorflow")

fun <- function(i){
  print(i)
  set.seed(i)
  ref <- lapply(1:5,function(i){
    x <- rnorm(300)
    (x-min(x))/(max(x)-min(x))
  })
  ref2 <- lapply(ref,function(x){
    x1 <- sin(x*100)
    x2 <- tan(x*100)
    x3 <- x ^ 2
    x4 <- x ^ (1/3); x4 <- ifelse(is.na(x4),0,x4)
    x <- apply(cbind(x1,x2,x3,x4),2,function(x){(x-min(x))/(max(x)-min(x))})
    x <- x + runif(length(x),0,0.2)
    x <- apply(x,2,function(x){(x-min(x))/(max(x)-min(x))})
    x
  })
  raw.x <- do.call(cbind,ref2)
  colnames(raw.x) <- paste0('x',1:ncol(raw.x))
  raw.y <- matrix(rnorm(20*100),20,100)
  raw.y[raw.y<0] <- 0
  raw.y <- t(t(raw.y)/colSums(raw.y))
  raw <- raw.x %*% raw.y
  raw <- raw * runif(length(raw),0.9,1.1)
  #with actual reference
  deconv.raw <- deconv(raw.x,raw)
  #with basic reference
  deconv.ref <- deconv(do.call(cbind,ref),raw)
    # mse(deconv.ref$fit,raw)
  #Autoencoder
  system.time(model.ae <- ae(raw,latent_dim=5,epochs=1000))
    # plot(model.ae$history)
  x.ae <- model.ae$encoder %>% predict(raw)
    # mse(model.ae$decoder %>% predict(x.ae),raw)
  deconv.ae <- deconv(do.call(cbind,ref),x.ae)
    # mse(model.ae$decoder %>% predict(deconv.ae$fit), raw)
  #Autoencoder2
    # system.time(model.ae <- ae(raw,latent_dim=5,epochs=3000))
    # x.ae <- model.ae$encoder %>% predict(raw)
    # deconv.ae2 <- deconv(do.call(cbind,ref),x.ae)
    # # mse(model.ae$decoder %>% predict(deconv.ae2$fit), raw)
  #QPCA
  model.pca <- pca(raw,ifscale=T,rank=20)
  pca.x<- (model.pca$X-min(model.pca$X))/(max(model.pca$X)-min(model.pca$X))
  pca.coef <- coef(lm(raw~pca.x))
  deconv.pca <- deconv(do.call(cbind,ref),pca.x)
  rlt.pca <- cbind(1,deconv.pca$fit) %*% pca.coef
    # mse(rlt.pca,raw)
  #Result
  c(seed=i,sumraw=sum(raw),
    raw=mse(deconv.ref$fit,raw),
    ae=mse(model.ae$decoder %>% predict(deconv.ae$fit), raw),
    # ae2=mse(model.ae$decoder %>% predict(deconv.ae2$fit), raw),
    qpca=mse(rlt.pca,raw))
}

system.time(test <- t(sapply(1500,fun)))
mse.raw_ref <- rep(test[,3],5);hist(mse.raw_ref)
mse.ae_ref <- rep(test[,4],5);hist(mse.ae_ref)
mse.pca_ref <- rep(test[,5],5);hist(mse.pca_ref)
