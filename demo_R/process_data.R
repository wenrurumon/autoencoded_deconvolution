
rm(list=ls())
library(keras)
library(dplyr)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/4nan')
load("/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/mdata4causal.rda")
load('4nan.rda')
load('/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/ref2.rda')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/ae/')

############################################################
############################################################

gene <- rownames(ref)[rownames(ref)%in%colnames(raw_exp)]
bulk.raw<- raw_exp[,match(gene,colnames(raw_exp))]
ref.raw <- t(ref[match(gene,rownames(ref)),])
rm(list=ls()[!ls()%in%c('bulk.raw','ref.raw')])
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
adgene <- do.call(rbind,strsplit(readLines('AD_gene.txt'),'\t|;'))[,2]
load_keras_models <- function(x){
  x <- paste(x,c('decoder','encoder','model','rlt'),sep='.')
  models <- lapply(x[1:2],load_model_hdf5)
  log <- readLines(x[4])[-8:-9]
  list(encoder=models[[2]],decoder=models[[1]],log=log)
}

############################################################
############################################################

#Gene filter
# sel <- which(colnames(bulk.raw) %in% adgene)
  sel <- 1:ncol(ref.raw)
  x <- apply(ref.raw[,sel],2,minmax)
  y <- apply(bulk.raw[,sel],2,minmax)
#Deconvolution with overexpressed gene
  x.sel <- apply(x,1,function(xi){
    names(which(xi>quantile(xi,0.99)))
  }) %>% unlist %>% unique
  y.sel <- y[,colnames(y)%in%x.sel]
  x.sel <- x[,colnames(x)%in%x.sel]
  dim(y.sel)
  system.time(deconv.sel <- deconv(t(x.sel),t(y.sel),ifprint=T))
  fit.sel <- dedeconv((x),deconv.sel$coef,(y))
  mse.sel <- mse(fit.sel,y)
  rlt.sel <- list(fit=fit.sel,mse=mse.sel)
#Deconvolution with autoencoder
  setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt')
  models <- unique(do.call(rbind,strsplit(dir(),'\\.'))[,1])
  fit_keras <- function(m){
    print(m)
    models <- load_keras_models(m)
    x.ae <- models$encoder %>% predict(x)
    y.ae <- models$encoder %>% predict(y)
    system.time(deconv.ae <- deconv(t(x.ae),t(y.ae),ifprint=F))
    fit.ae <- models$decoder %>% predict(t(deconv.ae$fit))
    mse.ae <- mse(fit.ae,y)
    list(model=m,fit=fit.ae,mse=mse.ae)
  }
  system.time(m1 <- fit_keras(models[1])); m1$mse
  system.time(m2 <- fit_keras(models[2])); m2$mse
  system.time(m3 <- fit_keras(models[3])); m3$mse
  system.time(m4 <- fit_keras(models[4])); m4$mse



############################################################
############################################################

#Gene filter
sel <- which(colnames(bulk.raw) %in% adgene)
x <- apply(ref.raw[,sel],2,minmax)
y <- apply(bulk.raw[,sel],2,minmax)
#Deconvolution with rawdata
system.time(deconv.raw <- deconv(t(x),t(y)))
fit.raw <- t(deconv.raw$fit)
mse(fit.raw,y)
#Deconvolution with autoencode
system.time(model.ae <- ae(y,latent_dim=20,epochs=2000))
plot(model.ae$history)
x.ae <- model.ae$encoder %>% predict(x)
y.ae <- model.ae$encoder %>% predict(y)
system.time(deconv.ae <- deconv(t(x.ae),t(y.ae)))
fit.ae <- model.ae$decoder %>% predict(t(deconv.ae$fit))
mse(fit.ae,y)
#Deconvolution with vae
system.time(model.vae <- vae(y,latent_dim=20,epochs=2000))
plot(model.vae$history)
x.vae <- model.vae$encoder %>% predict(x) %>% minmax()
y.vae <- model.vae$encoder %>% predict(y)
y.range <- range(y.vae)
y.vae <- (y.vae-y.range[1])/(y.range[2]-y.range[1])
system.time(deconv.vae <- deconv(t(x.vae),t(y.vae)))
fit.vae <- t(deconv.vae$fit * (y.range[2]-y.range[1]) + y.range[1])
fit.vae <- model.vae$decoder %>% predict(fit.vae)
mse(fit.vae,y)
#Deconvolution with PCA
system.time(model.pca <- qpca(y))
x.pca <- (x %*% solve(model.pca$Y)[,1:20])
y.pca <- (y %*% solve(model.pca$Y)[,1:20])
system.time(deconv.pca <- deconv(t(x.pca),t(y.pca)))
fit.pca <- cbind(1,t(deconv.pca$fit)) %*% coef(lm(y~y.pca))
mse(fit.pca,y)
#Deconvolution with other methods
x.sel <- apply(x,1,function(xi){
  # names(which(xi>=quantile(xi,0.99)))
  names(xi)[which(scale(xi)>1.96)]
}) %>% unlist %>% unique
y.sel <- y[,colnames(y)%in%x.sel]
x.sel <- x[,colnames(x)%in%x.sel]
system.time(deconv.sel <- deconv(t(x.sel),t(y.sel)))
fit.sel <- dedeconv((x),deconv.sel$coef,(y))
mse(fit.sel,y)
#Result
c(mse.raw=mse(fit.raw,y),
  mse.pca=mse(fit.pca,y),
  mse.ae=mse(fit.ae,y),
  mse.vae=mse(fit.vae,y),
  mse.sel=mse(fit.sel,y))

############################################################
############################################################

fun4 <- function(i){
  set.seed(i);sel <- sample(1:ncol(ref.raw),400)
  print(paste(i,sum(sel),Sys.time()))
  x <- apply(ref.raw[,sel],2,minmax)
  y <- apply(bulk.raw[,sel],2,minmax)
  st <- Sys.time()
  system.time(deconv.raw <- deconv(t(x),t(y)))
  fit.raw <- t(deconv.raw$fit)
  end.raw <- Sys.time()-st
  st <- Sys.time()
  system.time(model.ae <- ae(y,latent_dim=100,epochs=1000,verbose=0))
  x.ae <- model.ae$encoder %>% predict(x)
  y.ae <- model.ae$encoder %>% predict(y)
  system.time(deconv.ae <- deconv(t(x.ae),t(y.ae)))
  fit.ae <- model.ae$decoder %>% predict(t(deconv.ae$fit))
  end.ae <- Sys.time()-st
  st <- Sys.time()
  system.time(model.vae <- vae(y,latent_dim=20,epochs=1000))
  x.vae <- model.vae$encoder %>% predict(x) %>% minmax()
  y.vae <- model.vae$encoder %>% predict(y)
  y.range <- range(y.vae)
  y.vae <- (y.vae-y.range[1])/(y.range[2]-y.range[1])
  system.time(deconv.vae <- deconv(t(x.vae),t(y.vae)))
  fit.vae <- t(deconv.vae$fit * (y.range[2]-y.range[1]) + y.range[1])
  fit.vae <- model.vae$decoder %>% predict(fit.vae)
  end.vae <- Sys.time()-st
  st <- Sys.time()
  system.time(model.pca <- qpca(y))
  x.pca <- (x %*% solve(model.pca$Y)[,1:100])
  y.pca <- (y %*% solve(model.pca$Y)[,1:100])
  system.time(deconv.pca <- deconv(t(x.pca),t(y.pca)))
  fit.pca <- cbind(1,t(deconv.pca$fit)) %*% coef(lm(y~y.pca))
  end.pca <- Sys.time()-st
  st <- Sys.time()
  x.sel <- apply(x,1,function(xi){
    # names(which(xi>=quantile(xi,0.99)))
    names(xi)[which(scale(xi)>1.96)]
  }) %>% unlist %>% unique
  y.sel <- y[,colnames(y)%in%x.sel]
  x.sel <- x[,colnames(x)%in%x.sel]
  system.time(deconv.sel <- deconv(t(x.sel),t(y.sel)))
  fit.sel <- dedeconv((x),deconv.sel$coef,(y))
  end.sel <- Sys.time()-st
  cbind(mse = c(raw=mse(fit.raw,y),
                pca=mse(fit.pca,y),
                ae=mse(fit.ae,y),
                vae=mse(fit.vae,y),
                sel=mse(fit.sel,y)),
        time = c(raw=end.raw,
                 pca=end.pca,
                 ae=end.ae,
                 vae=end.vae,
                 sel=end.sel))
}

apply(sapply(test,function(x){x[,1]}),1,function(x){quantile(x,(1:4)/4)})
apply(sapply(test,function(x){x[,2]}),1,function(x){c(mean(x),quantile(x,(1:4)/4))})

############################################################
############################################################



