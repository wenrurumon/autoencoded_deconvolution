
rm(list=ls())
library(keras)

library(dplyr)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/4nan')
load("/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/mdata4causal.rda")
load('4nan.rda')
load('/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/ref2.rda')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/ae/')

############

gene <- rownames(ref)[rownames(ref)%in%colnames(raw_exp)]
bulk.raw<- raw_exp[,match(gene,colnames(raw_exp))]
ref.raw <- t(ref[match(gene,rownames(ref)),])
rm(list=ls()[!ls()%in%c('bulk.raw','ref.raw')])
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')

adgene <- do.call(rbind,strsplit(readLines('AD_gene.txt'),'\t|;'))[,2]

############

#Gene filter
  sel <- which(colnames(bulk.raw) %in% adgene)
  x <- apply(ref.raw[,sel],2,minmax)
  y <- apply(bulk.raw[,sel],2,minmax)
#Deconvolution with rawdata
  system.time(deconv.raw <- deconv(t(x),t(y)))
  fit.raw <- t(deconv.raw$fit)
  mse(fit.raw,y)
#Deconvolution with autoencode
  system.time(model.ae <- ae(y,latent_dim=20,epochs=200))
  plot(model.ae$history)
  x.ae <- model.ae$encoder %>% predict(x)
  y.ae <- model.ae$encoder %>% predict(y)
  system.time(deconv.ae <- deconv(t(x.ae),t(y.ae)))
  fit.ae <- model.ae$decoder %>% predict(t(deconv.ae$fit))
  mse(fit.ae,y)
#Deconvolution with vae
  system.time(model.vae <- vae(y,latent_dim=20,epochs=200))
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
#Result
  c(mse.raw=mse(fit.raw,y),
    mse.pca=mse(fit.pca,y),
    mse.ae=mse(fit.ae,y),
    mse.vae=mse(fit.vae,y))

############

fun4 <- function(i){
  set.seed(i);sel <- sample(1:ncol(ref.raw),400)
  print(paste(i,sum(sel),Sys.time()))
  x <- apply(ref.raw[,sel],2,minmax)
  y <- apply(bulk.raw[,sel],2,minmax)
  system.time(deconv.raw <- deconv(t(x),t(y)))
  fit.raw <- t(deconv.raw$fit)
  system.time(model.ae <- ae(y,latent_dim=100,epochs=1000,verbose=0))
  x.ae <- model.ae$encoder %>% predict(x)
  y.ae <- model.ae$encoder %>% predict(y)
  system.time(deconv.ae <- deconv(t(x.ae),t(y.ae)))
  fit.ae <- model.ae$decoder %>% predict(t(deconv.ae$fit))
  system.time(model.vae <- vae(y,latent_dim=20,epochs=1000))
  x.vae <- model.vae$encoder %>% predict(x) %>% minmax()
  y.vae <- model.vae$encoder %>% predict(y)
  y.range <- range(y.vae)
  y.vae <- (y.vae-y.range[1])/(y.range[2]-y.range[1])
  system.time(deconv.vae <- deconv(t(x.vae),t(y.vae)))
  fit.vae <- t(deconv.vae$fit * (y.range[2]-y.range[1]) + y.range[1])
  fit.vae <- model.vae$decoder %>% predict(fit.vae)
  system.time(model.pca <- qpca(y))
  x.pca <- (x %*% solve(model.pca$Y)[,1:100])
  y.pca <- (y %*% solve(model.pca$Y)[,1:100])
  system.time(deconv.pca <- deconv(t(x.pca),t(y.pca)))
  fit.pca <- cbind(1,t(deconv.pca$fit)) %*% coef(lm(y~y.pca))
  c(mse.raw=mse(fit.raw,y),
    mse.pca=mse(fit.pca,y),
    mse.ae=mse(fit.ae,y),
    mse.vae=mse(fit.vae,y))
}
system.time(test <- sapply(6:10,fun4)%>%t())

############



f
