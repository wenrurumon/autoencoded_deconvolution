rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/encoded_adni')

models <- dir(pattern='encoded')
models <- unique(do.call(rbind,strsplit(models,'\\.'))[,1])
models.data <- lapply(models,function(m){
  y <- paste0(m,'.bulk_encoded') %>% read.csv
  x <- paste0(m,'.ref_encoded') %>% read.csv
  list(y=y,x=x)
})

j <- 0

#DeconvSTF

models.deconv <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv$coef,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
names(models.deconv) <- models

#DeconvLM

models.deconlm <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv.lm(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
names(models.deconlm) <- models

#Raw

setwd("/Users/wenrurumon/Documents/uthealth/deconv/adni_data")
bulk <- fread('bulk_adni.csv')
ref <- fread('ref_adni.csv')
x <- ref %>% as.matrix 
y <- bulk %>% as.matrix
x.sel <- apply(x,1,function(xi){
  names(which(xi>quantile(xi,0.99)))
}) %>% unlist %>% unique
y.sel <- y[,colnames(y)%in%x.sel]
x.sel <- x[,colnames(x)%in%x.sel]
dim(y.sel)
system.time(deconv.sel <- deconv(t(x.sel),t(y.sel),ifprint=T))
models.sel <- list(deconv=deconv.sel,fit=dedeconv((x),deconv.sel$coef,(y)))
system.time(deconv.sel <- deconv.lm(t(x.sel),t(y.sel),ifprint=T))
fit.sel <- dedeconv((x),deconv.sel,(y))
models.sellm <- list(deconv=deconv.sel,fit=fit.sel)

rlt.adni <- list(
  models.deconv,
  models.deconlm,
  models.sel,
  models.sellm
)
# save(rlt.adni,file='deconv_rlt_adni.rda')
# 
# write.csv(t(rlt.adni[[1]][[2]]$deconv[[1]]),'adni_ae_stf.csv')
# write.csv(t(rlt.adni[[1]][[6]]$deconv[[1]]),'adni_vae_stf.csv')
# write.csv(t(rlt.adni[[2]][[2]][[1]]),'adni_ae_lm.csv')
# write.csv(t(rlt.adni[[2]][[6]][[1]]),'adni_vae_lm.csv')
# write.csv(t(rlt.adni[[3]][[1]]$coef),'adni_sel_stf.csv')
# write.csv(t(rlt.adni[[4]][[1]]),'adni_sel_lm.csv')

##################################################################
##################################################################

rm(list=ls())
library(keras)
library(dplyr)
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/encoded_rush')

models <- dir(pattern='encoded')
models <- unique(do.call(rbind,strsplit(models,'\\.'))[,1])
models.data <- lapply(models,function(m){
  y <- paste0(m,'.bulk_encoded') %>% read.csv
  x <- paste0(m,'.ref_encoded') %>% read.csv
  list(y=y,x=x)
})

j <- 0

#DeconvSTF

models.deconv <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv$coef,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
names(models.deconv) <- models

#DeconvLM

models.deconlm <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv.lm(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
names(models.deconlm) <- models

#Raw

setwd("/Users/wenrurumon/Documents/uthealth/deconv/ae")
bulk <- fread('bulk.csv') 
ref <- fread('reference.csv')
x <- ref %>% as.matrix 
y <- bulk %>% as.matrix
x.sel <- apply(x,1,function(xi){
  names(which(xi>quantile(xi,0.99)))
}) %>% unlist %>% unique
y.sel <- y[,colnames(y)%in%x.sel]
x.sel <- x[,colnames(x)%in%x.sel]
dim(y.sel)
system.time(deconv.sel <- deconv(t(x.sel),t(y.sel),ifprint=T))
models.sel <- list(deconv=deconv.sel,fit=dedeconv((x),deconv.sel$coef,(y)))
system.time(deconv.sel <- deconv.lm(t(x.sel),t(y.sel),ifprint=T))
fit.sel <- dedeconv((x),deconv.sel,(y))
models.sellm <- list(deconv=deconv.sel,fit=fit.sel)

rlt.rush <- list(
  models.deconv,
  models.deconlm,
  models.sel,
  models.sellm
)
# save(rlt.rush,file='deconv_rlt_rush.rda')
# 
# write.csv(t(rlt.rush[[1]][[1]]$deconv[[1]]),'rush_ae_stf.csv')
# write.csv(t(rlt.rush[[1]][[3]]$deconv[[1]]),'rush_vae_stf.csv')
# write.csv(t(rlt.rush[[2]][[1]][[1]]),'rush_ae_lm.csv')
# write.csv(t(rlt.rush[[2]][[3]][[1]]),'rush_vae_lm.csv')
# write.csv(t(rlt.rush[[3]][[1]]$coef),'rush_sel_stf.csv')
# write.csv(t(rlt.rush[[4]][[1]]),'rush_sel_lm.csv')

##################################################################
##################################################################

rm(list=ls())
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt')
for (i in dir(pattern='rda')) {print(i); load(i)}
ls()
adni1 <- rlt.adni[[1]]$ae_16055_1024_250_256_4000_2$deconv$coef
adni2 <- rlt.adni[[2]]$ae_16055_1024_250_256_4000_2$deconv
adni3 <- rlt.adni[[1]]$ae_16055_1024_200_256_4000_2$deconv$coef
adni4 <- rlt.adni[[1]]$vae_16055_1024_250_256_4000_2$deconv$coef

head(adni1[,1:10])
head(adni3[,1:10])
head(adni4[,1:10])

rush1 <- rlt.rush[[1]]$ae_25910_1024_250_256_4000_2$deconv$coef
rush2 <- rlt.rush[[1]]$ae_25910_1024_400_256_4000_2$deconv$coef
rush3 <- rlt.rush[[1]]$vae_25910_1024_250_256_4000_2$deconv$coef

head(rush1[,1:5])
head(rush2[,1:5])
head(rush3[,1:5])
