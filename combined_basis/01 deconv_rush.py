rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/encoded')

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
load('/Users/wenrurumon/Documents/uthealth/deconv/4nan/4nan.rda')
ref_cluster <- substr(colnames(ref),1,2)
rlt.deconv <- lapply(models.deconv,function(x){
  t(apply(x$deconv$coef,2,function(x){tapply(x,ref_cluster,sum)}))
})
# write.csv(rlt.deconv[[3]],paste0(models[[3]],'.csv'))
hist(cor(t(rlt.deconv[[3]]),t(rlt.deconv[[1]])))

#DeconvLM

models.deconlm <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv.lm(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
names(models.deconlm) <- models
rlt.deconlm <- lapply(models.deconlm,function(x){
  t(apply(x$deconv,2,function(x){tapply(x,ref_cluster,sum)}))
})

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

hist(cor(apply(models.sel$deconv$coef,2,function(x) tapply(x,ref_cluster,sum))
         ,t(rlt.deconv[[3]])) %>% diag)

rlt.rush <- list(
  models.deconv,
  models.deconlm,
  models.sel,
  models.sellm
)
