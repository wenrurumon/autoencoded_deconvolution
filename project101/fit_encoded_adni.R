rm(list=ls())
library(keras)
library(dplyr)
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/adni')
models <- dir(pattern='encoded')
models <- unique(do.call(rbind,strsplit(models,'\\.'))[,1])
models.data <- lapply(models,function(m){
  y <- paste0(m,'.bulk_encoded') %>% read.csv
  x <- paste0(m,'.ref_encoded') %>% read.csv
  list(y=y,x=x)
})
j <- 0
models.deconv <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv(t(m$x),t(m$y),ifprint=T)
  rlt.fit <- dedeconv((m$x),rlt.deconv$coef,(m$y))
  rlt.fit
})
for(i in 1:4){
  write.csv(models.deconv[[i]],paste0(models[i],'.fit_encoded'),row.names=F)
}

check <- lapply(models,function(m){
  read.csv(paste0(m,'.fit_encoded'))
})
