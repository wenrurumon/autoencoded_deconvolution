
rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
library(nnls)

#Data from autoencoder

setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/')
models <- dir(pattern='encoded')
models <- unique(do.call(rbind,strsplit(models,'\\.'))[,1])
models.data <- lapply(models,function(m){
  y <- paste0(m,'.bulk_encoded') %>% read.csv
  x <- paste0(m,'.ref_encoded') %>% read.csv
  list(y=y,x=x)
})
lapply(models.data,function(x){c(dim(x[[1]]),dim(x[[2]]))})
names(models.data) <- models

#Data from gene selected

genesel <- function(wf,bulk,ref){
  setwd(wf)
  bulk <- fread(bulk)
  ref <- fread(ref)
  x <- ref %>% as.matrix 
  y <- bulk %>% as.matrix
  x.sel <- apply(x,1,function(xi){
    names(which(xi>quantile(xi,0.99)))
  }) %>% unlist %>% unique
  y.sel <- y[,colnames(y)%in%x.sel]
  x.sel <- x[,colnames(x)%in%x.sel]
  list(y=y.sel,x=x.sel)
}
genesel.load <- list(
  c("/Users/wenrurumon/Documents/uthealth/deconv/adni_data",'bulk_adni.csv','ref_adni.csv'),
  c('/Users/wenrurumon/Documents/uthealth/deconv/lmy','bulk_lmy.csv','ref_lmy.csv'),
  c('/Users/wenrurumon/Documents/uthealth/deconv/ae','bulk.csv','reference.csv')
)
sel.data <- lapply(genesel.load,function(x){
  genesel(x[1],x[2],x[3])
})
lapply(sel.data,function(x){c(dim(x[[1]]),dim(x[[2]]))})
names(sel.data) <- gsub('ae','sel',models[1:3])

models.data <- do.call(c,list(models.data,sel.data))
t(sapply(models.data,function(x){c(dim(x[[1]]),dim(x[[2]]))}))

#Sourcing

setwd('/Users/wenrurumon/Documents/uthealth/deconv/lmy')
for(i in dir(pattern='R')){
  source(i)
}
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/')
load('/Users/wenrurumon/Documents/uthealth/deconv/4nan/4nan.rda')
ref_cluster <- substr(colnames(ref),1,2)

j <- 0

#DeconvSTF

models.deconv <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv$coef,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
tofit.deconv <- lapply(models.deconv,function(x){
  x$deconv$coef
})
rlt.deconv <- lapply(models.deconv,function(x){
  t(apply(x$deconv$coef,2,function(x){tapply(x,ref_cluster,sum)}))
})

#DeconvLM

models.deconlm <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- deconv.lm(t(m$x),t(m$y),ifprint=F)
  rlt.fit <- dedeconv((m$x),rlt.deconv,(m$y))
  list(deconv=rlt.deconv,fit=rlt.fit)
})
tofit.deconlm <- lapply(models.deconlm,function(x){
  x$deconv
})
rlt.deconlm <- lapply(models.deconlm,function(x){
  t(apply(x$deconv,2,function(x){tapply(x,ref_cluster,sum)}))
})

#LMY

models.deconvlmy <- lapply(models.data,function(m){
  print(paste('####################',j<<-j+1,'####################'))
  rlt.deconv <- music_quick(t(m$y),t(m$x),verbose=F)
  rlt.deconv
})
tofit.deconvnnls <- lapply(models.deconvlmy,function(x){
  t(x[[2]])
})
tofit.deconvlmy <- lapply(models.deconvlmy,function(x){
  t(x[[1]])
})
rlt.deconvnnls <- lapply(models.deconvlmy,function(x){
  t(apply(x[[2]],1,function(x){tapply(x,ref_cluster,sum)}))
})
rlt.deconvlmy <- lapply(models.deconvlmy,function(x){
  t(apply(x[[1]],1,function(x){tapply(x,ref_cluster,sum)}))
})

######################################################
######################################################

rlt <- list(
  rlt.deconv,rlt.deconlm,rlt.deconvnnls,rlt.deconvlmy
)
save(rlt,file='deconv_rlt.rda')
rlt <- list(
  tofit.deconv,tofit.deconlm,tofit.deconvnnls,tofit.deconvlmy
)
names(rlt[[1]]) <- sapply(strsplit(names(rlt[[1]]),'_'),function(x){paste0('stf_',x[1],'_',x[2])})
names(rlt[[2]]) <- sapply(strsplit(names(rlt[[2]]),'_'),function(x){paste0('own_',x[1],'_',x[2])})
names(rlt[[3]]) <- sapply(strsplit(names(rlt[[3]]),'_'),function(x){paste0('nnls_',x[1],'_',x[2])})
names(rlt[[4]]) <- sapply(strsplit(names(rlt[[4]]),'_'),function(x){paste0('lmy_',x[1],'_',x[2])})
rlt <- do.call(c,rlt)
names(rlt) <- gsub('16055','adni',names(rlt))
names(rlt) <- gsub('17121','lmy',names(rlt))
names(rlt) <- gsub('25910','rush',names(rlt))
rlt <- lapply(rlt,t)

######################################################
######################################################

rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
library(nnls)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/')
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')
load('deconv_coef.rda')
models <- dir(pattern='encoded')
models <- unique(do.call(rbind,strsplit(models,'\\.'))[,1])
models.data <- lapply(models,function(m){
  y <- paste0(m,'.bulk_encoded') %>% read.csv
  x <- paste0(m,'.ref_encoded') %>% read.csv
  list(y=y,x=x)
})
genesel <- function(wf,bulk,ref){
  setwd(wf)
  bulk <- fread(bulk)
  ref <- fread(ref)
  x <- ref %>% as.matrix 
  y <- bulk %>% as.matrix
  list(y=y,x=x)
}
genesel.load <- list(
  c("/Users/wenrurumon/Documents/uthealth/deconv/adni_data",'bulk_adni.csv','ref_adni.csv'),
  c('/Users/wenrurumon/Documents/uthealth/deconv/lmy','bulk_lmy.csv','ref_lmy.csv'),
  c('/Users/wenrurumon/Documents/uthealth/deconv/ae','bulk.csv','reference.csv')
)
sel.data <- lapply(genesel.load,function(x){
  genesel(x[1],x[2],x[3])
})
names(models.data) <- models 
names(models.data) <- sapply(strsplit(names(models.data),'_'),function(x){paste0(x[1],'_',x[2])})
names(models.data) <- gsub('16055','adni',names(models.data))
names(models.data) <- gsub('17121','lmy',names(models.data))
names(models.data) <- gsub('25910','rush',names(models.data))
names(sel.data) <- c('sel_adni','sel_lmy','sel_rush')
models.data <- do.call(c,list(models.data,sel.data))
test <- lapply(1:length(rlt),function(i){
  Coef <- rlt[[i]]
  Data <- models.data[[which(names(models.data)==substr(names(rlt)[i],regexpr('_',names(rlt)[i])+1,nchar(names(rlt)[i])))]]
  out <- list(dedeconv((Data$x),t(Coef),(Data$y)))
  names(out) <- names(models.data)[which(names(models.data)==substr(names(rlt)[i],regexpr('_',names(rlt)[i])+1,nchar(names(rlt)[i])))]
  out
})
test <- do.call(c,test)
names(test) <- names(rlt)
rlt <- test 
for (i in 1:length(test)){
  write.csv(rlt[[i]],paste0(names(rlt)[i],'.bulk_fit'),row.names=F)
}
