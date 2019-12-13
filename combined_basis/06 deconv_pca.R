
rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
library(nnls)

#Data from gene selected

pcabulk <- function(bulk,ref){
  Y.raw <- as.matrix(bulk)
  X.raw <- as.matrix(ref)
  Y.svd <- svd(scale(Y.raw))
  Y.prop <- (Y.svd$d)^2/nrow(Y.raw)
  Y.prop <- cumsum(Y.prop/sum(Y.prop))
  Y.score <- scale(Y.raw) %*% Y.svd$v
  X.score <- scale(X.raw) %*% Y.svd$v
  list(y=Y.score,x=X.score,bulk=bulk,ref=ref)
}
genesel <- function(wf,bulk,ref){
  setwd(wf)
  bulk <- fread(bulk)
  ref <- fread(ref)
  x <- ref %>% as.matrix 
  y <- bulk %>% as.matrix
  x.sel <- apply(x,1,function(xi){
    names(which(xi>quantile(xi,0)))
  }) %>% unlist %>% unique
  y.sel <- y[,colnames(y)%in%x.sel]
  x.sel <- x[,colnames(x)%in%x.sel]
  pcabulk(y.sel,x.sel)
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
models.data <- sel.data

#Sourcing

setwd('/Users/wenrurumon/Documents/uthealth/deconv/lmy')
for(i in dir(pattern='R')){
  print(i)
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
  # rlt.fit <- dedeconv((m$x),rlt.deconv$coef,(m$y))
  rlt.fit <- dedeconv((m$ref),rlt.deconv$coef,(m$bulk))
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
  # rlt.fit <- dedeconv((m$x),rlt.deconv,(m$y))
  rlt.fit <- dedeconv((m$ref),rlt.deconv,(m$bulk))
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
save(rlt,file='deconv2_rlt.rda')

rlt <- list(
  tofit.deconv,tofit.deconlm,tofit.deconvnnls,tofit.deconvlmy
)
names(rlt[[1]]) <- paste0('stf_pca_',c('adni','lmy','rush'))
names(rlt[[2]]) <- paste0('own_pca_',c('adni','lmy','rush'))
names(rlt[[3]]) <- paste0('nnls_pca_',c('adni','lmy','rush'))
names(rlt[[4]]) <- paste0('lmy_pca_',c('adni','lmy','rush'))
for(i in 1:3){
  rlt[[1]][[i]] <- dedeconv(models.data[[i]]$ref,rlt[[1]][[i]],models.data[[i]]$bulk)
  rlt[[2]][[i]] <- dedeconv(models.data[[i]]$ref,rlt[[2]][[i]],models.data[[i]]$bulk)
  rlt[[3]][[i]] <- dedeconv(models.data[[i]]$ref,rlt[[3]][[i]],models.data[[i]]$bulk)
}
rlt <- do.call(c,rlt)
for(i in 1:length(rlt)){
  write.csv(rlt[[i]],paste0(names(rlt)[[i]],'.bulk_fit'),row.names=F)
}
