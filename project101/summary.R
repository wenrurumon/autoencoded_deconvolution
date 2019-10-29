
rm(list=ls())
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')

setwd("/Users/wenrurumon/Documents/uthealth/deconv/adni_data")
bulk <- fread('bulk_adni.csv')
ref <- fread('ref_adni.csv')

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
fit.sel <- dedeconv((x),deconv.sel$coef,(y))
mse.sel <- mse(fit.sel,y)
rlt.sel <- list(fit=fit.sel,mse=mse.sel)

system.time(deconv.sel <- deconv.lm(t(x.sel),t(y.sel),ifprint=T))
fit.sel <- dedeconv((x),deconv.sel,(y))
mse.sel <- mse(fit.sel,y)
rlt.sel <- list(fit=fit.sel,mse=mse.sel)
