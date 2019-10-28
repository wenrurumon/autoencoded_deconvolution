
rm(list=ls())
setwd('/lustre/wangjc01/huzixin/deconv/data')

mse <- function(x,y){
  sum((x-y)^2)/sum((y-mean(y))^2)
}

minmax <- function(x){
  (x-min(x)) / (max(x)-min(x))
}

bulk_rush <- apply(read.csv('bulk.csv'),2,minmax)
bulk_adni <- apply(read.csv('bulk_adni.csv'),2,minmax)

setwd('/lustre/wangjc01/huzixin/deconv/log/rlt_adni')
fit.adni <- lapply(dir(pattern='bulk_fit_'),read.csv)
names(fit.adni) <- dir(pattern='bulk_fit_')

setwd('/lustre/wangjc01/huzixin/deconv/log/rlt_rush')
fit.rush <- lapply(dir(pattern='bulk_fit_'),read.csv)
names(fit.rush) <- dir(pattern='bulk_fit_')

cbind(sapply(fit.adni,function(x){mse(x,y=bulk_adni)}))
cbind(sapply(fit.rush,function(x){mse(x,y=bulk_rush)}))

