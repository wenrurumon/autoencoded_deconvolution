
rm(list=ls())
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt')
library(data.table)
library(dplyr)
bulk_fit <- lapply(dir(pattern='sel'),fread)
names(bulk_fit) <- dir(pattern='sel')
minmax <- function(x){
  x <- apply(x,2,function(x){(x-min(x))/(max(x)-min(x))})
  x[is.na(x)] <- 0
  x
}
mse <- function(x,y){
  mean((x-y)^2)/mean((y-mean(y))^2)
}

bulk_lmy <- fread('/Users/wenrurumon/Documents/uthealth/deconv/lmy/bulk_lmy.csv') %>% minmax
bulk_rush <- fread('/Users/wenrurumon/Documents/uthealth/deconv/ae/bulk.csv') %>% minmax
bulk_adni <- fread('/Users/wenrurumon/Documents/uthealth/deconv/adni_data/bulk_adni.csv') %>% minmax
bulk_fit <- lapply(bulk_fit,minmax)

fit_sel <- do.call(c,list(sapply(bulk_fit[grepl('lmy.bulk',names(bulk_fit))],function(x){mse(x,bulk_lmy)}),
sapply(bulk_fit[grepl('rush.bulk',names(bulk_fit))],function(x){mse(x,bulk_rush)}),
sapply(bulk_fit[grepl('adni.bulk',names(bulk_fit))],function(x){mse(x,bulk_adni)})))


fit_sel <- data.table('gene_selected',names(fit_sel),fit_sel)
fit_model <- fread('mse_score.csv',header=T)[,2:4]
colnames(fit_sel) <- colnames(fit_model) <- c('decoder','model','mse')
rlt <- rbind(fit_model,fit_sel)[,-1]
rlt <- data.table(do.call(rbind,strsplit(rlt$model,'_|\\.')),rlt[,-1])[,-4:-5]
colnames(rlt) <- c('deconvolution','reduction','dataset','mse')

boxplot(mse~reduction,data=rlt)
boxplot(mse~deconvolution,data=rlt)
