
rm(list=ls())
library(data.table)
library(dplyr)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/4nan')
load("/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/mdata4causal.rda")
load('4nan.rda')
load('/Users/wenrurumon/Documents/uthealth/deconv/causal5.1/ref2.rda')
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')

setwd('/Users/wenrurumon/Documents/uthealth/deconv/adni_data')
raw <- fread('ADNI_Gene_Expression_Profile.csv',header=T)
id <- raw[2,,drop=T]
raw <- raw[-1:-8,]
colnames(raw) <- ifelse(paste(id)=='',colnames(raw),paste(id))
x.raw <- apply(as.matrix(raw[,-1:-3][,-745]),2,as.numeric)
ud <- function(x){
  rlt <- tapply(x,raw$V3,mean)
  return(rlt)
}
x <- apply(x.raw,2,ud) 
rownames(x) <- names(ud(x.raw[,1]))

gene.adni <- rownames(x)
gene.ref <- rownames(ref)
gene <- gene.adni[gene.adni%in%gene.ref]
bulk_adni <- apply(t(x[rownames(x)%in%gene,]),2,minmax)
ref_adni <- apply(t(ref[rownames(ref)%in%gene,]),2,minmax)
write.csv(bulk_adni,'bulk_adni.csv',row.names=F)
write.csv(ref_adni,'ref_adni.csv',row.names=F)
