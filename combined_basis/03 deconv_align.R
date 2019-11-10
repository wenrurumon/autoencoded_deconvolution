
rm(list=ls())
library(dplyr)
library(pheatmap)
library(reshape2)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/')
load('deconv_rlt.rda')
names(rlt[[1]]) <- sapply(strsplit(names(rlt[[1]]),'_'),function(x){paste0('stf_',x[1],'_',x[2])})
names(rlt[[2]]) <- sapply(strsplit(names(rlt[[2]]),'_'),function(x){paste0('own_',x[1],'_',x[2])})
names(rlt) <- NULL
rlt <- do.call(c,rlt)
names(rlt) <- gsub('16055','adni',names(rlt))
names(rlt) <- gsub('17121','lmy',names(rlt))
names(rlt) <- gsub('25910','rush',names(rlt))
rlt <- lapply(rlt,t)

rlt.quantile <- lapply(rlt,function(x){
  apply(x,1,function(x){
    x <- tapply(round(x*100),round(x*100),length)
    out <- rep(0,101)
    out[as.numeric(names(x))+1] <- x
    out <- out/sum(out)
    out[out==0] <- 0.0000000001
    out
  })
})
kl <- function(x,y){
  sum(x * log(x/y))
}
rlt.dis <- sapply(rlt.quantile,function(x){
  sapply(rlt.quantile,function(y){
    (kl(x,y)+kl(y,x))/2
  })
})

par(mfrow=c(1,1))

pheatmap(rlt.dis,
         color = colorRampPalette(c("firebrick3", "white"))(50))
boxplot(value~Var1,data=melt(rlt.dis),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')

fdata <- function(x){
  r <- filter(melt(rlt.dis),grepl(x,Var1)&grepl(x,Var2))
  r <- data.frame(v1=paste(r$Var1),v2=paste(r$Var2),value=r$value)
  r
}

par(mfrow=c(1,3))
boxplot(value~v1,data=fdata('adni'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')
boxplot(value~v1,data=fdata('lmy'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')
boxplot(value~v1,data=fdata('rush'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')

