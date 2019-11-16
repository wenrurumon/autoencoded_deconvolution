
rm(list=ls())
library(pheatmap)

setwd('/Users/wenrurumon/Documents/uthealth/deconv/lmy')
for(i in dir(pattern='R')){
  source(i)
}

setwd("/Users/wenrurumon/Documents/uthealth/deconv/rlt")
load('deconv_rlt.rda')
names(rlt[[1]]) <- sapply(strsplit(names(rlt[[1]]),'_'),function(x){paste0('stf_',x[1],'_',x[2])})
names(rlt[[2]]) <- sapply(strsplit(names(rlt[[2]]),'_'),function(x){paste0('own_',x[1],'_',x[2])})
names(rlt[[3]]) <- sapply(strsplit(names(rlt[[3]]),'_'),function(x){paste0('nnls_',x[1],'_',x[2])})
names(rlt[[4]]) <- sapply(strsplit(names(rlt[[4]]),'_'),function(x){paste0('lmy_',x[1],'_',x[2])})
rlt <- do.call(c,rlt)
names(rlt) <- gsub('16055','adni',names(rlt))
names(rlt) <- gsub('17121','lmy',names(rlt))
names(rlt) <- gsub('25910','rush',names(rlt))
rlt <- lapply(rlt,t)
rlt <- rlt[match(sort(sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))
  ,sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))]

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

fdata <- function(x){
  r <- filter(melt(rlt.dis),grepl(x,Var1)&grepl(x,Var2)&Var1!=Var2)
  r <- data.frame(v1=paste(r$Var1),v2=paste(r$Var2),value=r$value)
  r
}

########################################################
########################################################

par(mfrow=c(1,1))
pheatmap(rlt.dis,
         color = colorRampPalette(c("firebrick3", "white"))(50),
         main='KL Distance cross methods')

boxplot(value~Var1,data=melt(rlt.dis)%>%filter(Var1!=Var2),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')

par(mfrow=c(1,3))
boxplot(value~v1,data=fdata('_ae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='autoencoder')
boxplot(value~v1,data=fdata('_sel_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='gene selected')
boxplot(value~v1,data=fdata('_vae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='variational autoencoder')

par(mfrow=c(1,4))
boxplot(value~v1,data=fdata('lmy_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='muSic')
boxplot(value~v1,data=fdata('own_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='own')
boxplot(value~v1,data=fdata('stf_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='stanford')
boxplot(value~v1,data=fdata('nnls_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='nnls')

