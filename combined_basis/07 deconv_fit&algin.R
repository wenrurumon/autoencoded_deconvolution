rm(list=ls())
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt')
library(data.table)
library(dplyr)
bulk_fit <- lapply(dir(pattern='sel|pca'),function(x){print(x);fread(x)})
names(bulk_fit) <- dir(pattern='sel|pca')
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
fit_sel <- data.table(rep(c('pca','sel'),length=24),names(fit_sel),fit_sel)
fit_model <- fread('mse_score.csv',header=T)[,2:4]
colnames(fit_sel) <- colnames(fit_model) <- c('decoder','model','mse')
rlt <- rbind(fit_model,fit_sel)[,-1]
rlt <- data.table(do.call(rbind,strsplit(rlt$model,'_|\\.')),rlt[,-1])[,-4:-5]
colnames(rlt) <- c('deconvolution','reduction','dataset','mse')
rlt.mse <- rlt

boxplot(mse~reduction,data=rlt,main='MSE by reduction method')
boxplot(mse~deconvolution,data=rlt,main='MSE by deconvolution method')

########################################
########################################

# rm(list=ls())
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
rlt1 <- rlt
load('deconv_rlt2.rda')
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
rlt2 <- rlt
rlt <- c(rlt1,rlt2)

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

rlt.mse <- as.matrix(rlt.mse)
rlt.kl <- melt(rlt.dis)
rlt.kl <- cbind(
  rlt.kl,
  m1 = do.call(rbind,strsplit(paste(rlt.kl$Var1),'_'))[,2],
  m2 = do.call(rbind,strsplit(paste(rlt.kl$Var2),'_'))[,2]
)
rlt.kl <- cbind(
  (rlt.kl %>% group_by(Var1) %>% summarise(kl1 = mean(value))),
  k2=(rlt.kl %>% filter(m1==m2) %>% group_by(Var1) %>% summarise(kl2 = mean(value)))$kl2
)
rlt.mse <- data.frame(Var1=paste(rlt.mse[,1],rlt.mse[,2],rlt.mse[,3],sep='_'),
                      mse=as.numeric(paste(rlt.mse[,4])))
rlt <- merge(rlt.mse,rlt.kl,by='Var1')
rlt <- cbind(do.call(rbind,strsplit(paste(rlt$Var1),'_')),rlt[,-1])
colnames(rlt)[1:3] <- c('deconvolution','reduction','dataset')

plot(rlt$mse,rlt$kl1)

library(ggplot2)
colnames(rlt)[4:6] <- c('MSE','KL_distance','KL_distance2') 
rlt$reduction <- toupper(rlt$reduction)
rlt$deconvolution <- toupper(rlt$deconvolution)
ggplot(rlt,aes(x = MSE, y = KL_distance, col= reduction)) + geom_point(size=3)
ggplot(rlt,aes(x = MSE, y = KL_distance2, col= reduction)) + geom_point(size=3)
ggplot(rlt,aes(x = MSE, y = KL_distance, col= deconvolution)) + geom_point(size=3)
ggplot(rlt,aes(x = MSE, y = KL_distance, col= dataset)) + geom_point(size=3)

########################################################
########################################################

par(mfrow=c(1,1))
pheatmap(rlt.dis,
         color = colorRampPalette(c("firebrick3", "white"))(50),
         main='KL Distance cross methods')

boxplot(value~Var1,data=melt(rlt.dis)%>%filter(Var1!=Var2),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')

par(mfrow=c(1,2))
boxplot(value~v1,data=fdata('_ae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='Autoencoder',ylim=c(0,140))
boxplot(value~v1,data=fdata('_sel_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='Gene Selected',ylim=c(0,140))
boxplot(value~v1,data=fdata('_vae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='Variational Autoencoder',ylim=c(0,140))
boxplot(value~v1,data=fdata('_pca_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='PCA',ylim=c(0,140))

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
