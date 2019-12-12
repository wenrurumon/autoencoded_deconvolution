
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
pcasel <- function(wf,bulk,ref){
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
names(sel.data) <- gsub('ae','sel',models[1:3])
sel.data2 <- lapply(genesel.load,function(x){
  pcasel(x[1],x[2],x[3])
})
lapply(sel.data2,function(x){c(dim(x[[1]]),dim(x[[2]]))})
names(sel.data2) <- gsub('ae','pca',models[1:3])
models.data <- do.call(c,list(models.data,sel.data,sel.data2))
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
models.data <- models.data[-1:-9]

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
save(rlt,file='deconv_rlt2.rda')
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
rlt1 <- rlt[match(sort(sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))
                 ,sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))]
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
rlt2 <- rlt[match(sort(sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))
                 ,sapply(strsplit(names(rlt),'_'),function(x){paste(x[2],x[1],x[3],sep="_")}))]
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

########################################################
########################################################

par(mfrow=c(1,1))
pheatmap(rlt.dis,
         color = colorRampPalette(c("firebrick3", "white"))(50),
         main='KL Distance cross methods')
rltmelt <- melt(rlt.dis)
rltmelt <- cbind(rltmelt,m1=do.call(rbind,strsplit(paste(rltmelt[,1]),'_'))[,2])
rltmelt <- cbind(rltmelt,m2=do.call(rbind,strsplit(paste(rltmelt[,2]),'_'))[,2])

rltmelt %>% group_by(m1) %>% summarise(mean(value))
rltmelt %>% filter(m1==m2) %>% group_by(m1) %>% summarise(mean(value))

boxplot(value~Var1,data=melt(rlt.dis)%>%filter(Var1!=Var2),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='')

par(mfrow=c(1,4))
boxplot(value~v1,data=fdata('_ae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='autoencoder')
boxplot(value~v1,data=fdata('_sel_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='gene selected')
boxplot(value~v1,data=fdata('_vae_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='variational autoencoder')
boxplot(value~v1,data=fdata('_pca_'),
        par(las='2',mar=c(10,5,5,5)),
        xlab='',ylab='',main='PCA')
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
