
rm(list=ls())
library(keras)
library(dplyr)
library(data.table)
setwd('/Users/wenrurumon/Documents/uthealth/deconv/rlt/')
load('/Users/wenrurumon/Documents/uthealth/deconv/4nan/4nan.rda')
ref_cluster <- substr(colnames(ref),1,2)
load('deconv_rlt.rda')
toclust <- function(x){
  tapply(x,ref_cluster,sum)
}
rlt_ae_stf <- lapply(rlt[[1]],function(x){
  apply(x$deconv$coef,2,toclust)
})
rlt_ae_lm <- lapply(rlt[[2]],function(x){
  apply(x$deconv,2,toclust)
})
rlt_sel_stf <- lapply(
  list(rlt[[3]][[1]]$deconv$coef,rlt[[4]][[1]]$deconv$coef),
  function(x){apply(x,2,toclust)}
)
rlt_sel_lm <- lapply(
  list(rlt[[3]][[2]]$deconv,rlt[[4]][[2]]$deconv),
  function(x){apply(x,2,toclust)}
)

s <- function(x){
  c(summary(x),count=length(x))
}

lapply(rlt_ae_stf,function(x){apply(x,1,s)})
lapply(rlt_ae_lm,function(x){apply(x,1,s)})
lapply(rlt_sel_stf,function(x){apply(x,1,s)})
lapply(rlt_sel_lm,function(x){apply(x,1,s)})

##########

names(rlt_ae_stf) <- paste('sanford',rep(c('adni','rush'),2),rep(c('ae','vae'),each=2))
names(rlt_ae_lm) <- paste('own',rep(c('adni','rush'),2),rep(c('ae','vae'),each=2))
names(rlt_sel_stf) <- paste('stanford',c('rush','adni'),'gene_selected')
names(rlt_sel_lm) <- paste('own',c('rush','adni'),'gene_selected')
rlt <- list(
  rlt_ae_stf,
  rlt_ae_lm,
  rlt_sel_stf,
  rlt_sel_lm
)
rlt <- do.call(c,rlt)
heatmap(t(sapply(rlt,function(x){apply(x,1,mean)})),Colv=NA)

rlt1 <- sapply(
  lapply(rlt,function(x){apply(x,1,mean)}),function(x){
    sapply(lapply(rlt,function(x){apply(x,1,mean)}),function(y){
      sum((x-y)^2)
    })
  }
)
heatmap(1-rlt1)

#########

rlt.quantile <- lapply(rlt,function(x){
  apply(x,1,function(x){quantile(x,(0:100)/100)})
})
names(rlt.quantile) <- 
  sapply(strsplit(names(rlt.quantile),' '),function(x){
    paste(toupper(substr(x,1,1)),collapse='_')
  })
rlt.dis1 <- sapply(rlt.quantile,function(x){
  sapply(rlt.quantile,function(y){
    sum((x-y)^2)
  })
})

########

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
# names(rlt.quantile) <-
#   sapply(strsplit(names(rlt.quantile),' '),function(x){
#     paste(toupper(substr(x,1,1)),collapse='_')
#   })

kl <- function(x,y){
  sum(x * log(x/y))
}
rlt.dis1 <- sapply(rlt.quantile,function(x){
  sapply(rlt.quantile,function(y){
    sum((x-y)^2)
  })
})
rlt.dis2 <- sapply(rlt.quantile,function(x){
  sapply(rlt.quantile,function(y){
    kl(x,y)
  })
})
rlt.dis2 <- rlt.dis2 + t(rlt.dis2)
heatmap(rlt.dis1/max(rlt.dis1))
heatmap(1-rlt.dis2/max(rlt.dis2))
