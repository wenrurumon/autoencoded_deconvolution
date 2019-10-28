
rm(list=ls())
library(data.table)
library(dplyr)
source('/Users/wenrurumon/Documents/uthealth/deconv/ae/model.R')

setwd('~/Documents/uthealth/deconv/ae')
bulk_rush <- fread('bulk.csv')
setwd('~/Documents/uthealth/deconv/adni_data')
bulk_adni <- fread('bulk_adni.csv')

read_csv <- function(x){
  print(i<<-i+1)
  read.csv(x)
}

i <- 0
setwd('~/Documents/uthealth/deconv/rlt/rush')
fit_rush <- lapply(dir(pattern='bulk_fit'),read_csv)
setwd('~/Documents/uthealth/deconv/rlt/adni')
fit_adni <- lapply(dir(pattern='bulk_fit'),fread)
