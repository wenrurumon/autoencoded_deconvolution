
rm(list=ls())
suppressMessages(library(MuSiC))
suppressMessages(library(cowplot))
suppressMessages(library(xbioc))
suppressMessages(library(keras))
minmax <- function(x){
  (x-min(x))/(max(x)-min(x))
}
vae <- function(X,intermediate_dim=ncol(X),latent_dim=floor(ncol(X)/10),epsilon_std=1,epochs=50,batch_size=100,klpen=0.1,verbose=2){
  original_dim <- ncol(X)
  x <- layer_input(shape = c(original_dim))
  h <- layer_dense(x, intermediate_dim, activation = "sigmoid")
  z_mean <- layer_dense(h, latent_dim)
  z_log_var <- layer_dense(h, latent_dim)
  sampling <- function(arg){
    z_mean <- arg[, 1:(latent_dim)]
    z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
    epsilon <- k_random_normal(
      shape = c(k_shape(z_mean)[[1]]), 
      mean=0.,
      stddev=epsilon_std
    )
    z_mean + k_exp(z_log_var/2)*epsilon
  }
  z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
    layer_lambda(sampling)
  decoder_h <- layer_dense(units = intermediate_dim, activation = "sigmoid")
  decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
  h_decoded <- decoder_h(z)
  x_decoded_mean <- decoder_mean(h_decoded)
  vae <- keras_model(x, x_decoded_mean)
  encoder <- keras_model(x, z_mean)
  decoder_input <- layer_input(shape = latent_dim)
  h_decoded_2 <- decoder_h(decoder_input)
  x_decoded_mean_2 <- decoder_mean(h_decoded_2)
  generator <- keras_model(decoder_input, x_decoded_mean_2)
  vae_loss <- function(x, x_decoded_mean){
    xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    xent_loss + 0.1*kl_loss
  }
  vae %>% compile(optimizer = "rmsprop", loss = vae_loss, metrics = c('accuracy') )
  history <- vae %>% fit(
    X,X, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size,
    verbose = verbose
  )
  return(list(score=(encoder%>%predict(X)),
              encoder=encoder,decoder=generator,model=vae,history=history))
}
pca <- function(X,prop=1,ifscale=TRUE){
  if(ifscale){X <- scale(X)[,]}
  m = nrow(X)
  n = ncol(X)
  Xeigen <- svd(X)
  value <- (Xeigen$d)^2/m
  value <- cumsum(value/sum(value))
  score <- X %*% Xeigen$v
  list(score=score,value=value,mat=Xeigen$v)
}

################################################

#data

setwd('/Users/wenrurumon/Documents/uthealth/deconv/lmy/rpack')
EMTAB.eset = readRDS('vignettes/data/EMTABesethealthy.rds')
XinT2D.eset = readRDS('vignettes/data/XinT2Deset.rds')
XinT2D.construct.full = bulk_construct(XinT2D.eset, clusters = 'cellType', samples = 'SubjectName')
XinT2D.construct.full$prop.real = relative.ab(XinT2D.construct.full$num.real, by.col = FALSE)

#MUSIC

bulk.eset = XinT2D.construct.full$Bulk.counts
sc.eset = EMTAB.eset
markers = NULL
clusters = 'cellType' 
samples = 'sampleID'
select.ct = c('alpha', 'beta', 'delta', 'gamma')
cell_size = NULL
ct.cov = FALSE
verbose = FALSE
iter.max = 1000
nu = 1e-04
eps = 0.01
centered = FALSE 
normalize = FALSE

bulk.gene = rownames(bulk.eset)[rowMeans(exprs(bulk.eset)) != 0]
bulk.eset = bulk.eset[bulk.gene, , drop = FALSE]
sc.markers = bulk.gene
sc.basis = music_basis(sc.eset, non.zero = TRUE, markers = sc.markers, clusters = clusters, samples = samples, select.ct = select.ct, cell_size = cell_size, ct.cov = ct.cov, verbose = verbose)
cm.gene = intersect(rownames(sc.basis$Disgn.mtx), bulk.gene)
m.sc = match(cm.gene, rownames(sc.basis$Disgn.mtx))
m.bulk = match(cm.gene, bulk.gene)
D1 = sc.basis$Disgn.mtx[m.sc, ]
M.S = colMeans(sc.basis$S, na.rm = T)
Yjg = relative.ab(exprs(bulk.eset)[m.bulk, ])
N.bulk = ncol(bulk.eset)

Sigma = sc.basis$Sigma[m.sc, ]
valid.ct = (colSums(is.na(Sigma))==0)&(colSums(is.na(D1))==0)&(!is.na(M.S))
D1 = D1[, valid.ct]
M.S = M.S[valid.ct]
Sigma = Sigma[, valid.ct]
Est.prop.allgene = NULL
Est.prop.weighted = NULL
Weight.gene = NULL
r.squared.full = NULL
Var.prop = NULL

for (i in 1:N.bulk) {
  if (sum(Yjg[, i] == 0) > 0) {
    D1.temp = D1[Yjg[, i] != 0, ]
    Yjg.temp = Yjg[Yjg[, i] != 0, i]
    Sigma.temp = Sigma[Yjg[, i] != 0, ]
  }
  else {
    D1.temp = D1
    Yjg.temp = Yjg[, i]
    Sigma.temp = Sigma
  }
  lm.D1.weighted = music.iter(Yjg.temp, D1.temp, M.S, 
                              Sigma.temp, iter.max = iter.max, nu = nu, eps = eps, 
                              centered = centered, normalize = normalize)
  Est.prop.allgene = rbind(Est.prop.allgene, lm.D1.weighted$p.nnls)
  Est.prop.weighted = rbind(Est.prop.weighted, lm.D1.weighted$p.weight)
  weight.gene.temp = rep(NA, nrow(Yjg))
  weight.gene.temp[Yjg[, i] != 0] = lm.D1.weighted$weight.gene
  Weight.gene = cbind(Weight.gene, weight.gene.temp)
  r.squared.full = c(r.squared.full, lm.D1.weighted$R.squared)
  Var.prop = rbind(Var.prop, lm.D1.weighted$var.p)
}
colnames(Est.prop.weighted) = colnames(D1)
rownames(Est.prop.weighted) = colnames(Yjg)
colnames(Est.prop.allgene) = colnames(D1)
rownames(Est.prop.allgene) = colnames(Yjg)
names(r.squared.full) = colnames(Yjg)
colnames(Weight.gene) = colnames(Yjg)
rownames(Weight.gene) = cm.gene
colnames(Var.prop) = colnames(D1)
rownames(Var.prop) = colnames(Yjg)

Est.prop.Xin <- list(Est.prop.weighted = Est.prop.weighted, Est.prop.allgene = Est.prop.allgene, Weight.gene = Weight.gene, r.squared.full = r.squared.full, Var.prop = Var.prop)

#Keras latent

Yk <- minmax(Yjg); dim(Yk)
Dk <- minmax(D1); dim(Dk)
Sk <- minmax(Sigma); dim(Sk)
M.S <- minmax(M.S)

Est.prop.allgene = NULL
Est.prop.weighted = NULL
Weight.gene = NULL
r.squared.full = NULL
Var.prop = NULL

for (i in 1:N.bulk) {
  print(i)
  if (sum(Yk[, i] == 0) > 0) {
    D1.temp = Dk[Yk[, i] != 0, ]
    Yjg.temp = Yk[Yk[, i] != 0, i]
    Sigma.temp = Sk[Yk[, i] != 0, ]
  } else {
    D1.temp = Dk
    Yjg.temp = Yk[, i]
    Sigma.temp = Sk
  }
  lm.D1.weighted = music.iter(Yjg.temp, D1.temp, M.S, 
                              Sigma.temp, iter.max = iter.max, nu = nu, eps = eps, 
                              centered = centered, normalize = normalize)
  Est.prop.allgene = rbind(Est.prop.allgene, lm.D1.weighted$p.nnls)
  Est.prop.weighted = rbind(Est.prop.weighted, lm.D1.weighted$p.weight)
}
colnames(Est.prop.weighted) = colnames(D1)
rownames(Est.prop.weighted) = colnames(Yjg)
colnames(Est.prop.allgene) = colnames(D1)
rownames(Est.prop.allgene) = colnames(Yjg)

#Resulting

rlt <- Eval_multi(prop.real <- data.matrix(XinT2D.construct.full$prop.real),
                  prop.est <- list(data.matrix(Est.prop.Xin$Est.prop.weighted),
                                   data.matrix(Est.prop.Xin$Est.prop.allgene),
                                   data.matrix(Est.prop.weighted)),
                  method.name = c('MuSiC', 'NNLS','model'))
print(rlt)
