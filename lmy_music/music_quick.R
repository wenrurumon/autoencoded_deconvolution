
music_quick <- function(bulk,ref,markers=NULL,select.ct=NULL,verbose=T,iter.max=1000,nu=0.0001,eps=0.01,centered=FALSE,normalize=FALSE){
  bulk <- apply(bulk,2,minmax)
  ref <- apply(ref,2,minmax)
  if(is.null(rownames(bulk))){
    rownames(bulk) <- rownames(ref) <- paste0('g',1:nrow(bulk))  
  }
  M.theta <- Sigma <- ref
  Sigma[,] <- 1
  S <- rbind(colMeans(M.theta))
  S[S == 0] = NA
  M.S = colMeans(S, na.rm = TRUE)
  D <- t(t(M.theta)*M.S)
  gene <- rownames(ref)[which(rownames(ref)%in%rownames(bulk))]
  D1 <- D[match(gene,rownames(D)),,drop=F]
  M.S <- colMeans(S,na.rm=T)
  Yjg <- relative.ab(bulk[match(gene,rownames(bulk)),,drop=F])
  N.bulk <- ncol(Yjg)
  Sigma <- Sigma[match(gene,rownames(Sigma)),,drop=F]
  Est.prop.allgene = NULL
  Est.prop.weighted = NULL
  Weight.gene = NULL
  r.squared.full = NULL
  Var.prop = NULL
  for(i in 1:N.bulk){
    if(verbose){print(i)}
    if(sum(Yjg[, i] == 0) > 0){
      D1.temp = D1[Yjg[, i]!=0, ];
      Yjg.temp = Yjg[Yjg[, i]!=0, i];
      Sigma.temp = Sigma[Yjg[,i]!=0, ];
    }else{
      D1.temp = D1;
      Yjg.temp = Yjg[, i];
      Sigma.temp = Sigma;
    }
    lm.D1.weighted = music.iter(Yjg.temp, D1.temp, M.S, Sigma.temp, iter.max = iter.max, nu = nu, eps = eps, centered = centered, normalize =normalize)
    Est.prop.allgene = rbind(Est.prop.allgene, lm.D1.weighted$p.nnls)
    Est.prop.weighted = rbind(Est.prop.weighted, lm.D1.weighted$p.weight)
    weight.gene.temp = rep(NA, nrow(Yjg)); weight.gene.temp[Yjg[,i]!=0] = lm.D1.weighted$weight.gene;
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
  rownames(Weight.gene) = gene
  colnames(Var.prop) = colnames(D1)
  rownames(Var.prop) = colnames(Yjg)
  return(list(Est.prop.weighted = Est.prop.weighted, Est.prop.allgene = Est.prop.allgene,
        Weight.gene = Weight.gene, r.squared.full = r.squared.full, Var.prop = Var.prop))
}





