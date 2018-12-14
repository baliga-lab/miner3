setwd("~/Desktop/miner/src")

#install.packages('GeneCycle')
#install.packages('GeneNet')
#install.packages('ggm')
#install.packages('getopt')

suppressMessages(library(methods))
suppressMessages(library(MASS)) # standard, no need to install
suppressMessages(library(class))	# standard, no need to install
suppressMessages(library(cluster))
suppressMessages(library(getopt))
suppressMessages(library(doParallel))

outputFolder <- "causal"
numCores <- 5
sigRegFile <- "../results/regStratAll.csv"
bcTfFile <- "../results/bcTfIncidence.csv" #bicluster rows, tfs columns
eigengeneFile <- "../results/eigengenes.csv"
regExpFile <- "../results/tfExpression.csv"
mutationsFile <- "../results/filteredMutations.csv"

#load mutations incidence matrix
mutations <- read.csv(file=mutationsFile, as.is=T, header=T, row.names=1 )

#load significant regulators incidence matrix
sigRegIncDf <- read.csv(file=sigRegFile, as.is=T, header=T, row.names=1 )
sigRegFC <- list()
for(mut1 in colnames(sigRegIncDf)) {
  sigRegFC[mut1] <- list(rownames(sigRegIncDf)[which(sigRegIncDf[,mut1]==1)])
}
  
# load regulator expression file
regExp <- read.csv(file=regExpFile, as.is=T, header=T, row.names=1 )

# Load bicluster eigengenes
cat('\nLoading bicluster eigengene...')
be1 <- read.csv(file=eigengeneFile, row.names=1, header=T)
rownames(be1) <- paste('bic',rownames(be1),sep='_')

#create reference list of biclusters to test for a given tf
bcTfDf <- read.csv(file=bcTfFile, as.is=T, header=T, row.names=1 )
rownames(bcTfDf) <- paste('bic',rownames(bcTfDf),sep='_')
tfToBc <- list()
for(tf in colnames(bcTfDf)) {
  tfToBc[tf] <- list(rownames(bcTfDf)[which(bcTfDf[,tf]==1)])
}

ol1 <- intersect(intersect(colnames(be1),colnames(mutations)),colnames(regExp))
cat(paste('\nsom_muts = ',nrow(mutations),'; regExp = ',nrow(regExp),'; be1 = ',nrow(be1),sep=''))
d2 <- rbind(as.matrix(mutations[,ol1]), as.matrix(regExp[,ol1]), as.matrix(be1[,ol1]))
d3 <- t(na.omit(t(d2)))
#  

########################
## Causality analysis ##
########################
## Use for filtering:
#  1. Signficant differntial expression of regulator between wt and mutant (FC <= 0.8 or FC >= 1.25, and T-test p-value <= 0.05)
cat('\nRunning NEO...')
source('neoSourceCode.R')
registerDoParallel(cores=numCores)
dir.create(paste('../results/',outputFolder,sep=''), showWarnings=F)
foreach(mut1=names(sigRegFC)) %dopar% {
  # Make a place to store out the data from the analysis
  mut2 = mut1
  if(nchar(mut2)>75) {
    mut2 = substr(mut2,1,75)
  }
  dir.create(paste('../results/',outputFolder,'/causal_', mut2, sep=''))
  
  # Change the names to be compatible with NEO
  print(paste('Starting ',mut1,'...',sep=''))
  for(reg1 in sigRegFC[[mut1]]) {
    # Make the data matrix with all genes strsplit(mut1)[[1]][1]
    #d3 = t(na.omit(t(d2[c(mut1,reg1,rownames(be1)),]))) #test against all biclusters
    if (reg1 %in% names(tfToBc)) {
      if (mut1 %in% rownames(d2)) {
        d3 = t(na.omit(t(d2[c(mut1,reg1,tfToBc[[reg1]]),])))
      } else {
        next
      }
    } else {
      next
    }
    dMut1 = matrix(data=as.numeric(d3),nrow=dim(d3)[1],ncol=dim(d3)[2],byrow=F,dimnames=dimnames(d3))
    print(paste('  Starting ',mut1,' vs. ', reg1,' testing ', length(rownames(be1)), ' biclusters...', sep=''))
    sm1 = try(single.marker.analysis(t(dMut1),1,2,3:length(rownames(dMut1))),silent=TRUE)
    if (!(class(sm1)=='try-error')) {
      write.csv(sm1[order(sm1[,6],decreasing=T),1:7],paste('../results/',outputFolder,'/causal_', mut2, '/sm.nonsilent_somatic.',mut2,'_',reg1,'.csv',sep=''))
      print(paste('Finished ',reg1,'.',sep=''))
    } else { 
      print(paste('  Error ',mut1,'.',sep=''))
    }
  }
  print(paste('Finished ',mut1,'.',sep=''))
}
cat('\nDone!')