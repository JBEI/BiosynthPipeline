from retrotide import structureDB, designPKS
import retrotide
testcluster = retrotide.Cluster(modules=[retrotide.allStarterTypes[0],
                                         retrotide.allModuleTypes[0],
                                         retrotide.allModuleTypes[1]])
testcluster.computeProduct(structureDB)
