param location string = resourceGroup().location
param environmentName string
param storageAccountName string
param registryName string

module storage './components/resourcegroup.bicep' = {
    name: 'storage-deployment'
    params: {
        location: location
        storageAccountName: storageAccountName
    }
}

module containerRegistry './components/container-registry.bicep' = {
    name: 'registry-deployment'
    params: {
        location: location
        registryName: registryName
    }
}

module containerAppEnvironment './components/container-app-environment.bicep' = {
  name: 'container-app-environment-deployment'
  params: {
    location: location
    environmentName: environmentName
  }
}

module movieRecommenderApp './components/container-app.bicep' = {
  name: 'movie-recommender-deployment'
  params: {
    location: location
    environmentName: environmentName
    containerAppEnvironmentId: containerAppEnvironment.outputs.id
    registryName: registryName
    registryLoginServer: containerRegistry.outputs.loginServer
  }
}

output CONTAINER_REGISTRY_LOGIN_SERVER string = containerRegistry.outputs.loginServer
output SERVICE_MOVIE_RECOMMENDER_IMAGE_NAME string = '${containerRegistry.outputs.loginServer}/movie-recommender:latest'
output SERVICE_MOVIE_RECOMMENDER_URI string = movieRecommenderApp.outputs.uri
