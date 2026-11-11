param location string = resourceGroup().location
param environmentName string
param storageAccountName string
param registryName string
@secure()
param databaseAdministratorPassword string
@description('Application (client) ID of the single-tenant Microsoft Entra app registration that protects the API.')
param entraApplicationClientId string
@description('Microsoft Entra object IDs permitted to call the API. Include the developer/test user object ID.')
param entraAllowedPrincipalIds array
@description('Fully qualified image name published by azd to Azure Container Registry.')
param movieRecommenderImageName string = ''

var resourceToken = toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))
var databaseServerName = 'pg${resourceToken}'
var databaseName = 'movie_recommender'
var databaseAdministratorLogin = 'movieadmin'

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

module network './components/network.bicep' = {
  name: 'network-deployment'
  params: {
    location: location
    environmentName: environmentName
  }
}

module containerAppEnvironment './components/container-app-environment.bicep' = {
  name: 'container-app-environment-deployment'
  params: {
    location: location
    environmentName: environmentName
    infrastructureSubnetId: network.outputs.containerAppsSubnetId
  }
}

module postgres './components/postgres.bicep' = {
  name: 'postgres-deployment'
  params: {
    location: location
    serverName: databaseServerName
    databaseName: databaseName
    administratorLogin: databaseAdministratorLogin
    administratorLoginPassword: databaseAdministratorPassword
    delegatedSubnetId: network.outputs.postgresSubnetId
    privateDnsZoneId: network.outputs.privateDnsZoneId
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
    databaseHost: postgres.outputs.host
    databaseName: postgres.outputs.database
    databaseUser: postgres.outputs.administratorLogin
    databasePassword: databaseAdministratorPassword
    entraApplicationClientId: entraApplicationClientId
    entraAllowedPrincipalIds: entraAllowedPrincipalIds
    imageName: empty(movieRecommenderImageName)
      ? '${containerRegistry.outputs.loginServer}/movie-recommender:latest'
      : movieRecommenderImageName
  }
}

output CONTAINER_REGISTRY_LOGIN_SERVER string = containerRegistry.outputs.loginServer
output SERVICE_MOVIE_RECOMMENDER_IMAGE_NAME string = '${containerRegistry.outputs.loginServer}/movie-recommender:latest'
output SERVICE_MOVIE_RECOMMENDER_URI string = movieRecommenderApp.outputs.uri
output POSTGRES_HOST string = postgres.outputs.host
