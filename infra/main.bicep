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
    // Keep provisioning independent of package pushes. `azd deploy api`
    // replaces this already-existing bootstrap image with the current API image.
    imageName: '${containerRegistry.outputs.loginServer}/movie-recommender:latest'
  }
}

module webApp './components/web-container-app.bicep' = {
  name: 'movie-recommender-web-deployment'
  params: {
    location: location
    environmentName: environmentName
    containerAppEnvironmentId: containerAppEnvironment.outputs.id
    registryName: registryName
    registryLoginServer: containerRegistry.outputs.loginServer
    apiUri: movieRecommenderApp.outputs.uri
    entraApplicationClientId: entraApplicationClientId
    // A public bootstrap image lets `azd provision` create the new frontend
    // Container App before the first private web image has been pushed. The
    // subsequent `azd deploy web` replaces it with the generated frontend.
    imageName: 'nginx:1.27-alpine'
  }
}

output CONTAINER_REGISTRY_LOGIN_SERVER string = containerRegistry.outputs.loginServer
output SERVICE_API_IMAGE_NAME string = '${containerRegistry.outputs.loginServer}/api:latest'
output SERVICE_API_URI string = movieRecommenderApp.outputs.uri
output SERVICE_WEB_IMAGE_NAME string = '${containerRegistry.outputs.loginServer}/web:latest'
output SERVICE_WEB_URI string = webApp.outputs.uri
output POSTGRES_HOST string = postgres.outputs.host
