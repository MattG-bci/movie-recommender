param location string
param environmentName string
param containerAppEnvironmentId string
param registryName string
param registryLoginServer string

var serviceName = 'movie-recommender'
var appName = '${serviceName}-${toLower(uniqueString(resourceGroup().id))}'

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    'azd-service-name': serviceName
    'azd-env-name': environmentName
  }
  properties: {
    managedEnvironmentId: containerAppEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: registryLoginServer
          identity: 'system'
        }
      ]
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: serviceName
          // This public image lets the app provision before azd has pushed the
          // application's first image to the private registry. `azd deploy`
          // replaces it with SERVICE_MOVIE_RECOMMENDER_IMAGE_NAME.
          image: 'mcr.microsoft.com/k8se/quickstart:latest'
          resources: {
            cpu: '0.5'
            memory: '1Gi'
          }
        }
      ]
    }
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, app.id, 'AcrPull')
  scope: registry
  properties: {
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
  }
}

output id string = app.id
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
