param location string
param environmentName string
param containerAppEnvironmentId string
param registryName string
param registryLoginServer string
param databaseHost string
param databaseName string
param databaseUser string
@secure()
param databasePassword string

var serviceName = 'movie-recommender'
var appName = '${serviceName}-${toLower(uniqueString(resourceGroup().id))}'
var registryIdentityName = 'id-${serviceName}-${toLower(uniqueString(resourceGroup().id))}'

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource registryIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: registryIdentityName
  location: location
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, registryIdentity.id, 'AcrPull')
  scope: registry
  properties: {
    principalId: registryIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  dependsOn: [
    acrPull
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${registryIdentity.id}': {}
    }
  }
  tags: {
    'azd-service-name': serviceName
    'azd-env-name': environmentName
  }
  properties: {
    managedEnvironmentId: containerAppEnvironmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: [
        {
          name: 'db-password'
          value: databasePassword
        }
      ]
      registries: [
        {
          server: registryLoginServer
          identity: registryIdentity.id
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
          image: 'mcr.microsoft.com/dotnet/samples:aspnetapp'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'DB_HOST'
              value: databaseHost
            }
            {
              name: 'DB_PORT'
              value: '5432'
            }
            {
              name: 'DB_NAME'
              value: databaseName
            }
            {
              name: 'DB_USER'
              value: databaseUser
            }
            {
              name: 'DB_PASS'
              secretRef: 'db-password'
            }
          ]
        }
      ]
    }
  }
}

output id string = app.id
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
