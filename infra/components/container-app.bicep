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
@description('Application (client) ID of the Microsoft Entra app registration representing this API.')
param entraApplicationClientId string
@description('Microsoft Entra user or application object IDs allowed to access this API.')
param entraAllowedPrincipalIds array
@description('Fully qualified image reference hosted in the project Azure Container Registry.')
param imageName string

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
          image: imageName
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

// The API remains publicly addressable so laptops and iPhones can reach it without
// a VPN. Container Apps performs Entra authentication before requests reach the
// application container; callers must also be on this explicit object-ID allowlist.
resource authConfig 'Microsoft.App/containerApps/authConfigs@2024-03-01' = {
  parent: app
  name: 'current'
  properties: {
    platform: {
      enabled: true
      runtimeVersion: '~1'
    }
    globalValidation: {
      unauthenticatedClientAction: 'Return401'
    }
    httpSettings: {
      requireHttps: true
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: entraApplicationClientId
          openIdIssuer: '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'
        }
        validation: {
          // The client ID is the standard v2 access-token audience. The API URI
          // supports clients configured to request the default exposed API scope.
          allowedAudiences: [
            entraApplicationClientId
            'api://${entraApplicationClientId}'
          ]
          defaultAuthorizationPolicy: {
            allowedPrincipals: {
              identities: entraAllowedPrincipalIds
            }
          }
        }
      }
    }
  }
}

output id string = app.id
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
