param location string
param registryName string

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
    name: registryName
    location: location
    sku: {
        name: 'Basic'
    }
    properties: {
        adminUserEnabled: false
    }
}

output id string = containerRegistry.id
output loginServer string = containerRegistry.properties.loginServer
