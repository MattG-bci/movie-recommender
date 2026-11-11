param location string
param environmentName string

var resourceToken = toLower(uniqueString(resourceGroup().id, environmentName))
var virtualNetworkName = 'vnet-${environmentName}-${resourceToken}'
var privateDnsZoneName = 'movie-recommender.postgres.database.azure.com'

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: virtualNetworkName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.20.0.0/16'
      ]
    }
    // Use Azure-provided DNS. Private PostgreSQL records are resolved through
    // the linked Private DNS zone below, so a DNS Private Resolver is not needed.
    dhcpOptions: {
      dnsServers: []
    }
  }
}

// Manage project subnets as child resources rather than an inline VNet subnet
// collection. This preserves unrelated subnets (such as an existing GatewaySubnet)
// during incremental deployments.
resource containerAppsSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' = {
  parent: virtualNetwork
  name: 'container-apps'
  properties: {
    addressPrefix: '10.20.0.0/27'
    delegations: [
      {
        name: 'container-apps-delegation'
        properties: {
          serviceName: 'Microsoft.App/environments'
        }
      }
    ]
  }
}

resource postgresSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' = {
  parent: virtualNetwork
  name: 'postgres'
  properties: {
    addressPrefix: '10.20.1.0/28'
    delegations: [
      {
        name: 'postgres-delegation'
        properties: {
          serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
        }
      }
    ]
  }
}

resource privateDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: privateDnsZoneName
  location: 'global'
}

resource privateDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: privateDnsZone
  name: 'vnet-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

output containerAppsSubnetId string = containerAppsSubnet.id
output postgresSubnetId string = postgresSubnet.id
output privateDnsZoneId string = privateDnsZone.id
