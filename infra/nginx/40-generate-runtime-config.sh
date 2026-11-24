#!/bin/sh
set -eu

envsubst '${API_BASE_URL} ${ENTRA_CLIENT_ID} ${ENTRA_TENANT_ID} ${API_SCOPE}' \
  < /usr/share/nginx/config.js.template \
  > /usr/share/nginx/html/config.js
