#!/bin/bash
set -e

# Update file permissions of certificates
chmod 600 /var/lib/postgresql/postgres.key /var/lib/postgresql/postgres.crt /var/lib/postgresql/ca.crt
chown postgres:postgres /var/lib/postgresql/postgres.key /var/lib/postgresql/postgres.crt /var/lib/postgresql/ca.crt

# Run the base entrypoint
docker-entrypoint.sh postgres 
