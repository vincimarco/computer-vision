#!/bin/bash

# Configuration
CA_KEY="certs/ca.key"
CA_CERT="certs/ca.crt"
CLIENT_KEY="certs/client.key"
CLIENT_CERT="certs/client.crt"
SERVER_KEY="certs/server.key"
SERVER_CERT="certs/server.crt"

# 1. Generate Root CA
echo "Generating Root CA..."
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $CA_KEY -out $CA_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=Root Authority"

# 2. Generate Client Certificate
echo "Generating Client Cert..."
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $CLIENT_KEY -out $CLIENT_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=NodeREDClient"
openssl x509 -req -in $CLIENT_CERT -CA $CA_CERT -CAkey $CA_KEY -CAcreateserial -out $CLIENT_CERT -days 365

# 3. Generate Server Certificate
echo "Generating Server Cert for Localhost..."
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $SERVER_KEY -out $SERVER_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=localhost" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"

# Sign the server cert with the Root CA
openssl x509 -req -in $SERVER_CERT -CA $CA_CERT -CAkey $CA_KEY -CAcreateserial -out $SERVER_CERT -days 365

echo "Certificates regenerated for localhost."
