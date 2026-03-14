#!/bin/bash

# Configuration
CA_KEY="certs/ca.key"
CA_CERT="certs/ca.crt"
CA_SERIAL="certs/ca.srl"

TEST_CLIENT_KEY="certs/testclient.key"
TEST_CLIENT_CERT="certs/testclient.crt"

METER_KEY="certs/meter.key"
METER_CERT="certs/meter.crt"

MOSQUITTO_KEY="certs/mosquitto.key"
MOSQUITTO_CERT="certs/mosquitto.crt"

# CLIENT_KEY="certs/client.key"
# CLIENT_CERT="certs/client.crt"
# SERVER_KEY="certs/server.key"
# SERVER_CERT="certs/server.crt"

echo "00" > $CA_SERIAL

# ROOT CA
echo "Generating Root CA..."
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $CA_KEY -out $CA_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=Root Authority"

# TEST CLIENT
echo $'\nGenerating Test Client Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $TEST_CLIENT_KEY -out $TEST_CLIENT_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=client" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $TEST_CLIENT_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $TEST_CLIENT_CERT -days 365


# METER
echo $'\nGenerating Meter Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_KEY -out $METER_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-sn-12345678" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_CERT -days 365

# MOSQUITTO
echo $'\nGenerating Mosquitto Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $MOSQUITTO_KEY -out $MOSQUITTO_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=localhost" \
  -reqexts SAN \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=IP:127.0.0.1,DNS:localhost")) \
  # -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $MOSQUITTO_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $MOSQUITTO_CERT -days 365

# # 2. Generate Client Certificate
# echo "Generating Client Cert..."
# openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
#   -keyout $CLIENT_KEY -out $CLIENT_CERT \
#   -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=NodeREDClient"
# openssl x509 -req -in $CLIENT_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $CLIENT_CERT -days 365

# # 3. Generate Server Certificate
# echo "Generating Server Cert for Localhost..."
# openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
#   -keyout $SERVER_KEY -out $SERVER_CERT \
#   -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=localhost" \
#   -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"

# # Sign the server cert with the Root CA
# openssl x509 -req -in $SERVER_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial  -out $SERVER_CERT -days 365

echo $'\nCertificates regenerated for localhost.'
