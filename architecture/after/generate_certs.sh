#!/bin/bash

# Configuration
CA_KEY="certs/ca.key"
CA_CERT="certs/ca.crt"
CA_SERIAL="certs/ca.srl"

TEST_CLIENT_KEY="certs/testclient.key"
TEST_CLIENT_CERT="certs/testclient.crt"

MOSQUITTO_KEY="certs/mosquitto.key"
MOSQUITTO_CERT="certs/mosquitto.crt"

NODERED_KEY="certs/nodered.key"
NODERED_CERT="certs/nodered.crt"

POSTGRES_KEY="certs/postgres.key"
POSTGRES_CERT="certs/postgres.crt"

GRAFANA_KEY="certs/grafana.key"
GRAFANA_CERT="certs/grafana.crt"

METER_115138_KEY="certs/meter-115138.key"
METER_115138_CERT="certs/meter-115138.crt"

METER_15805_KEY="certs/meter-15805.key"
METER_15805_CERT="certs/meter-15805.crt"

METER_50176_KEY="certs/meter-50176.key"
METER_50176_CERT="certs/meter-50176.crt"

METER_7001_KEY="certs/meter-7001.key"
METER_7001_CERT="certs/meter-7001.crt"

METER_18052_KEY="certs/meter-18052.key"
METER_18052_CERT="certs/meter-18052.crt"


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

# MOSQUITTO
echo $'\nGenerating Mosquitto Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $MOSQUITTO_KEY -out $MOSQUITTO_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=mosquitto" \
  -reqexts SAN \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=IP:127.0.0.1,DNS:localhost")) \
  # -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $MOSQUITTO_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $MOSQUITTO_CERT -days 365
cp -rf $CA_CERT ./config/mosquitto/certs/ca.crt
cp -rf $MOSQUITTO_KEY ./config/mosquitto/certs/mosquitto.key
cp -rf $MOSQUITTO_CERT ./config/mosquitto/certs/mosquitto.crt


# NODE-RED
echo $'\nGenerating Node-RED Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $NODERED_KEY -out $NODERED_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=localhost" \
  -reqexts SAN \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=IP:127.0.0.1,DNS:localhost")) \
  # -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $NODERED_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $NODERED_CERT -days 365

# POSTGRES
echo $'\nGenerating Postgres Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $POSTGRES_KEY -out $POSTGRES_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=postgres" \
  -reqexts SAN \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=IP:127.0.0.1,DNS:localhost")) \
  # -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $POSTGRES_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $POSTGRES_CERT -days 365

# GRAFANA
echo $'\nGenerating Grafana Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $GRAFANA_KEY -out $GRAFANA_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=grafana" \
  -reqexts SAN \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=IP:127.0.0.1,DNS:localhost")) \
  # -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $GRAFANA_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL  -out $GRAFANA_CERT -days 365

# METER 115138
echo $'\nGenerating Meter-115138 Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_115138_KEY -out $METER_115138_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-115138" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_115138_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_115138_CERT -days 365
cp -rf $CA_CERT ./config/meter-115138/ca.crt
cp -rf $METER_115138_KEY ./config/meter-115138/meter-115138.key
cp -rf $METER_115138_CERT ./config/meter-115138/meter-115138.crt

# METER 15805
echo $'\nGenerating Meter-15805 Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_15805_KEY -out $METER_15805_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-15805" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_15805_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_15805_CERT -days 365
cp -rf $CA_CERT ./config/meter-15805/ca.crt
cp -rf $METER_15805_KEY ./config/meter-15805/meter-15805.key
cp -rf $METER_15805_CERT ./config/meter-15805/meter-15805.crt

# METER 50176
echo $'\nGenerating Meter-50176 Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_50176_KEY -out $METER_50176_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-50176" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_50176_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_50176_CERT -days 365
cp -rf $CA_CERT ./config/meter-50176/ca.crt
cp -rf $METER_50176_KEY ./config/meter-50176/meter-50176.key
cp -rf $METER_50176_CERT ./config/meter-50176/meter-50176.crt

# METER 7001
echo $'\nGenerating Meter-7001 Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_7001_KEY -out $METER_7001_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-7001" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_7001_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_7001_CERT -days 365
cp -rf $CA_CERT ./config/meter-7001/ca.crt
cp -rf $METER_7001_KEY ./config/meter-7001/meter-7001.key
cp -rf $METER_7001_CERT ./config/meter-7001/meter-7001.crt

# METER 18052
echo $'\nGenerating Meter-18052 Certificate...'
openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout $METER_18052_KEY -out $METER_18052_CERT \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=meter-18052" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost"
openssl x509 -req -in $METER_18052_CERT -CA $CA_CERT -CAkey $CA_KEY -CAserial $CA_SERIAL -out $METER_18052_CERT -days 365
cp -rf $CA_CERT ./config/meter-18052/ca.crt
cp -rf $METER_18052_KEY ./config/meter-18052/meter-18052.key
cp -rf $METER_18052_CERT ./config/meter-18052/meter-18052.crt

echo $'\nCertificates regenerated for localhost.'
