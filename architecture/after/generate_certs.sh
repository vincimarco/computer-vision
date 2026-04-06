#!/bin/bash
set -euo pipefail

mkdir -p certs

CA_KEY="certs/ca.key"
CA_CERT="certs/ca.crt"
CA_SERIAL="certs/ca.srl"

echo "00" > "$CA_SERIAL"

echo "Generating Root CA..."
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout "$CA_KEY" -out "$CA_CERT" \
  -subj "/C=XX/ST=State/L=City/O=Academic Project/OU=Security Team/CN=Root Authority"

generate_cert() {
  local key="$1"
  local cert="$2"
  local config="$3"
  shift 3
  local extra_args=("$@")

  openssl req -newkey rsa:4096 -sha256 -days 365 -nodes \
    -keyout "$key" -out "$cert" \
    -config "$config" \
    "${extra_args[@]}"

  openssl x509 -req -in "$cert" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAserial "$CA_SERIAL" \
    -out "$cert" -days 365 \
    -extensions req_ext -extfile $config
}

TEST_CLIENT_KEY="certs/testclient.key"
TEST_CLIENT_CERT="certs/testclient.crt"

echo $'\nGenerating Test Client Certificate...'
generate_cert "$TEST_CLIENT_KEY" "$TEST_CLIENT_CERT" "config/testclient.conf"

services=(mosquitto nodered postgres grafana modello-cnn3d)
for svc in "${services[@]}"; do
  key="certs/${svc}.key"
  cert="certs/${svc}.crt"

  echo $'\nGenerating '"$svc"' Certificate...'
  generate_cert "$key" "$cert" "config/$svc.conf" \

  mkdir -p "./config/${svc}/certs"
  sudo cp -f "$CA_CERT" "$key" "$cert" "./config/${svc}/certs/"
done

meters=(115138 15805 50176 7001 18052)
for meter in "${meters[@]}"; do
  key="certs/meter-${meter}.key"
  cert="certs/meter-${meter}.crt"

  echo $'\nGenerating Meter-'"$meter"' Certificate...'
  generate_cert "$key" "$cert" "config/meter-$meter.conf" \

  mkdir -p "./config/meter-${meter}"
  sudo cp -f "$CA_CERT" "$key" "$cert" "./config/meter-${meter}/"
done

echo $'\nCertificates regenerated for localhost.'
