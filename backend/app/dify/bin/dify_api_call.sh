
api_key="app-H9L7z3QJlU5zn0oA0nXSg9AI"
# location="London"
# location="Tokyo"
location="Osaka"

curl -X POST -sSL 'http://192.168.1.11/v1/workflows/run' \
--header "Authorization: Bearer $api_key" \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {"location": "'$location'"},
    "response_mode": "blocking",
    "user": "abc-123"
}'

#    "response_mode": "streaming",
