curl -sS -X POST "http://localhost:5000/env" \
    -H  "accept: application/json" \
    -H  "Content-Type: application/json" \
    -d '{
"type":"hello",
"message":"world!"
}' \
| jq .
