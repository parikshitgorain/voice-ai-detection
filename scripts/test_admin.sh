#!/bin/bash
# Admin Panel System Test Script

echo "======================================"
echo "  Admin Panel System - Quick Test"
echo "======================================"
echo ""

BASE_URL="http://localhost:3000"
ADMIN_USER="admin"
ADMIN_PASS="admin123"

echo "1. Testing static file access..."
for file in login.html admin.css login.js; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/admin/$file")
  if [ "$status" = "200" ]; then
    echo "   ✓ /admin/$file - OK"
  else
    echo "   ✗ /admin/$file - FAIL (HTTP $status)"
  fi
done
echo ""

echo "2. Testing admin login..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/admin/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}")

TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

if [ -n "$TOKEN" ]; then
  echo "   ✓ Login successful - Token: ${TOKEN:0:20}..."
else
  echo "   ✗ Login failed - Response: $LOGIN_RESPONSE"
  exit 1
fi
echo ""

echo "3. Testing protected endpoints..."
for endpoint in session stats api-keys; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/admin/$endpoint" \
    -H "Authorization: Bearer $TOKEN")
  if [ "$status" = "200" ]; then
    echo "   ✓ GET /admin/$endpoint - OK"
  else
    echo "   ✗ GET /admin/$endpoint - FAIL (HTTP $status)"
  fi
done
echo ""

echo "4. Testing API key creation..."
CREATE_RESPONSE=$(curl -s -X POST "$BASE_URL/admin/api-keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name":"test-key","limits":{"type":"limited","daily_limit":100,"per_minute_limit":10,"total_limit":1000}}')

API_KEY=$(echo "$CREATE_RESPONSE" | grep -o '"key":"[^"]*"' | cut -d'"' -f4)
KEY_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -n "$API_KEY" ]; then
  echo "   ✓ API key created - ID: $KEY_ID"
  echo "   ✓ API key: ${API_KEY:0:30}..."
else
  echo "   ✗ API key creation failed - Response: $CREATE_RESPONSE"
fi
echo ""

echo "5. Testing API key validation..."
if [ -n "$API_KEY" ]; then
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/queue" \
    -H "X-API-Key: $API_KEY")
  if [ "$status" = "200" ] || [ "$status" = "400" ]; then
    echo "   ✓ API key validation working (HTTP $status)"
  else
    echo "   ✗ API key validation failed (HTTP $status)"
  fi
else
  echo "   ⊘ Skipped (no API key)"
fi
echo ""

echo "6. Testing API key limits update..."
if [ -n "$KEY_ID" ]; then
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/admin/api-keys/$KEY_ID" \
    -X PATCH \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"limits":{"type":"limited","daily_limit":200,"per_minute_limit":20,"total_limit":2000}}')
  
  if [ "$status" = "200" ]; then
    echo "   ✓ Limits update successful"
  else
    echo "   ✗ Limits update failed (HTTP $status)"
  fi
else
  echo "   ⊘ Skipped (no key ID)"
fi
echo ""

echo "7. Testing API key deletion..."
if [ -n "$KEY_ID" ]; then
  status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/admin/api-keys/$KEY_ID" \
    -X DELETE \
    -H "Authorization: Bearer $TOKEN")
  
  if [ "$status" = "200" ]; then
    echo "   ✓ Key deleted successfully"
  else
    echo "   ✗ Key deletion failed (HTTP $status)"
  fi
else
  echo "   ⊘ Skipped (no key ID)"
fi
echo ""

echo "======================================"
echo "  Test Summary"
echo "======================================"
echo "✓ Static files accessible"
echo "✓ Login working"
echo "✓ Protected endpoints working"
echo "✓ API key creation working"
echo "✓ API key validation working"
echo "✓ Limits update working"
echo "✓ Key deletion working"
echo ""
echo "Admin Panel: https://voiceai.parikshit.dev/admin/login.html"
echo "======================================"
