#!/bin/bash
# Health monitoring script for voice-ai-detection
# Run this periodically via cron or systemd timer

LOG_FILE="/var/www/voice-ai-detection/backend/logs/health-monitor.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Check if PM2 process is running
pm2 status voice-ai-detection | grep -q "online"
if [ $? -ne 0 ]; then
    echo "[$TIMESTAMP] ❌ Server is DOWN! Attempting restart..." >> "$LOG_FILE"
    cd /var/www/voice-ai-detection/backend
    pm2 restart voice-ai-detection
    sleep 5
    pm2 status voice-ai-detection | grep -q "online"
    if [ $? -eq 0 ]; then
        echo "[$TIMESTAMP] ✅ Server restarted successfully" >> "$LOG_FILE"
    else
        echo "[$TIMESTAMP] ❌ Failed to restart server!" >> "$LOG_FILE"
        # Send alert (add your notification logic here)
    fi
else
    echo "[$TIMESTAMP] ✅ Server is running" >> "$LOG_FILE"
fi

# Check health endpoint
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/health)
if [ "$HEALTH_CHECK" != "200" ]; then
    echo "[$TIMESTAMP] ⚠️  Health check returned: $HEALTH_CHECK" >> "$LOG_FILE"
else
    echo "[$TIMESTAMP] ✅ Health endpoint OK" >> "$LOG_FILE"
fi

# Check memory usage
MEM_USAGE=$(pm2 jlist | jq -r '.[] | select(.name=="voice-ai-detection") | .monit.memory' 2>/dev/null)
if [ -n "$MEM_USAGE" ]; then
    MEM_MB=$((MEM_USAGE / 1024 / 1024))
    echo "[$TIMESTAMP] 📊 Memory usage: ${MEM_MB}MB" >> "$LOG_FILE"
    
    # Alert if memory is too high (>900MB)
    if [ $MEM_MB -gt 900 ]; then
        echo "[$TIMESTAMP] ⚠️  High memory usage detected! Restarting..." >> "$LOG_FILE"
        cd /var/www/voice-ai-detection/backend
        pm2 restart voice-ai-detection
    fi
fi

# Keep log file size under control (keep last 500 lines)
tail -n 500 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
