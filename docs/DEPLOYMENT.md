# Deployment Guide

## Process Management: systemd

This project uses **systemd** for production process management instead of PM2.

### Why systemd?

✓ **Built-in** - No extra dependencies, comes with Linux
✓ **Auto-restart** - Automatically restarts on crash
✓ **Boot startup** - Starts automatically when server boots
✓ **Logging** - Integrated with system journal
✓ **Security** - Better security isolation
✓ **Resource limits** - Built-in memory and CPU limits
✓ **Production-ready** - Industry standard for Linux services

### How It Works

When your server crashes or stops:
1. systemd detects the process died
2. Waits 5 seconds (RestartSec=5)
3. Automatically restarts the service
4. Logs the restart event
5. Continues monitoring

## Quick Setup

### 1. Install the Service

```bash
# Copy service file
sudo cp voice-ai-detection.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable voice-ai-detection

# Start the service
sudo systemctl start voice-ai-detection
```

### 2. Verify It's Running

```bash
sudo systemctl status voice-ai-detection
```

Expected output:
```
● voice-ai-detection.service - Voice AI Detection API Server
   Loaded: loaded (/etc/systemd/system/voice-ai-detection.service; enabled)
   Active: active (running) since Mon 2024-01-15 10:30:00 UTC; 5min ago
   Main PID: 12345 (node)
   Tasks: 11
   Memory: 150.0M
   CGroup: /system.slice/voice-ai-detection.service
           └─12345 /usr/bin/node server.js
```

## Service Management

### View Logs (Real-time)
```bash
sudo journalctl -u voice-ai-detection -f
```

### View Recent Logs
```bash
sudo journalctl -u voice-ai-detection -n 100
```

### View Logs Since Boot
```bash
sudo journalctl -u voice-ai-detection -b
```

### Restart Service
```bash
sudo systemctl restart voice-ai-detection
```

### Stop Service
```bash
sudo systemctl stop voice-ai-detection
```

### Check Status
```bash
sudo systemctl status voice-ai-detection
```

### Disable Auto-start
```bash
sudo systemctl disable voice-ai-detection
```

## Testing Auto-Restart

### Test 1: Kill the Process
```bash
# Find the process ID
sudo systemctl status voice-ai-detection | grep "Main PID"

# Kill it
sudo kill -9 <PID>

# Wait 5 seconds, then check status
sleep 5
sudo systemctl status voice-ai-detection
```

Result: Service should be running again with a new PID.

### Test 2: Simulate Crash
```bash
# Stop the service
sudo systemctl stop voice-ai-detection

# Start it
sudo systemctl start voice-ai-detection

# Watch logs
sudo journalctl -u voice-ai-detection -f
```

## Configuration

The service file is located at: `/etc/systemd/system/voice-ai-detection.service`

Key settings:
```ini
Restart=always          # Always restart on any exit
RestartSec=5           # Wait 5 seconds before restart
MemoryMax=2G           # Limit memory to 2GB
LimitNOFILE=65536      # Max open files
```

### Modify Settings

1. Edit the service file:
```bash
sudo nano /etc/systemd/system/voice-ai-detection.service
```

2. Reload systemd:
```bash
sudo systemctl daemon-reload
```

3. Restart service:
```bash
sudo systemctl restart voice-ai-detection
```

## Monitoring

### Check Memory Usage
```bash
systemctl status voice-ai-detection | grep Memory
```

### Check Uptime
```bash
systemctl status voice-ai-detection | grep Active
```

### Check Restart Count
```bash
sudo journalctl -u voice-ai-detection | grep "Started Voice AI Detection"
```

## Troubleshooting

### Service Won't Start

Check logs:
```bash
sudo journalctl -u voice-ai-detection -n 50
```

Common issues:
- Missing .env file
- Wrong file permissions
- Port already in use
- Missing dependencies

### Service Keeps Restarting

Check logs for errors:
```bash
sudo journalctl -u voice-ai-detection -f
```

Check if port is available:
```bash
sudo netstat -tlnp | grep 3000
```

### High Memory Usage

Check current usage:
```bash
systemctl status voice-ai-detection
```

Adjust memory limit in service file:
```ini
MemoryMax=4G  # Increase to 4GB
```

## Comparison: systemd vs PM2

| Feature | systemd | PM2 |
|---------|---------|-----|
| Installation | Built-in | npm install -g pm2 |
| Auto-restart | ✓ Yes | ✓ Yes |
| Boot startup | ✓ Native | Requires setup |
| Logging | ✓ journalctl | Custom logs |
| Security | ✓ Built-in | Limited |
| Resource limits | ✓ Native | Limited |
| Clustering | Manual | ✓ Built-in |
| Best for | Production servers | Development/Node.js apps |

## Production Checklist

- [ ] Service file installed
- [ ] Service enabled for auto-start
- [ ] Service running successfully
- [ ] Logs are clean (no errors)
- [ ] Auto-restart tested
- [ ] Nginx configured and running
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Health endpoint responding
- [ ] API key configured

## Health Monitoring

### Manual Check
```bash
curl http://localhost:3000/health
```

### Automated Monitoring (Optional)

Create a cron job to check health:
```bash
# Edit crontab
crontab -e

# Add this line (check every 5 minutes)
*/5 * * * * curl -f http://localhost:3000/health || sudo systemctl restart voice-ai-detection
```

## Logs Location

- **systemd logs**: `sudo journalctl -u voice-ai-detection`
- **Application logs**: `/var/www/voice-ai-detection/backend/logs/`
- **Nginx logs**: `/var/log/nginx/`

## Security Features

The service file includes security hardening:
- `NoNewPrivileges=true` - Prevents privilege escalation
- `PrivateTmp=true` - Isolated /tmp directory
- `ProtectSystem=strict` - Read-only system directories
- `ProtectHome=true` - No access to home directories
- `ReadWritePaths` - Only specific directories writable

## Summary

systemd provides enterprise-grade process management with:
- ✓ Automatic crash recovery
- ✓ Boot-time startup
- ✓ Integrated logging
- ✓ Security isolation
- ✓ Resource management
- ✓ Zero dependencies

Your server will automatically restart if it crashes, ensuring maximum uptime for production.
