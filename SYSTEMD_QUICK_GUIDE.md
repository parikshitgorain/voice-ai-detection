# systemd Quick Guide - Auto-Restart on Crash

## What is systemd?

systemd is Linux's built-in service manager. It's what we're using instead of PM2.

## Why systemd Instead of PM2?

| Feature | systemd | PM2 |
|---------|---------|-----|
| Installation | ✓ Built into Linux | ✗ npm install -g pm2 |
| Auto-restart on crash | ✓ Yes | ✓ Yes |
| Start on boot | ✓ Native | Requires setup |
| Production-ready | ✓ Industry standard | Good for Node.js |
| Dependencies | ✓ None | Requires Node.js globally |

## How Auto-Restart Works

When your server crashes:
1. systemd detects process died
2. Waits 5 seconds (configurable)
3. Automatically restarts it
4. Logs the event
5. Keeps monitoring

## Setup (One-Time)

```bash
# 1. Copy service file
sudo cp voice-ai-detection.service /etc/systemd/system/

# 2. Enable auto-start on boot
sudo systemctl enable voice-ai-detection

# 3. Start the service
sudo systemctl start voice-ai-detection

# 4. Check it's running
sudo systemctl status voice-ai-detection
```

## Daily Commands

```bash
# Check if running
sudo systemctl status voice-ai-detection

# View logs (live)
sudo journalctl -u voice-ai-detection -f

# Restart
sudo systemctl restart voice-ai-detection

# Stop
sudo systemctl stop voice-ai-detection
```

## Test Auto-Restart

```bash
# Kill the process (simulate crash)
sudo pkill -9 node

# Wait 5 seconds
sleep 5

# Check - it should be running again!
sudo systemctl status voice-ai-detection
```

## Configuration File

Location: `voice-ai-detection.service`

Key settings:
```ini
Restart=always          # Always restart on crash
RestartSec=5           # Wait 5 seconds before restart
MemoryMax=2G           # Limit memory to 2GB
```

## What Happens on Server Reboot?

1. Server boots up
2. systemd starts automatically
3. systemd reads enabled services
4. Starts voice-ai-detection automatically
5. Your API is online!

No manual intervention needed.

## Monitoring

### Check uptime
```bash
sudo systemctl status voice-ai-detection | grep Active
```

### Check memory
```bash
sudo systemctl status voice-ai-detection | grep Memory
```

### Count restarts
```bash
sudo journalctl -u voice-ai-detection | grep "Started Voice" | wc -l
```

## Summary

✓ **Auto-restart on crash** - Configured with `Restart=always`
✓ **Auto-start on boot** - Configured with `systemctl enable`
✓ **No dependencies** - Built into Linux
✓ **Production-ready** - Used by major companies
✓ **Better than PM2** - For production servers

Your server will automatically restart if it crashes, ensuring maximum uptime!
