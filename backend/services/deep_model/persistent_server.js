/**
 * Persistent Python inference server manager
 * Keeps Python process alive with models in GPU memory for instant responses
 */
const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

class PersistentInferenceServer {
  constructor(config = {}) {
    this.pythonPath = config.pythonPath || 'python3';
    this.device = config.device || 'cuda';
    this.scriptPath = config.scriptPath || path.join(__dirname, '..', '..', 'deep', 'inference_server.py');
    
    this.process = null;
    this.ready = false;
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.startTime = null;
    
    console.log('[PersistentServer] Configured:', {
      pythonPath: this.pythonPath,
      device: this.device,
      scriptPath: this.scriptPath
    });
  }
  
  async start() {
    if (this.process) {
      console.log('[PersistentServer] Already running');
      return;
    }
    
    return new Promise((resolve, reject) => {
      console.log('[PersistentServer] Starting server...');
      this.startTime = Date.now();
      
      try {
        this.process = spawn(this.pythonPath, [
          this.scriptPath,
          '--device', this.device
        ], {
          stdio: ['pipe', 'pipe', 'pipe']
        });
      } catch (err) {
        return reject(new Error(`Failed to spawn Python server: ${err.message}`));
      }
      
      // Setup readline for stdout
      const rl = readline.createInterface({
        input: this.process.stdout,
        crlfDelay: Infinity
      });
      
      rl.on('line', (line) => {
        try {
          const response = JSON.parse(line);
          
          // Handle pending request
          const reqId = this.currentRequestId;
          if (reqId !== undefined && this.pendingRequests.has(reqId)) {
            const { resolve: resolveReq } = this.pendingRequests.get(reqId);
            this.pendingRequests.delete(reqId);
            resolveReq(response);
          }
        } catch (err) {
          console.error('[PersistentServer] Failed to parse response:', line);
        }
      });
      
      // Monitor stderr for ready signal
      const stderrRl = readline.createInterface({
        input: this.process.stderr,
        crlfDelay: Infinity
      });
      
      stderrRl.on('line', (line) => {
        console.log('[PersistentServer]', line);
        
        if (line.includes('Ready for requests')) {
          this.ready = true;
          const elapsed = Date.now() - this.startTime;
          console.log(`[PersistentServer] Server ready in ${elapsed}ms`);
          resolve();
        }
      });
      
      this.process.on('error', (err) => {
        console.error('[PersistentServer] Process error:', err);
        this.ready = false;
        reject(err);
      });
      
      this.process.on('exit', (code, signal) => {
        console.log(`[PersistentServer] Process exited: code=${code}, signal=${signal}`);
        this.ready = false;
        this.process = null;
        
        // Reject all pending requests
        for (const [reqId, { reject: rejectReq }] of this.pendingRequests) {
          rejectReq(new Error('Server process exited'));
        }
        this.pendingRequests.clear();
      });
      
      // Timeout if server doesn't start in 30 seconds
      setTimeout(() => {
        if (!this.ready) {
          reject(new Error('Server startup timeout'));
          this.stop();
        }
      }, 30000);
    });
  }
  
  async infer(audioPath, modelPath, timeoutMs = 15000) {
    if (!this.ready || !this.process) {
      throw new Error('Server not ready');
    }
    
    return new Promise((resolve, reject) => {
      const reqId = this.requestId++;
      this.currentRequestId = reqId;
      
      const timer = setTimeout(() => {
        if (this.pendingRequests.has(reqId)) {
          this.pendingRequests.delete(reqId);
          reject(new Error('Inference timeout'));
        }
      }, timeoutMs);
      
      this.pendingRequests.set(reqId, {
        resolve: (response) => {
          clearTimeout(timer);
          if (response.error) {
            reject(new Error(response.error));
          } else {
            resolve(response);
          }
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        }
      });
      
      // Send request
      const request = {
        action: 'infer',
        audio: audioPath,
        model: modelPath
      };
      
      try {
        this.process.stdin.write(JSON.stringify(request) + '\n');
      } catch (err) {
        clearTimeout(timer);
        this.pendingRequests.delete(reqId);
        reject(new Error(`Failed to send request: ${err.message}`));
      }
    });
  }
  
  async ping() {
    if (!this.ready || !this.process) {
      return false;
    }
    
    try {
      const request = { action: 'ping' };
      this.process.stdin.write(JSON.stringify(request) + '\n');
      return true;
    } catch (err) {
      return false;
    }
  }
  
  stop() {
    if (!this.process) {
      return;
    }
    
    console.log('[PersistentServer] Stopping server...');
    
    try {
      // Try graceful shutdown
      const request = { action: 'exit' };
      this.process.stdin.write(JSON.stringify(request) + '\n');
      
      // Force kill after 2 seconds
      setTimeout(() => {
        if (this.process) {
          this.process.kill('SIGKILL');
        }
      }, 2000);
    } catch (err) {
      // Force kill immediately if graceful fails
      if (this.process) {
        this.process.kill('SIGKILL');
      }
    }
    
    this.ready = false;
    this.process = null;
  }
  
  isReady() {
    return this.ready && this.process !== null;
  }
}

// Singleton instance
let serverInstance = null;

const getServer = (config) => {
  if (!serverInstance) {
    serverInstance = new PersistentInferenceServer(config);
  }
  return serverInstance;
};

const startServer = async (config) => {
  const server = getServer(config);
  if (!server.isReady()) {
    await server.start();
  }
  return server;
};

const stopServer = () => {
  if (serverInstance) {
    serverInstance.stop();
    serverInstance = null;
  }
};

module.exports = {
  PersistentInferenceServer,
  getServer,
  startServer,
  stopServer
};
