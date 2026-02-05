// Logs Page JavaScript
let currentLogs = [];

const checkAuth = async () => {
  const token = localStorage.getItem('adminToken');
  if (!token) {
    window.location.href = '/admin/login.html';
    return false;
  }
  
  try {
    const res = await fetch('/admin/session', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) {
      localStorage.removeItem('adminToken');
      window.location.href = '/admin/login.html';
      return false;
    }
    return true;
  } catch (err) {
    console.error('Auth check failed:', err);
    window.location.href = '/admin/login.html';
    return false;
  }
};

const loadLogs = async () => {
  const token = localStorage.getItem('adminToken');
  const statusFilter = document.getElementById('statusFilter').value;
  const limit = document.getElementById('limitFilter').value;
  const ipFilter = document.getElementById('ipFilter').value.trim();
  
  const logsContainer = document.getElementById('logsContainer');
  logsContainer.innerHTML = '<div class="loading">Loading logs...</div>';
  
  try {
    const params = new URLSearchParams({
      limit,
      status: statusFilter !== 'all' ? statusFilter : ''
    });
    
    const res = await fetch(`/admin/logs?${params}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    
    if (!res.ok) {
      throw new Error('Failed to fetch logs');
    }
    
    const data = await res.json();
    currentLogs = data.logs || [];
    
    // Apply IP filter client-side
    let filteredLogs = currentLogs;
    if (ipFilter) {
      filteredLogs = currentLogs.filter(log => 
        log.ip && log.ip.toLowerCase().includes(ipFilter.toLowerCase())
      );
    }
    
    updateStats(filteredLogs);
    renderLogs(filteredLogs);
  } catch (err) {
    console.error('Failed to load logs:', err);
    logsContainer.innerHTML = '<div class="no-logs">Failed to load logs. Please try again.</div>';
  }
};

const updateStats = (logs) => {
  const total = logs.length;
  const success = logs.filter(log => log.status === 'success').length;
  const errors = logs.filter(log => log.status === 'error').length;
  
  // Calculate average processing time from successful requests
  const processingTimes = logs
    .filter(log => log.status === 'success' && log.processingMs !== undefined && log.processingMs !== null)
    .map(log => log.processingMs);
  
  const avgProcessing = processingTimes.length > 0
    ? Math.round(processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length)
    : '-';
  
  document.getElementById('totalLogs').textContent = total;
  document.getElementById('successCount').textContent = success;
  document.getElementById('errorCount').textContent = errors;
  document.getElementById('avgProcessing').textContent = avgProcessing === '-' ? avgProcessing : `${avgProcessing}ms`;
};

const renderLogs = (logs) => {
  const logsContainer = document.getElementById('logsContainer');
  
  if (logs.length === 0) {
    logsContainer.innerHTML = '<div class="no-logs">No logs found matching the filters.</div>';
    return;
  }
  
  logsContainer.innerHTML = logs.map(log => {
    const date = new Date(log.ts);
    const timestamp = date.toLocaleString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
    
    const statusClass = log.status === 'success' ? 'success' : 'error';
    
    let details = [];
    
    if (log.ip) {
      const shortIp = log.ip.length > 30 ? log.ip.substring(0, 30) + '...' : log.ip;
      details.push(`<div class="log-detail" title="${log.ip}"><strong>IP:</strong><span class="log-detail-value">${shortIp}</span></div>`);
    }
    
    if (log.requestId) {
      details.push(`<div class="log-detail" title="${log.requestId}"><strong>Request ID:</strong><span class="log-detail-value">${log.requestId.substring(0, 8)}...</span></div>`);
    }
    
    if (log.language) {
      details.push(`<div class="log-detail"><strong>Language:</strong><span class="log-detail-value">${log.language}</span></div>`);
    }
    
    if (log.classification) {
      const classification = log.classification === 'AI_GENERATED' ? 'AI Generated' : 'Human';
      details.push(`<div class="log-detail"><strong>Classification:</strong><span class="log-detail-value">${classification}</span></div>`);
    }
    
    if (log.confidenceScore !== undefined && log.confidenceScore !== null) {
      const confidence = (log.confidenceScore * 100).toFixed(2);
      details.push(`<div class="log-detail"><strong>Confidence:</strong><span class="log-detail-value">${confidence}%</span></div>`);
    }
    
    if (log.processingMs !== undefined && log.processingMs !== null) {
      details.push(`<div class="log-detail"><strong>Processing:</strong><span class="log-detail-value">${log.processingMs}ms</span></div>`);
    }
    
    if (log.queued !== undefined) {
      details.push(`<div class="log-detail"><strong>Queued:</strong><span class="log-detail-value">${log.queued ? 'Yes' : 'No'}</span></div>`);
    }
    
    if (log.reason) {
      details.push(`<div class="log-detail"><strong>Error:</strong><span class="log-detail-value">${log.reason}</span></div>`);
    }
    
    return `
      <div class="log-entry">
        <div class="log-header">
          <div class="log-timestamp">${timestamp}</div>
          <div class="log-status ${statusClass}">${log.status}</div>
        </div>
        ${details.length > 0 ? `<div class="log-details">${details.join('')}</div>` : ''}
      </div>
    `;
  }).join('');
};

// Logout handler
document.getElementById('logoutBtn').addEventListener('click', (e) => {
  e.preventDefault();
  localStorage.removeItem('adminToken');
  window.location.href = '/admin/login.html';
});

// Auto-refresh logs every 30 seconds
setInterval(loadLogs, 30000);

// Filter change handlers
document.getElementById('statusFilter').addEventListener('change', loadLogs);
document.getElementById('limitFilter').addEventListener('change', loadLogs);
document.getElementById('ipFilter').addEventListener('input', () => {
  // Debounce IP filter
  clearTimeout(window.ipFilterTimeout);
  window.ipFilterTimeout = setTimeout(() => {
    const ipFilter = document.getElementById('ipFilter').value.trim();
    let filteredLogs = currentLogs;
    if (ipFilter) {
      filteredLogs = currentLogs.filter(log => 
        log.ip && log.ip.toLowerCase().includes(ipFilter.toLowerCase())
      );
    }
    updateStats(filteredLogs);
    renderLogs(filteredLogs);
  }, 300);
});

// Initialize
checkAuth().then(authed => {
  if (authed) loadLogs();
});
