let keys = [];

async function loadKeys() {
  try {
    const res = await apiCall('/admin/api-keys');
    const data = await res.json();
    keys = data.keys;
    renderKeys();
  } catch (err) {
    console.error('Failed to load keys:', err);
    document.getElementById('keysTableBody').innerHTML = 
      '<tr><td colspan="7" class="loading">Failed to load keys</td></tr>';
  }
}

function renderKeys() {
  const tbody = document.getElementById('keysTableBody');
  
  if (keys.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading">No API keys yet</td></tr>';
    return;
  }
  
  tbody.innerHTML = keys.map(key => `
    <tr>
      <td><span class="key-preview">...${key.preview}</span></td>
      <td><span class="status-badge status-${key.status}">${key.status}</span></td>
      <td>${new Date(key.created_at).toLocaleDateString()}</td>
      <td>${key.usage.total_requests.toLocaleString()}</td>
      <td>${key.usage.today_requests.toLocaleString()}</td>
      <td>${key.usage.last_used ? new Date(key.usage.last_used).toLocaleString() : 'Never'}</td>
      <td>
        <div class="action-btns">
          ${key.status === 'active' 
            ? `<button onclick="toggleKey('${key.id}', 'inactive')">Disable</button>`
            : `<button onclick="toggleKey('${key.id}', 'active')">Enable</button>`
          }
          <button class="delete-btn" onclick="deleteKey('${key.id}')">Delete</button>
        </div>
      </td>
    </tr>
  `).join('');
}

async function createKey() {
  try {
    const res = await apiCall('/admin/api-keys', { method: 'POST' });
    const data = await res.json();
    
    // Show modal with new key
    document.getElementById('newKeyValue').value = data.key;
    document.getElementById('newKeyModal').style.display = 'flex';
    
    // Reload keys list
    await loadKeys();
  } catch (err) {
    console.error('Failed to create key:', err);
    alert('Failed to create API key');
  }
}

async function toggleKey(keyId, status) {
  try {
    await apiCall(`/admin/api-keys/${keyId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status })
    });
    await loadKeys();
  } catch (err) {
    console.error('Failed to toggle key:', err);
    alert('Failed to update API key');
  }
}

async function deleteKey(keyId) {
  if (!confirm('Are you sure you want to delete this API key? This cannot be undone.')) {
    return;
  }
  
  try {
    await apiCall(`/admin/api-keys/${keyId}`, { method: 'DELETE' });
    await loadKeys();
  } catch (err) {
    console.error('Failed to delete key:', err);
    alert('Failed to delete API key');
  }
}

function copyKey() {
  const input = document.getElementById('newKeyValue');
  input.select();
  document.execCommand('copy');
  
  const btn = document.getElementById('copyKeyBtn');
  btn.textContent = 'Copied!';
  setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
}

document.addEventListener('DOMContentLoaded', () => {
  loadKeys();
  
  document.getElementById('createKeyBtn').addEventListener('click', createKey);
  document.getElementById('copyKeyBtn').addEventListener('click', copyKey);
  document.getElementById('closeModalBtn').addEventListener('click', () => {
    document.getElementById('newKeyModal').style.display = 'none';
  });
});
