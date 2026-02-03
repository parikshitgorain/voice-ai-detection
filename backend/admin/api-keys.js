// API Keys page logic
const token = localStorage.getItem("adminToken");
if (!token) {
  window.location.href = "/admin/login.html";
}

// Elements
const keysTableBody = document.getElementById("keysTableBody");
const createKeyBtn = document.getElementById("createKeyBtn");
const logoutBtn = document.getElementById("logoutBtn");

// Modals
const createKeyModal = document.getElementById("createKeyModal");
const showKeyModal = document.getElementById("showKeyModal");
const editLimitsModal = document.getElementById("editLimitsModal");
const createKeyForm = document.getElementById("createKeyForm");
const editLimitsForm = document.getElementById("editLimitsForm");

// Load API keys
async function loadApiKeys() {
  try {
    const response = await fetch("/admin/api-keys", {
      headers: { "Authorization": `Bearer ${token}` }
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem("adminToken");
        window.location.href = "/admin/login.html";
        return;
      }
      throw new Error("Failed to load API keys");
    }
    
    const data = await response.json();
    renderApiKeys(data.keys);
  } catch (err) {
    console.error("Error loading API keys:", err);
    keysTableBody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: red;">Failed to load API keys</td></tr>';
  }
}

// Render API keys table
function renderApiKeys(keys) {
  if (!keys || keys.length === 0) {
    keysTableBody.innerHTML = '<tr><td colspan="10" style="text-align: center;">No API keys found</td></tr>';
    document.getElementById('mobileCards').innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-secondary);">No API keys found</div>';
    return;
  }
  
  // Render desktop table
  keysTableBody.innerHTML = keys.map(key => {
    const usageData = key.usage || {};
    const dailyLimit = key.daily_limit || 0;
    const perMinuteLimit = key.per_minute_limit || 0;
    const totalLimit = key.total_limit || 0;
    
    return `
      <tr>
        <td>${key.id}</td>
        <td>${key.name || "Unnamed"}</td>
        <td><span class="badge badge-${key.type === 'unlimited' ? 'success' : 'info'}">${key.type}</span></td>
        <td><span class="badge badge-${key.status === 'active' ? 'success' : 'warning'}">${key.status}</span></td>
        <td>${key.type === 'limited' ? (dailyLimit || '∞') : '∞'}</td>
        <td>${key.type === 'limited' ? (perMinuteLimit || '∞') : '∞'}</td>
        <td>${key.type === 'limited' ? (totalLimit || '∞') : '∞'}</td>
        <td>
          Today: ${usageData.today_requests || 0}<br>
          Total: ${usageData.total_requests || 0}
        </td>
        <td>${new Date(key.created_at).toLocaleDateString()}</td>
        <td class="action-buttons">
          ${key.type === 'limited' ? `<button class="btn btn-sm btn-secondary" onclick="editLimits('${key.id}', ${dailyLimit}, ${perMinuteLimit}, ${totalLimit})">Edit Limits</button>` : ''}
          <button class="btn btn-sm btn-${key.status === 'active' ? 'warning' : 'success'}" onclick="toggleStatus('${key.id}', '${key.status}')">${key.status === 'active' ? 'Disable' : 'Enable'}</button>
          <button class="btn btn-sm btn-danger" onclick="deleteKey('${key.id}')">Delete</button>
        </td>
      </tr>
    `;
  }).join("");
  
  // Render mobile cards
  const mobileCardsContainer = document.getElementById('mobileCards');
  if (mobileCardsContainer) {
    mobileCardsContainer.innerHTML = keys.map(key => {
      const usageData = key.usage || {};
      const dailyLimit = key.daily_limit || 0;
      const perMinuteLimit = key.per_minute_limit || 0;
      const totalLimit = key.total_limit || 0;
      
      return `
        <div class="mobile-card">
          <div class="mobile-card-header">
            <div class="mobile-card-title">${key.name || "Unnamed"}</div>
            <span class="badge badge-${key.status === 'active' ? 'success' : 'warning'}">${key.status}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Type</span>
            <span class="mobile-card-value"><span class="badge badge-${key.type === 'unlimited' ? 'success' : 'info'}">${key.type}</span></span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Daily Limit</span>
            <span class="mobile-card-value">${key.type === 'limited' ? (dailyLimit || '∞') : '∞'}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Per Minute</span>
            <span class="mobile-card-value">${key.type === 'limited' ? (perMinuteLimit || '∞') : '∞'}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Usage Today</span>
            <span class="mobile-card-value">${usageData.today_requests || 0}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Total Usage</span>
            <span class="mobile-card-value">${usageData.total_requests || 0}</span>
          </div>
          <div class="mobile-card-actions">
            ${key.type === 'limited' ? `<button class="btn btn-sm btn-secondary" onclick="editLimits('${key.id}', ${dailyLimit}, ${perMinuteLimit}, ${totalLimit})">Edit</button>` : ''}
            <button class="btn btn-sm btn-${key.status === 'active' ? 'warning' : 'success'}" onclick="toggleStatus('${key.id}', '${key.status}')">${key.status === 'active' ? 'Disable' : 'Enable'}</button>
            <button class="btn btn-sm btn-danger" onclick="deleteKey('${key.id}')">Delete</button>
          </div>
        </div>
      `;
    }).join("");
  }
}

// Toggle key type input
document.getElementById("keyType").addEventListener("change", (e) => {
  const limitsSection = document.getElementById("limitsSection");
  if (e.target.value === "limited") {
    limitsSection.style.display = "block";
  } else {
    limitsSection.style.display = "none";
  }
});

// Create new key
createKeyBtn.addEventListener("click", () => {
  createKeyModal.style.display = "block";
  document.getElementById("createError").textContent = "";
  createKeyForm.reset();
  document.getElementById("limitsSection").style.display = "none";
});

createKeyForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const formData = new FormData(createKeyForm);
  const data = {
    name: formData.get("name"),
    limits: {
      type: formData.get("type")
    }
  };
  
  if (data.limits.type === "limited") {
    data.limits.daily_limit = parseInt(formData.get("daily_limit")) || 0;
    data.limits.per_minute_limit = parseInt(formData.get("per_minute_limit")) || 0;
    data.limits.total_limit = parseInt(formData.get("total_limit")) || 0;
  }
  
  try {
    const response = await fetch("/admin/api-keys", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    
    if (response.ok) {
      createKeyModal.style.display = "none";
      showNewKey(result.key, result.name, result.id);
      loadApiKeys();
    } else {
      document.getElementById("createError").textContent = result.error || "Failed to create key";
    }
  } catch (err) {
    console.error("Error creating key:", err);
    document.getElementById("createError").textContent = "Failed to create key";
  }
});

// Show new key modal
function showNewKey(key, name, id) {
  document.getElementById("newApiKey").textContent = key;
  document.getElementById("keyName").textContent = name;
  document.getElementById("keyId").textContent = id;
  showKeyModal.style.display = "block";
}

// Copy key to clipboard
document.getElementById("copyKeyBtn").addEventListener("click", () => {
  const keyText = document.getElementById("newApiKey").textContent;
  navigator.clipboard.writeText(keyText).then(() => {
    document.getElementById("copyKeyBtn").textContent = "Copied!";
    setTimeout(() => {
      document.getElementById("copyKeyBtn").textContent = "Copy";
    }, 2000);
  });
});

// Edit limits
function editLimits(keyId, dailyLimit, perMinuteLimit, totalLimit) {
  document.getElementById("editKeyId").value = keyId;
  document.getElementById("editDailyLimit").value = dailyLimit || 0;
  document.getElementById("editPerMinuteLimit").value = perMinuteLimit || 0;
  document.getElementById("editTotalLimit").value = totalLimit || 0;
  editLimitsModal.style.display = "block";
  document.getElementById("editError").textContent = "";
}

editLimitsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const keyId = document.getElementById("editKeyId").value;
  const limits = {
    type: "limited",
    daily_limit: parseInt(document.getElementById("editDailyLimit").value) || 0,
    per_minute_limit: parseInt(document.getElementById("editPerMinuteLimit").value) || 0,
    total_limit: parseInt(document.getElementById("editTotalLimit").value) || 0
  };
  
  try {
    const response = await fetch(`/admin/api-keys/${keyId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify({ limits })
    });
    
    if (response.ok) {
      editLimitsModal.style.display = "none";
      loadApiKeys();
    } else {
      const result = await response.json();
      document.getElementById("editError").textContent = result.error || "Failed to update limits";
    }
  } catch (err) {
    console.error("Error updating limits:", err);
    document.getElementById("editError").textContent = "Failed to update limits";
  }
});

// Toggle key status
async function toggleStatus(keyId, currentStatus) {
  const newStatus = currentStatus === "active" ? "inactive" : "active";
  
  try {
    const response = await fetch(`/admin/api-keys/${keyId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify({ status: newStatus })
    });
    
    if (response.ok) {
      loadApiKeys();
    } else {
      alert("Failed to update key status");
    }
  } catch (err) {
    console.error("Error updating key status:", err);
    alert("Failed to update key status");
  }
}

// Delete key
async function deleteKey(keyId) {
  if (!confirm("Are you sure you want to delete this API key? This action cannot be undone.")) {
    return;
  }
  
  try {
    const response = await fetch(`/admin/api-keys/${keyId}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${token}` }
    });
    
    if (response.ok) {
      loadApiKeys();
    } else {
      alert("Failed to delete key");
    }
  } catch (err) {
    console.error("Error deleting key:", err);
    alert("Failed to delete key");
  }
}

// Close modals
document.querySelectorAll(".close").forEach(closeBtn => {
  closeBtn.addEventListener("click", function() {
    this.closest(".modal").style.display = "none";
  });
});

window.addEventListener("click", (e) => {
  if (e.target.classList.contains("modal")) {
    e.target.style.display = "none";
  }
});

// Logout
logoutBtn.addEventListener("click", (e) => {
  e.preventDefault();
  localStorage.removeItem("adminToken");
  window.location.href = "/admin/login.html";
});

// Load keys on page load
loadApiKeys();
