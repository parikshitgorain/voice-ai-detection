// Settings page logic
const passwordForm = document.getElementById("passwordForm");
const passwordError = document.getElementById("passwordError");
const passwordSuccess = document.getElementById("passwordSuccess");
const logoutBtn = document.getElementById("logoutBtn");

// Check authentication
const token = localStorage.getItem("adminToken");
if (!token) {
  window.location.href = "/admin/login.html";
}

// Handle password change
passwordForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const currentPassword = document.getElementById("currentPassword").value;
  const newPassword = document.getElementById("newPassword").value;
  const confirmPassword = document.getElementById("confirmPassword").value;
  
  passwordError.textContent = "";
  passwordSuccess.textContent = "";
  
  if (newPassword !== confirmPassword) {
    passwordError.textContent = "New passwords do not match";
    return;
  }
  
  try {
    const response = await fetch("/admin/change-password", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify({
        currentPassword,
        newPassword,
        confirmPassword
      })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      passwordSuccess.textContent = data.message || "Password changed successfully";
      passwordForm.reset();
      
      // Clear token after 2 seconds and redirect to login
      setTimeout(() => {
        localStorage.removeItem("adminToken");
        window.location.href = "/admin/login.html";
      }, 2000);
    } else {
      passwordError.textContent = data.error || "Failed to change password";
    }
  } catch (err) {
    console.error("Error changing password:", err);
    passwordError.textContent = "Failed to change password";
  }
});

// Handle logout
logoutBtn.addEventListener("click", (e) => {
  e.preventDefault();
  localStorage.removeItem("adminToken");
  window.location.href = "/admin/login.html";
});
