// Main JavaScript for Multi-Label Text Classifier

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeCharacterCounter();
    initializeSampleTexts();
    initializeFormValidation();
    initializeTooltips();
    initializeAnimations();
});

// Character counter functionality
function initializeCharacterCounter() {
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('charCount');
    
    if (textArea && charCount) {
        textArea.addEventListener('input', function() {
            const currentLength = this.value.length;
            const maxLength = 10000;
            
            charCount.textContent = currentLength;
            
            // Update color based on character count
            charCount.className = '';
            if (currentLength > maxLength * 0.9) {
                charCount.classList.add('char-limit-danger');
            } else if (currentLength > maxLength * 0.8) {
                charCount.classList.add('char-limit-warning');
            }
            
            // Update progress bar if exists
            const progressBar = document.querySelector('.char-progress');
            if (progressBar) {
                const percentage = (currentLength / maxLength) * 100;
                progressBar.style.width = percentage + '%';
                
                if (percentage > 90) {
                    progressBar.className = 'progress-bar bg-danger';
                } else if (percentage > 80) {
                    progressBar.className = 'progress-bar bg-warning';
                } else {
                    progressBar.className = 'progress-bar bg-success';
                }
            }
        });
        
        // Trigger initial update
        textArea.dispatchEvent(new Event('input'));
    }
}

// Sample text functionality
function initializeSampleTexts() {
    const sampleButtons = document.querySelectorAll('.sample-text');
    const textArea = document.getElementById('text');
    
    sampleButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (textArea) {
                // Add fade effect
                textArea.style.opacity = '0.5';
                
                setTimeout(() => {
                    textArea.value = this.dataset.text;
                    textArea.style.opacity = '1';
                    textArea.focus();
                    
                    // Trigger character counter update
                    textArea.dispatchEvent(new Event('input'));
                    
                    // Add success feedback
                    this.classList.add('btn-success');
                    this.innerHTML = '<i class="fas fa-check me-1"></i>Yüklendi';
                    
                    setTimeout(() => {
                        this.classList.remove('btn-success');
                        this.innerHTML = this.innerHTML.replace('Yüklendi', this.textContent.split(' ')[1]);
                    }, 1000);
                }, 200);
            }
        });
    });
}

// Form validation
function initializeFormValidation() {
    const form = document.getElementById('predictionForm');
    const textArea = document.getElementById('text');
    const submitBtn = document.getElementById('predictBtn');
    const spinner = document.getElementById('loadingSpinner');
    
    if (form && textArea && submitBtn) {
        form.addEventListener('submit', function(e) {
            const text = textArea.value.trim();
            
            if (!text) {
                e.preventDefault();
                showAlert('Lütfen bir metin girin!', 'warning');
                textArea.focus();
                return;
            }
            
            if (text.length < 10) {
                e.preventDefault();
                showAlert('Metin çok kısa! En az 10 karakter gerekli.', 'warning');
                textArea.focus();
                return;
            }
            
            if (text.length > 10000) {
                e.preventDefault();
                showAlert('Metin çok uzun! Maksimum 10,000 karakter.', 'danger');
                textArea.focus();
                return;
            }
            
            // Show loading state
            submitBtn.disabled = true;
            if (spinner) spinner.classList.remove('d-none');
            
            // Add loading class to form
            form.classList.add('loading');
            
            // Set timeout to re-enable button if something goes wrong
            setTimeout(() => {
                if (submitBtn.disabled) {
                    submitBtn.disabled = false;
                    if (spinner) spinner.classList.add('d-none');
                    form.classList.remove('loading');
                }
            }, 30000);
        });
        
        // Clear button functionality
        const clearBtn = document.getElementById('clearText');
        if (clearBtn) {
            clearBtn.addEventListener('click', function() {
                textArea.value = '';
                textArea.dispatchEvent(new Event('input'));
                textArea.focus();
                
                // Add visual feedback
                this.classList.add('btn-success');
                this.innerHTML = '<i class="fas fa-check me-1"></i>Temizlendi';
                
                setTimeout(() => {
                    this.classList.remove('btn-success');
                    this.innerHTML = '<i class="fas fa-eraser me-1"></i>Temizle';
                }, 1000);
            });
        }
    }
}

// Initialize tooltips
function initializeTooltips() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Custom tooltips for model info
    const modelBadges = document.querySelectorAll('.model-badge');
    modelBadges.forEach(badge => {
        badge.addEventListener('mouseenter', function() {
            showModelInfo(this.dataset.model);
        });
    });
}

// Initialize animations
function initializeAnimations() {
    // Fade in animations for cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.1) + 's';
        card.classList.add('fade-in');
    });
    
    // Slide in animations for stat cards
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.15) + 's';
        card.classList.add('slide-in');
    });
}

// Utility functions
function showAlert(message, type = 'info', duration = 5000) {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.insertBefore(alert, mainContent.firstChild);
        
        // Auto dismiss
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }
}

function showModelInfo(modelName) {
    // This could show detailed model information in a tooltip or modal
    console.log('Model info requested for:', modelName);
}

// API functions for AJAX requests
async function predictText(text) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// Real-time prediction (optional feature)
function enableRealTimePreview() {
    const textArea = document.getElementById('text');
    const previewContainer = document.getElementById('realtime-preview');
    
    if (textArea && previewContainer) {
        let timeout;
        
        textArea.addEventListener('input', function() {
            clearTimeout(timeout);
            
            timeout = setTimeout(async () => {
                const text = this.value.trim();
                
                if (text.length > 50) { // Minimum length for preview
                    try {
                        previewContainer.innerHTML = '<div class="text-muted">Analiz ediliyor...</div>';
                        const results = await predictText(text);
                        displayPreview(results, previewContainer);
                    } catch (error) {
                        previewContainer.innerHTML = '<div class="text-danger">Önizleme hatası</div>';
                    }
                } else {
                    previewContainer.innerHTML = '';
                }
            }, 2000); // 2 second delay
        });
    }
}

function displayPreview(results, container) {
    if (!results.summary || !results.summary.consensus_labels) {
        container.innerHTML = '<div class="text-muted">Henüz tahmin yok</div>';
        return;
    }
    
    let html = '<div class="preview-results">';
    html += '<h6>Hızlı Önizleme:</h6>';
    
    if (results.summary.consensus_labels.length > 0) {
        results.summary.consensus_labels.forEach(label => {
            html += `<span class="badge bg-primary me-1">${label}</span>`;
        });
    } else {
        html += '<span class="text-muted">Belirsiz sonuç</span>';
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// Export functions
function exportResults(format, data) {
    switch (format) {
        case 'json':
            downloadJSON(data);
            break;
        case 'csv':
            downloadCSV(data);
            break;
        case 'txt':
            downloadTXT(data);
            break;
        default:
            console.error('Unsupported format:', format);
    }
}

function downloadJSON(data) {
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    downloadBlob(dataBlob, `results_${Date.now()}.json`);
}

function downloadCSV(data) {
    let csvContent = "Model,Category,Predicted,Probability\n";
    
    Object.entries(data.predictions).forEach(([modelName, result]) => {
        if (result.label_details) {
            result.label_details.forEach(detail => {
                csvContent += `"${modelName}","${detail.label}","${detail.predicted}","${detail.probability || 'N/A'}"\n`;
            });
        }
    });
    
    const dataBlob = new Blob([csvContent], { type: 'text/csv' });
    downloadBlob(dataBlob, `results_${Date.now()}.csv`);
}

function downloadTXT(data) {
    let content = `Text Classification Results\n`;
    content += `==============================\n\n`;
    content += `Original Text: ${data.original_text}\n\n`;
    
    Object.entries(data.predictions).forEach(([modelName, result]) => {
        content += `${modelName}:\n`;
        if (result.predicted_labels && result.predicted_labels.length > 0) {
            content += `  Categories: ${result.predicted_labels.join(', ')}\n`;
        } else {
            content += `  Categories: None\n`;
        }
        content += '\n';
    });
    
    const dataBlob = new Blob([content], { type: 'text/plain' });
    downloadBlob(dataBlob, `results_${Date.now()}.txt`);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to clear text
    if (e.key === 'Escape') {
        const textArea = document.getElementById('text');
        if (textArea && document.activeElement === textArea) {
            textArea.value = '';
            textArea.dispatchEvent(new Event('input'));
        }
    }
});

// Performance monitoring
function trackPerformance() {
    // Track page load time
    window.addEventListener('load', function() {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
        
        // Send to analytics if needed
        // analytics.track('page_load_time', loadTime);
    });
    
    // Track form submission time
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function() {
            const startTime = performance.now();
            
            // You could track this and send to analytics
            console.log('Form submission started at:', startTime);
        });
    }
}

// Initialize performance tracking
trackPerformance();

// Service Worker registration (for future PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('SW registered: ', registration);
            })
            .catch(function(registrationError) {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
