// ==========================================
// Hybrid Face Recognition System
// Utility Functions
// ==========================================

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Validate image file
function isValidImage(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    const maxSize = 16 * 1024 * 1024; // 16MB

    if (!validTypes.includes(file.type)) {
        Toast.error('Invalid file type. Please upload JPG, PNG, or WebP images.');
        return false;
    }

    if (file.size > maxSize) {
        Toast.error(`File too large. Maximum size is 16MB.`);
        return false;
    }

    return true;
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Copy to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        Toast.success('Copied to clipboard');
    } catch (error) {
        Toast.error('Failed to copy');
    }
}

// Download file
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Export data as JSON
function exportAsJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, filename);
    URL.revokeObjectURL(url);
    Toast.success('Data exported successfully');
}

// Export data as CSV
function exportAsCSV(data, filename) {
    if (!data || data.length === 0) {
        Toast.error('No data to export');
        return;
    }

    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => 
            headers.map(header => JSON.stringify(row[header] || '')).join(',')
        )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, filename);
    URL.revokeObjectURL(url);
    Toast.success('Data exported successfully');
}

// Generate unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    
    return date.toLocaleDateString();
}

// Calculate percentage
function calculatePercentage(part, whole) {
    if (whole === 0) return 0;
    return Math.round((part / whole) * 100);
}

// Clamp number between min and max
function clamp(number, min, max) {
    return Math.min(Math.max(number, min), max);
}

// Random number generator
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Smooth scroll to element
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Image compression
function compressImage(file, maxWidth = 800, maxHeight = 800, quality = 0.8) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                // Maintain aspect ratio
                if (width > height) {
                    if (width > maxWidth) {
                        height *= maxWidth / width;
                        width = maxWidth;
                    }
                } else {
                    if (height > maxHeight) {
                        width *= maxHeight / height;
                        height = maxHeight;
                    }
                }

                canvas.width = width;
                canvas.height = height;

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, file.type, quality);
            };
            img.onerror = reject;
        };
        reader.onerror = reject;
    });
}

// Create image thumbnail
function createThumbnail(file, size = 150) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = size;
                canvas.height = size;

                const ctx = canvas.getContext('2d');
                
                // Crop to square
                const minDimension = Math.min(img.width, img.height);
                const sx = (img.width - minDimension) / 2;
                const sy = (img.height - minDimension) / 2;

                ctx.drawImage(img, sx, sy, minDimension, minDimension, 0, 0, size, size);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, 'image/jpeg', 0.7);
            };
            img.onerror = reject;
        };
        reader.onerror = reject;
    });
}

// Drag and drop handler
function setupDragDrop(element, onDrop) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        element.addEventListener(eventName, () => {
            element.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, () => {
            element.classList.remove('drag-over');
        }, false);
    });

    element.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        onDrop(files);
    }, false);
}

// Batch file processor
class BatchProcessor {
    constructor(files, processFn) {
        this.files = Array.from(files);
        this.processFn = processFn;
        this.currentIndex = 0;
        this.results = [];
        this.errors = [];
    }

    async process() {
        for (let i = 0; i < this.files.length; i++) {
            this.currentIndex = i;
            const file = this.files[i];

            try {
                const result = await this.processFn(file, i, this.files.length);
                this.results.push(result);
            } catch (error) {
                this.errors.push({ file: file.name, error: error.message });
            }

            this.onProgress?.(i + 1, this.files.length);
        }

        return {
            results: this.results,
            errors: this.errors,
            success: this.results.length,
            failed: this.errors.length
        };
    }

    onProgress(current, total) {
        // Override this method to handle progress updates
    }
}

// Local storage with expiration
const StorageEx = {
    set(key, value, ttl = null) {
        const item = {
            value: value,
            expiration: ttl ? Date.now() + ttl : null
        };
        localStorage.setItem(key, JSON.stringify(item));
    },

    get(key) {
        const itemStr = localStorage.getItem(key);
        if (!itemStr) return null;

        const item = JSON.parse(itemStr);
        
        if (item.expiration && Date.now() > item.expiration) {
            localStorage.removeItem(key);
            return null;
        }

        return item.value;
    },

    remove(key) {
        localStorage.removeItem(key);
    },

    clear() {
        localStorage.clear();
    }
};

// Event emitter for custom events
class EventEmitter {
    constructor() {
        this.events = {};
    }

    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }

    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
}

// Performance timer
const Timer = {
    timers: {},

    start(label) {
        this.timers[label] = performance.now();
    },

    end(label) {
        if (!this.timers[label]) return 0;
        const elapsed = performance.now() - this.timers[label];
        delete this.timers[label];
        return elapsed;
    }
};

// Console styling
function styledLog(message, color = '#667eea') {
    console.log(`%c${message}`, `color: ${color}; font-weight: bold; font-size: 14px;`);
}

// Initialize utilities
document.addEventListener('DOMContentLoaded', () => {
    styledLog('Hybrid Face Recognition Utilities loaded', '#667eea');
});
