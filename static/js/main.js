// ==========================================
// Hybrid Face Recognition System
// Main Application JavaScript
// ==========================================

// Toast Notification System
class Toast {
    static success(message) {
        this.show(message, 'success');
    }

    static error(message) {
        this.show(message, 'error');
    }

    static info(message) {
        this.show(message, 'info');
    }

    static show(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                     type === 'error' ? 'exclamation-circle' : 'info-circle';
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.remove()" class="text-white/80 hover:text-white">
                <i class="fas fa-times"></i>
            </button>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
}

// Modal System
class Modal {
    constructor(title, content) {
        this.title = title;
        this.content = content;
        this.modal = null;
    }

    show() {
        this.modal = document.createElement('div');
        this.modal.className = 'modal-overlay';
        this.modal.onclick = (e) => {
            if (e.target === this.modal) this.hide();
        };

        this.modal.innerHTML = `
            <div class="modal-content" style="width: 600px; max-width: 90vw;">
                <div class="glass-header">
                    <h3 class="text-xl font-bold">${this.title}</h3>
                    <button onclick="this.closest('.modal-overlay').remove()" class="btn btn-ghost">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="glass-content">
                    ${this.content}
                </div>
            </div>
        `;

        document.body.appendChild(this.modal);
    }

    hide() {
        if (this.modal) {
            this.modal.remove();
            this.modal = null;
        }
    }
}

// Loading Overlay
function showLoading(message = 'Processing...') {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.querySelector('p').textContent = message;
        loader.classList.remove('hidden');
    }
}

function hideLoading() {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.classList.add('hidden');
    }
}

// Chart Helper
function createChart(elementId, config) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), config);
}

// API Communication Layer
const API = {
    async predict(formData) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    },

    async batchPredict(formData) {
        try {
            const response = await fetch('/api/batch-predict', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('Batch prediction error:', error);
            throw error;
        }
    },

    async cameraCapture(imageData) {
        try {
            const response = await fetch('/api/camera-capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });
            return await response.json();
        } catch (error) {
            console.error('Camera capture error:', error);
            throw error;
        }
    },

    async getHistory(page = 1, limit = 20) {
        try {
            const response = await fetch(`/api/history?page=${page}&limit=${limit}`);
            return await response.json();
        } catch (error) {
            console.error('History fetch error:', error);
            throw error;
        }
    },

    async getStats() {
        try {
            const response = await fetch('/api/stats');
            return await response.json();
        } catch (error) {
            console.error('Stats fetch error:', error);
            throw error;
        }
    },
    
    async getEvaluation() {
        try {
            const response = await fetch('/api/evaluation');
            if (response.ok) {
                return await response.json();
            }
            return null;
        } catch (error) {
            console.error('Evaluation fetch error:', error);
            return null;
        }
    },

    async getPersons() {
        try {
            const response = await fetch('/api/persons');
            return await response.json();
        } catch (error) {
            console.error('Persons fetch error:', error);
            throw error;
        }
    }
};

// Local Storage Manager
const Storage = {
    getHistory() {
        try {
            return JSON.parse(localStorage.getItem('recognitionHistory') || '[]');
        } catch {
            return [];
        }
    },

    addToHistory(record) {
        const history = this.getHistory();
        history.unshift({
            ...record,
            timestamp: new Date().toISOString()
        });
        
        // Keep only last 100 records
        if (history.length > 100) {
            history.length = 100;
        }
        
        localStorage.setItem('recognitionHistory', JSON.stringify(history));
    },

    clearHistory() {
        localStorage.removeItem('recognitionHistory');
    },

    getSettings() {
        return JSON.parse(localStorage.getItem('appSettings') || '{}');
    },

    saveSettings(settings) {
        localStorage.setItem('appSettings', JSON.stringify(settings));
    }
};

// View Navigation
const Navigation = {
    currentView: 'dashboard',

    navigateTo(viewId) {
        // Hide all views
        document.querySelectorAll('.view-section').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(`${viewId}View`);
        if (targetView) {
            targetView.classList.add('active');
        }

        // Update sidebar active state
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('bg-white/20', 'active');
        });

        const activeNav = document.querySelector(`[data-view="${viewId}"]`);
        if (activeNav) {
            activeNav.classList.add('bg-white/20', 'active');
        }

        this.currentView = viewId;

        // Trigger view-specific initialization
        this.onViewChange(viewId);
    },

    onViewChange(viewId) {
        switch(viewId) {
            case 'dashboard':
                Dashboard.init();
                break;
            case 'history':
                HistoryView.init();
                break;
            case 'database':
                DatabaseView.init();
                break;
            case 'analytics':
                AnalyticsView.init();
                break;
        }
    }
};

// Dashboard Module
const Dashboard = {
    charts: {},

    init() {
        this.loadStats();
        this.loadRecentActivity();
    },

    async loadStats() {
        try {
            const stats = await API.getStats();
            this.updateStatCards(stats);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    },

    updateStatCards(stats) {
        const cards = {
            'totalRecognitions': stats.totalRecognitions || 0,
            'averageConfidence': stats.averageConfidence || '0%',
            'personsInDatabase': stats.personsCount || 0,
            'modelAccuracy': stats.modelAccuracy || '0%'
        };

        Object.entries(cards).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    },

    loadRecentActivity() {
        const history = Storage.getHistory().slice(0, 5);
        const container = document.getElementById('recentActivity');
        
        if (!container || history.length === 0) return;

        container.innerHTML = history.map(record => `
            <div class="flex items-center space-x-4 p-3 rounded-lg hover:bg-gray-50 transition cursor-pointer"
                 onclick="showRecognitionDetail('${record.timestamp}')">
                <img src="${record.processedImage || '/static/samples/sample_person1_1.png'}" 
                     class="w-12 h-12 rounded-lg object-cover">
                <div class="flex-1">
                    <p class="font-semibold text-sm">Person #${record.classId || 'Unknown'}</p>
                    <p class="text-xs text-gray-500">${new Date(record.timestamp).toLocaleString()}</p>
                </div>
                <span class="badge badge-success">${record.confidence || '0%'}</span>
            </div>
        `).join('');
    },

    updateResults(data) {
        const dashboard = document.getElementById('resultsDashboard');
        if (!dashboard) return;

        dashboard.classList.remove('hidden');
        dashboard.classList.add('animate-slide-up');

        document.getElementById('resIdentity').textContent = `Person #${data.class_id}`;
        document.getElementById('resConfidence').textContent = data.confidence;
        document.getElementById('resVector').textContent = data.feature_count;
        
        if (data.processed_image) {
            document.getElementById('resProc').src = `/${data.processed_image}?${Date.now()}`;
        }

        // Update chart
        this.updateProbabilityChart(data.top_predictions);

        // Add to history
        Storage.addToHistory({
            classId: data.class_id,
            confidence: data.confidence,
            processedImage: data.processed_image,
            featureCount: data.feature_count
        });

        // Scroll to results
        dashboard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },

    updateProbabilityChart(predictions) {
        const ctx = document.getElementById('probChart');
        if (!ctx) return;

        if (this.chances.probChart) {
            this.chances.probChart.destroy();
        }

        const labels = predictions.map(p => `ID ${p.class_id}`);
        const values = predictions.map(p => parseFloat(p.confidence));

        this.chances.probChart = createChart('probChart', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence %',
                    data: values,
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(118, 75, 162, 0.8)',
                        'rgba(79, 172, 254, 0.8)',
                        'rgba(240, 147, 251, 0.8)',
                        'rgba(250, 112, 154, 0.8)'
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.parsed.y}%`
                        }
                    }
                },
                scales: { 
                    y: { 
                        beginAtZero: true, 
                        max: 100, 
                        grid: { display: false } 
                    },
                    x: { 
                        grid: { display: false } 
                    }
                }
            }
        });
    }
};

// History View Module
const HistoryView = {
    currentPage: 1,
    perPage: 20,

    init() {
        this.loadHistory();
        this.setupFilters();
    },

    loadHistory() {
        const history = Storage.getHistory();
        const container = document.getElementById('historyList');
        
        if (!container) return;

        if (history.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-history text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-500">No recognition history yet</p>
                </div>
            `;
            return;
        }

        container.innerHTML = history.map((record, index) => `
            <div class="glass-card p-4 card-hover cursor-pointer" onclick="HistoryView.showDetail(${index})">
                <div class="flex items-center space-x-4">
                    <img src="${record.processedImage || '/static/samples/sample_person1_1.png'}" 
                         class="w-16 h-16 rounded-lg object-cover">
                    <div class="flex-1">
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="font-bold text-lg">Person #${record.classId || 'Unknown'}</h4>
                                <p class="text-sm text-gray-500">
                                    ${new Date(record.timestamp).toLocaleString()}
                                </p>
                            </div>
                            <span class="badge badge-success">${record.confidence}</span>
                        </div>
                        <div class="mt-2 flex items-center space-x-4 text-xs text-gray-400">
                            <span><i class="fas fa-vector-square mr-1"></i>${record.featureCount || 0} features</span>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    },

    showDetail(index) {
        const record = Storage.getHistory()[index];
        if (!record) return;

        const modal = new Modal('Recognition Details', `
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <img src="${record.processedImage || '/static/samples/sample_person1_1.png'}" 
                             class="w-full rounded-lg">
                    </div>
                    <div class="space-y-2">
                        <p><strong>Person ID:</strong> #${record.classId}</p>
                        <p><strong>Confidence:</strong> ${record.confidence}</p>
                        <p><strong>Features:</strong> ${record.featureCount || 0}</p>
                        <p><strong>Time:</strong> ${new Date(record.timestamp).toLocaleString()}</p>
                    </div>
                </div>
            </div>
        `);
        modal.show();
    },

    setupFilters() {
        const searchInput = document.getElementById('historySearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterHistory(e.target.value);
            });
        }
    },

    filterHistory(query) {
        const history = Storage.getHistory();
        const filtered = history.filter(record => 
            record.classId.toString().includes(query) ||
            record.confidence.includes(query)
        );
        
        // Re-render with filtered data
        const container = document.getElementById('historyList');
        if (!container) return;

        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-search text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-500">No matching records found</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filtered.map((record, index) => `
            <div class="glass-card p-4 card-hover cursor-pointer">
                <div class="flex items-center space-x-4">
                    <img src="${record.processedImage || '/static/samples/sample_person1_1.png'}" 
                         class="w-16 h-16 rounded-lg object-cover">
                    <div class="flex-1">
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="font-bold text-lg">Person #${record.classId}</h4>
                                <p class="text-sm text-gray-500">
                                    ${new Date(record.timestamp).toLocaleString()}
                                </p>
                            </div>
                            <span class="badge badge-success">${record.confidence}</span>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    },

    clearHistory() {
        if (confirm('Are you sure you want to clear all history?')) {
            Storage.clearHistory();
            this.loadHistory();
            Toast.success('History cleared');
        }
    }
};

// Database View Module
const DatabaseView = {
    init() {
        this.loadPersons();
    },

    async loadPersons() {
        try {
            const persons = await API.getPersons();
            this.renderPersons(persons);
        } catch (error) {
            console.error('Failed to load persons:', error);
        }
    },

    renderPersons(persons) {
        const container = document.getElementById('personsGrid');
        if (!container) return;

        if (!persons || persons.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12 col-span-full">
                    <i class="fas fa-users text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-500">No persons registered yet</p>
                </div>
            `;
            return;
        }

        container.innerHTML = persons.map(person => `
            <div class="glass-card p-6 card-hover">
                <div class="text-center">
                    <div class="w-24 h-24 mx-auto rounded-full bg-gradient-primary flex items-center justify-center text-white text-3xl font-bold mb-3">
                        ${person.name.charAt(0).toUpperCase()}
                    </div>
                    <h4 class="font-bold text-lg">${person.name}</h4>
                    <p class="text-sm text-gray-500">${person.samples || 0} samples</p>
                    <div class="mt-4 flex justify-center space-x-2">
                        <button class="btn btn-ghost btn-sm" onclick="DatabaseView.editPerson(${person.id})">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-ghost btn-sm text-red-500" onclick="DatabaseView.deletePerson(${person.id})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    },

    showAddPersonForm() {
        const modal = new Modal('Add New Person', `
            <form id="addPersonForm" class="space-y-4">
                <div>
                    <label class="block text-sm font-semibold mb-2">Person Name</label>
                    <input type="text" id="personName" class="input-modern" required>
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-2">Sample Images</label>
                    <input type="file" id="personImages" class="input-modern" multiple accept="image/*" required>
                </div>
                <div class="flex justify-end space-x-2">
                    <button type="button" class="btn btn-ghost" onclick="this.closest('.modal-overlay').remove()">
                        Cancel
                    </button>
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-plus mr-2"></i>Add Person
                    </button>
                </div>
            </form>
        `);
        modal.show();

        document.getElementById('addPersonForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addPerson();
        });
    },

    async addPerson() {
        const name = document.getElementById('personName').value;
        const images = document.getElementById('personImages').files;

        if (!name || images.length === 0) {
            Toast.error('Please fill all fields');
            return;
        }

        showLoading('Adding person...');

        try {
            const formData = new FormData();
            formData.append('name', name);
            for (let img of images) {
                formData.append('images', img);
            }

            const response = await fetch('/api/persons', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                Toast.success('Person added successfully');
                this.loadPersons();
                document.querySelector('.modal-overlay').remove();
            } else {
                Toast.error(result.error || 'Failed to add person');
            }
        } catch (error) {
            Toast.error('Network error');
        } finally {
            hideLoading();
        }
    },

    editPerson(id) {
        Toast.info('Edit functionality coming soon');
    },

    async deletePerson(id) {
        if (!confirm('Are you sure you want to delete this person?')) return;

        try {
            const response = await fetch(`/api/persons/${id}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                Toast.success('Person deleted');
                this.loadPersons();
            } else {
                Toast.error('Failed to delete person');
            }
        } catch (error) {
            Toast.error('Network error');
        }
    }
};

// Analytics View Module
const AnalyticsView = {
    charts: {},

    init() {
        this.loadAnalytics();
    },

    async loadAnalytics() {
        try {
            const stats = await API.getStats();
            const evaluation = await API.getEvaluation();
            this.renderCharts(stats, evaluation);
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    },

    renderCharts(stats, evaluation) {
        this.renderConfidenceDistribution();
        this.renderFeatureContributions();
        
        if (evaluation && evaluation.metrics) {
            this.renderEvaluationMetrics(evaluation.metrics);
        }
    },

    renderEvaluationMetrics(metrics) {
        const container = document.getElementById('evaluationMetrics');
        if (!container) return;
        
        const overall = metrics.overall;
        if (!overall) return;
        
        container.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase">Accuracy</p>
                    <p class="text-2xl font-bold text-gradient mt-2">${(overall.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase">Precision</p>
                    <p class="text-2xl font-bold text-gradient mt-2">${(overall.precision_macro * 100).toFixed(2)}%</p>
                </div>
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase">Recall</p>
                    <p class="text-2xl font-bold text-gradient mt-2">${(overall.recall_macro * 100).toFixed(2)}%</p>
                </div>
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase">F1-Score</p>
                    <p class="text-2xl font-bold text-gradient mt-2">${(overall.f1_macro * 100).toFixed(2)}%</p>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase mb-2">ROC-AUC</p>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <p class="text-sm text-gray-600">Micro Average</p>
                            <p class="text-xl font-bold">${overall.roc_auc_micro ? (overall.roc_auc_micro * 100).toFixed(2) : 'N/A'}%</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600">Macro Average</p>
                            <p class="text-xl font-bold">${overall.roc_auc_macro ? (overall.roc_auc_macro * 100).toFixed(2) : 'N/A'}%</p>
                        </div>
                    </div>
                </div>
                <div class="glass-card p-4">
                    <p class="text-xs text-gray-500 font-semibold uppercase mb-2">Inference Time</p>
                    <p class="text-2xl font-bold text-gradient mt-2">${overall.inference_time_ms.toFixed(2)} ms</p>
                    <p class="text-xs text-gray-500 mt-1">per image</p>
                </div>
            </div>
            
            ${metrics.error_analysis ? `
            <div class="glass-card p-6 mb-8">
                <h4 class="font-bold text-lg mb-4">Error Analysis</h4>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                        <p class="text-xs text-gray-500">Total Samples</p>
                        <p class="text-xl font-bold">${metrics.error_analysis.total_samples}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Correct</p>
                        <p class="text-xl font-bold text-green-600">${metrics.error_analysis.correct_predictions}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Incorrect</p>
                        <p class="text-xl font-bold text-red-600">${metrics.error_analysis.incorrect_predictions}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Error Rate</p>
                        <p class="text-xl font-bold">${(metrics.error_analysis.error_rate * 100).toFixed(2)}%</p>
                    </div>
                </div>
                
                ${metrics.error_analysis.most_confused_pairs && metrics.error_analysis.most_confused_pairs.length > 0 ? `
                <div class="mt-4">
                    <p class="text-sm font-semibold mb-2">Most Confused Pairs:</p>
                    <div class="space-y-2">
                        ${metrics.error_analysis.most_confused_pairs.slice(0, 5).map(pair => `
                            <div class="flex justify-between items-center p-2 bg-gray-50 rounded">
                                <span class="text-sm">${pair.true_class} → ${pair.predicted_class}</span>
                                <span class="badge badge-warning">${pair.count} times</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
            ` : ''}
        `;
    },

    renderConfidenceDistribution() {
        const history = Storage.getHistory();
        if (history.length === 0) return;

        const confidences = history.map(r => parseFloat(r.confidence));
        
        const ctx = document.getElementById('confidenceChart');
        if (!ctx) return;

        if (this.charts.confidence) {
            this.charts.confidence.destroy();
        }

        this.charts.confidence = createChart('confidenceChart', {
            type: 'line',
            data: {
                labels: confidences.map((_, i) => i + 1),
                datasets: [{
                    label: 'Confidence %',
                    data: confidences,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    },

    renderFeatureContributions() {
        const ctx = document.getElementById('featureChart');
        if (!ctx) return;

        if (this.charts.feature) {
            this.charts.feature.destroy();
        }

        this.charts.feature = createChart('featureChart', {
            type: 'radar',
            data: {
                labels: ['SIFT', 'HOG', 'Gabor', 'CNN'],
                datasets: [{
                    label: 'Feature Contribution',
                    data: [15, 85, 10, 90],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
};

// Batch Export Module
const BatchExport = {
    results: [],

    exportResults() {
        if (this.results.length === 0) {
            Toast.error('No results to export');
            return;
        }

        exportAsCSV(this.results, `batch_results_${Date.now()}.csv`);
    },

    addResult(result) {
        this.results.push(result);
    },

    clearResults() {
        this.results = [];
    }
};

// Utility Functions
function animateNumber(element, target, duration = 1000) {
    const start = parseInt(element.textContent) || 0;
    const increment = (target - start) / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= target) || (increment < 0 && current <= target)) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    // Setup navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) {
                Navigation.navigateTo(view);
            }
        });
    });

    // Initialize dashboard
    Dashboard.init();

    console.log('Hybrid Face Recognition System initialized');
});
