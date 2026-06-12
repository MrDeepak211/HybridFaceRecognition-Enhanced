// ==========================================
// Hybrid Face Recognition System
// Camera/Webcam Module
// ==========================================

const Camera = {
    stream: null,
    videoElement: null,
    canvasElement: null,
    isStreaming: false,
    currentFacingMode: 'user', // 'user' for front, 'environment' for back

    init() {
        this.videoElement = document.getElementById('cameraVideo');
        this.canvasElement = document.getElementById('cameraCanvas');
        
        if (this.videoElement) {
            this.setupCameraControls();
        }
    },

    setupCameraControls() {
        const startBtn = document.getElementById('startCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const switchBtn = document.getElementById('switchCameraBtn');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.start());
        }

        if (captureBtn) {
            captureBtn.addEventListener('click', () => this.capture());
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stop());
        }

        if (switchBtn) {
            switchBtn.addEventListener('click', () => this.switchCamera());
        }
    },

    async start() {
        try {
            showLoading('Starting camera...');

            const constraints = {
                video: {
                    facingMode: this.currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (this.videoElement) {
                this.videoElement.srcObject = this.stream;
                this.videoElement.classList.remove('hidden');
            }

            this.isStreaming = true;
            
            // Update UI
            this.updateCameraUI(true);
            Toast.success('Camera started successfully');

        } catch (error) {
            console.error('Camera start error:', error);
            
            let errorMessage = 'Failed to access camera';
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Camera permission denied. Please allow camera access in your browser settings.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No camera found on your device.';
            }
            
            Toast.error(errorMessage);
        } finally {
            hideLoading();
        }
    },

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement.classList.add('hidden');
        }

        this.isStreaming = false;
        this.updateCameraUI(false);
        Toast.info('Camera stopped');
    },

    capture() {
        if (!this.isStreaming || !this.videoElement || !this.canvasElement) {
            Toast.error('Camera is not running');
            return;
        }

        // Set canvas dimensions to match video
        this.canvasElement.width = this.videoElement.videoWidth;
        this.canvasElement.height = this.videoElement.videoHeight;

        // Draw current frame to canvas
        const context = this.canvasElement.getContext('2d');
        context.drawImage(this.videoElement, 0, 0);

        // Get image data as base64
        const imageData = this.canvasElement.toDataURL('image/png');

        // Display captured image
        this.displayCapturedImage(imageData);

        // Send to server for prediction
        this.processCapturedImage(imageData);

        Toast.success('Image captured!');
    },

    displayCapturedImage(imageData) {
        const previewElement = document.getElementById('capturedPreview');
        if (previewElement) {
            previewElement.src = imageData;
            previewElement.classList.remove('hidden');
        }

        const previewContainer = document.getElementById('previewContainer');
        if (previewContainer) {
            previewContainer.classList.remove('hidden');
        }
    },

    async processCapturedImage(imageData) {
        showLoading('Analyzing face...');

        try {
            const result = await API.cameraCapture(imageData);
            
            if (result.error) {
                Toast.error(result.error);
                return;
            }

            // Display results
            this.displayRecognitionResult(result);

        } catch (error) {
            console.error('Processing error:', error);
            Toast.error('Failed to process image');
        } finally {
            hideLoading();
        }
    },

    displayRecognitionResult(result) {
        const resultContainer = document.getElementById('cameraResult');
        if (!resultContainer) return;

        resultContainer.classList.remove('hidden');
        resultContainer.classList.add('animate-slide-up');

        // Update result elements
        const identityEl = document.getElementById('cameraIdentity');
        const confidenceEl = document.getElementById('cameraConfidence');
        const vectorEl = document.getElementById('cameraVector');

        if (identityEl) {
            identityEl.textContent = `Person #${result.class_id}`;
        }

        if (confidenceEl) {
            confidenceEl.textContent = result.confidence;
        }

        if (vectorEl) {
            vectorEl.textContent = result.feature_count;
        }

        // Update processed image
        const procImage = document.getElementById('cameraProcImage');
        if (procImage && result.processed_image) {
            procImage.src = `/${result.processed_image}?${Date.now()}`;
        }

        // Add to history
        Storage.addToHistory({
            classId: result.class_id,
            confidence: result.confidence,
            processedImage: result.processed_image,
            featureCount: result.feature_count,
            source: 'camera'
        });
    },

    async switchCamera() {
        this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
        
        if (this.isStreaming) {
            this.stop();
            await this.start();
        }

        Toast.info(`Switched to ${this.currentFacingMode === 'user' ? 'front' : 'back'} camera`);
    },

    updateCameraUI(isActive) {
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        const switchBtn = document.getElementById('switchCameraBtn');

        if (startBtn) {
            startBtn.disabled = isActive;
        }

        if (stopBtn) {
            stopBtn.disabled = !isActive;
        }

        if (captureBtn) {
            captureBtn.disabled = !isActive;
        }

        if (switchBtn) {
            switchBtn.disabled = !isActive;
        }
    },

    // Auto-capture mode with timer
    startAutoCapture(interval = 3000) {
        if (!this.isStreaming) {
            Toast.error('Please start camera first');
            return;
        }

        this.autoCaptureInterval = setInterval(() => {
            this.capture();
        }, interval);

        Toast.info(`Auto-capture started (every ${interval/1000}s)`);
    },

    stopAutoCapture() {
        if (this.autoCaptureInterval) {
            clearInterval(this.autoCaptureInterval);
            this.autoCaptureInterval = null;
            Toast.info('Auto-capture stopped');
        }
    }
};

// Initialize camera module when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    Camera.init();
});
