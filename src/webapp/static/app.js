/*
 * Vietnamese Sign Language Recognition - Web App (OPTIMIZED)
 * Real-time frame capture with full debug logging
 */

// Initialize smooth scroll snap behavior
window.addEventListener('load', function() {
    // Ensure page starts at top
    window.scrollTo(0, 0);
    // Enable scroll snap
    document.documentElement.style.scrollSnapType = 'y mandatory';
});

const socket = io();
let video = null;
let canvas = null;
let ctx = null;

let isConnected = false;
let frameCount = 0;
let totalPredictions = 0;
let framesSent = 0;
let recognizedSequence = []; // Track all recognized gestures

// const FPS = 10;
const FPS = 25;
const FRAME_INTERVAL = 1000 / FPS;
let lastFrameTime = 0;

// State tracking
let currentState = {
    buffer_size: 0,
    is_ready: false,
    is_inferring: false,
};

// ===================================================================
// SOCKET.IO EVENTS
// ===================================================================

socket.on('connect', function () {
    console.log('✅ Connected to server');
    isConnected = true;
    updateStatus('🟢 Connected', 'status-connected');
});

socket.on('connect_error', function (error) {
    console.error('❌ Connection error:', error);
    updateStatus('🔴 Connection Error', 'status-error');
});

socket.on('disconnect', function () {
    console.log('❌ Disconnected from server');
    isConnected = false;
    updateStatus('🔴 Disconnected', 'status-error');
});

socket.on('prediction', function (data) {
    console.log('🎉 PREDICTION RECEIVED:', data);
    
    const label = data.label;
    const confidence = (data.confidence * 100).toFixed(1);
    const votes = data.votes || '?';
    const buffer = data.buffer_size || '?';
    
    // Update UI
    document.getElementById('prediction-label').textContent = `${label}`;
    document.getElementById('prediction-confidence').textContent = `Confidence: ${confidence}%`;
    document.getElementById('prediction-frames').textContent = `Votes: ${votes} | Buffer: ${buffer}`;
    
    totalPredictions++;
    document.getElementById('total-predictions').textContent = totalPredictions;
    
    addToHistory(label, confidence);
    
    // Flash effect
    document.querySelector('.prediction-box').style.backgroundColor = '#4CAF50';
    setTimeout(() => {
        document.querySelector('.prediction-box').style.backgroundColor = '';
    }, 500);
});

socket.on('status', function (data) {
    console.log('📊 STATUS UPDATE:', data);
    currentState = data;
    updateStatusDisplay();
});

// ===================================================================
// UI FUNCTIONS
// ===================================================================

function updateStatus(text, cssClass) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = text;
    statusEl.className = cssClass;
}

function updateStatusDisplay() {
    const stateEl = document.getElementById('fsm-state');
    const bufferSize = currentState.buffer_size || 0;
    const isReady = currentState.is_ready;
    const isInferring = currentState.is_inferring;
    
    let statusText = '';
    let statusClass = 'state-waiting';
    
    if (isInferring) {
        statusText = 'Inferring...';
        statusClass = 'state-recording';
    } else if (isReady) {
        statusText = 'Ready';
        statusClass = 'state-recording';
    } else {
        statusText = `Buffer: ${bufferSize}/10`;
        statusClass = 'state-waiting';
    }
    
    stateEl.textContent = statusText;
    stateEl.className = statusClass;
    
    document.getElementById('segment-size').textContent = bufferSize;
    document.getElementById('still-count').textContent = isInferring ? 'Processing' : 'Ready';
}

function addToHistory(label, confidence) {
    const history = document.getElementById('history');
    const entry = document.createElement('div');
    entry.className = 'history-entry';
    
    const time = new Date().toLocaleTimeString();
    entry.textContent = `${time} - ${label} (${confidence}%)`;
    
    history.insertBefore(entry, history.firstChild);
    
    while (history.children.length > 10) {
        history.removeChild(history.lastChild);
    }
    
    // Update recognized sequence summary
    recognizedSequence.push(label);
    updateSummary();
}

function updateSummary() {
    const summaryEl = document.getElementById('summary-content');
    if (recognizedSequence.length === 0) {
        summaryEl.textContent = 'No gestures recognized yet';
    } else {
        summaryEl.textContent = recognizedSequence.join(' - ');
    }
}

// ===================================================================
// CAMERA INITIALIZATION
// ===================================================================

async function initializeCamera() {
    console.log('📹 Requesting camera access...');
    
    try {
        video = document.getElementById('videoElement');
        canvas = document.getElementById('canvasOutput');
        ctx = canvas.getContext('2d');
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        console.log('✅ Camera accessed');
        video.srcObject = stream;
        
        video.onloadedmetadata = function () {
            console.log(`✅ Camera initialized (${video.videoWidth}x${video.videoHeight})`);
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            console.log('🎬 Starting frame capture at 10 FPS...');
            captureFrames();
        };
        
    } catch (error) {
        console.error('❌ Camera error:', error);
        updateStatus('🔴 Camera Error', 'status-error');
    }
}

// ===================================================================
// FRAME CAPTURE & SENDING
// ===================================================================

function captureFrames() {
    const now = Date.now();
    
    // Capture every FRAME_INTERVAL ms (determined by FPS setting)
    if (now - lastFrameTime >= FRAME_INTERVAL) {
        if (isConnected && video && video.readyState === video.HAVE_ENOUGH_DATA) {
            try {
                // Draw frame to canvas
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
                ctx.restore();
                
                // Convert to JPEG (smaller size)
                const imageData = canvas.toDataURL('image/jpeg', 0.7);
                
                // Send frame with metadata
                socket.emit('frame', {
                    image: imageData,
                    frame_num: frameCount,
                    timestamp: now
                });
                
                framesSent++;
                frameCount++;
                lastFrameTime = now;
                
                // Debug log every 10 frames
                if (frameCount % 10 === 0) {
                    console.log(`📤 Sent ${framesSent} frames (frame #${frameCount})`);
                }
                
            } catch (e) {
                console.error('Canvas error:', e);
            }
        }
    }
    
    // Request status from server every 5 frames
    if (frameCount % 5 === 0 && isConnected) {
        socket.emit('status');
    }
    
    // Continue loop
    requestAnimationFrame(captureFrames);
}

// ===================================================================
// PAGE INITIALIZATION
// ===================================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('Vietnamese Sign Language Recognition - Web App');
    console.log('Configuration: 10 FPS, 15-frame buffer, 55% min confidence');
    
    initializeCamera();
});

window.addEventListener('beforeunload', function () {
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
});
