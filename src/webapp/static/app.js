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

let maxUploadFiles = 30;
let uploadMethod = 'file';
let isUploadModalOpen = false;
let availableLabels = [];
let selectedExistingLabel = '';
let isCreateNewMode = false;

let recordPreviewStream = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordedBlob = null;
let recordedAccepted = false;
let previewObjectURL = null;


function toggleUploadModal(show) {
    const modal = document.getElementById('uploadModal');
    if (!modal) {
        return;
    }

    isUploadModalOpen = show;

    modal.classList.toggle('hidden', !show);
    modal.setAttribute('aria-hidden', show ? 'false' : 'true');

    if (show) {
        updateMethodUI('file');
        exitCreateNewMode();
        selectedExistingLabel = '';
        updateSelectedLabelText();
        loadUploadOptions();
    } else {
        closeLabelDropdown();
        stopPreviewStream();
        clearRecordingState(true);
        updateMethodUI('file');
    }
}


function updateMethodUI(method) {
    uploadMethod = method;
    const toggleBtn = document.getElementById('toggleRecordBtn');

    const fileField = document.querySelector('.upload-files');
    const fileInput = document.getElementById('videoFiles');
    const recordPanel = document.getElementById('recordPanel');
    const uploadBtn = document.getElementById('uploadBtn');

    if (fileField) {
        fileField.classList.toggle('hidden', method !== 'file');
    }
    if (fileInput) {
        fileInput.disabled = method !== 'file';
    }
    if (recordPanel) {
        recordPanel.classList.toggle('hidden', method !== 'record');
    }
    if (toggleBtn) {
        toggleBtn.classList.toggle('active', method === 'record');
        toggleBtn.textContent = method === 'record' ? 'Use File Upload' : 'Record Now';
    }
    if (uploadBtn) {
        uploadBtn.textContent = method === 'record' ? 'Upload Recorded Video' : 'Upload Videos';
    }

    if (method === 'record') {
        prepareRecordPreview();
    } else {
        stopPreviewStream();
        clearRecordingState(false);
    }
}


function updateSelectedLabelText() {
    const selectedLabelText = document.getElementById('selectedLabelText');
    if (!selectedLabelText) {
        return;
    }
    if (selectedExistingLabel) {
        selectedLabelText.textContent = selectedExistingLabel;
    } else if (isCreateNewMode) {
        selectedLabelText.textContent = 'Create New Sign';
    } else {
        selectedLabelText.textContent = 'Select a sign';
    }
}


function closeLabelDropdown() {
    const panel = document.getElementById('labelDropdownPanel');
    const trigger = document.getElementById('labelDropdownTrigger');
    if (panel) panel.classList.add('hidden');
    if (trigger) trigger.setAttribute('aria-expanded', 'false');
}


function openLabelDropdown() {
    const panel = document.getElementById('labelDropdownPanel');
    const trigger = document.getElementById('labelDropdownTrigger');
    if (!panel || !trigger || isCreateNewMode) {
        return;
    }
    panel.classList.remove('hidden');
    trigger.setAttribute('aria-expanded', 'true');
    const searchInput = document.getElementById('labelSearchInput');
    if (searchInput) {
        searchInput.value = '';
        renderLabelOptions('');
        searchInput.focus();
    }
}


function renderLabelOptions(filterText = '') {
    const optionsList = document.getElementById('labelOptionsList');
    if (!optionsList) {
        return;
    }

    const keyword = (filterText || '').trim().toLowerCase();
    const filtered = availableLabels.filter((label) => label.toLowerCase().includes(keyword));

    optionsList.innerHTML = '';

    const createBtn = document.createElement('button');
    createBtn.type = 'button';
    createBtn.className = 'label-option create';
    createBtn.textContent = 'Create a new sign';
    createBtn.addEventListener('click', () => {
        activateCreateNewMode();
        closeLabelDropdown();
    });
    optionsList.appendChild(createBtn);

    filtered.forEach((label) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'label-option';
        button.textContent = label;
        button.addEventListener('click', () => {
            selectedExistingLabel = label;
            exitCreateNewMode();
            updateSelectedLabelText();
            closeLabelDropdown();
        });
        optionsList.appendChild(button);
    });
}


function activateCreateNewMode() {
    isCreateNewMode = true;
    selectedExistingLabel = '';
    updateSelectedLabelText();

    const trigger = document.getElementById('labelDropdownTrigger');
    const wrap = document.getElementById('newSignWrap');
    const newLabelInput = document.getElementById('newLabel');
    if (trigger) {
        trigger.classList.add('disabled');
        trigger.disabled = true;
    }
    if (wrap) {
        wrap.classList.remove('hidden');
    }
    if (newLabelInput) {
        newLabelInput.value = '';
        newLabelInput.focus();
    }
}


function exitCreateNewMode() {
    isCreateNewMode = false;
    const trigger = document.getElementById('labelDropdownTrigger');
    const wrap = document.getElementById('newSignWrap');
    if (trigger) {
        trigger.classList.remove('disabled');
        trigger.disabled = false;
    }
    if (wrap) {
        wrap.classList.add('hidden');
    }
    updateSelectedLabelText();
}


function setRecordStatus(text, isError = false) {
    const el = document.getElementById('recordStatus');
    if (!el) {
        return;
    }
    el.textContent = text;
    el.style.color = isError ? '#c23232' : '#39527d';
}


function clearRecordingState(resetPreview = false) {
    recordedBlob = null;
    recordedAccepted = false;
    recordingChunks = [];

    const stopBtn = document.getElementById('stopRecordBtn');
    const retryBtn = document.getElementById('retryRecordBtn');
    const acceptBtn = document.getElementById('acceptRecordBtn');
    const startBtn = document.getElementById('startRecordBtn');

    if (stopBtn) stopBtn.disabled = true;
    if (retryBtn) retryBtn.disabled = true;
    if (acceptBtn) acceptBtn.disabled = true;
    if (startBtn) startBtn.disabled = false;

    const preview = document.getElementById('recordPreview');
    if (preview && resetPreview) {
        if (previewObjectURL) {
            URL.revokeObjectURL(previewObjectURL);
            previewObjectURL = null;
        }
        preview.removeAttribute('src');
        preview.srcObject = null;
        preview.controls = false;
        preview.style.transform = 'scaleX(1)';
    }
}


function stopPreviewStream() {
    if (recordPreviewStream) {
        recordPreviewStream.getTracks().forEach((track) => track.stop());
        recordPreviewStream = null;
    }
}


async function prepareRecordPreview() {
    const preview = document.getElementById('recordPreview');
    if (!preview) {
        return;
    }

    try {
        stopPreviewStream();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        recordPreviewStream = stream;

        preview.srcObject = stream;
        preview.controls = false;
        preview.muted = true;
        preview.style.transform = 'scaleX(-1)';
        await preview.play();
        setRecordStatus('Ready to record.');
    } catch (error) {
        console.error('Record preview error:', error);
        setRecordStatus('Cannot access camera for recording.', true);
    }
}


function startRecording() {
    const preview = document.getElementById('recordPreview');
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const retryBtn = document.getElementById('retryRecordBtn');
    const acceptBtn = document.getElementById('acceptRecordBtn');

    if (!preview || !preview.srcObject) {
        setRecordStatus('Camera is not ready for recording.', true);
        return;
    }

    recordedBlob = null;
    recordedAccepted = false;
    recordingChunks = [];

    try {
        const options = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? { mimeType: 'video/webm;codecs=vp9' }
            : { mimeType: 'video/webm' };

        mediaRecorder = new MediaRecorder(preview.srcObject, options);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordingChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordingChunks, { type: 'video/webm' });
            preview.srcObject = null;
            if (previewObjectURL) {
                URL.revokeObjectURL(previewObjectURL);
            }
            previewObjectURL = URL.createObjectURL(recordedBlob);
            preview.src = previewObjectURL;
            preview.controls = true;
            preview.muted = false;
            preview.style.transform = 'scaleX(1)';
            preview.play();

            if (retryBtn) retryBtn.disabled = false;
            if (acceptBtn) acceptBtn.disabled = false;
            setRecordStatus('Review your recording, then click Accept Clip.');
        };

        mediaRecorder.start();
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (retryBtn) retryBtn.disabled = true;
        if (acceptBtn) acceptBtn.disabled = true;
        setRecordStatus('Recording... click Stop when finished.');
    } catch (error) {
        console.error('Start recording error:', error);
        setRecordStatus('Cannot start recording.', true);
    }
}


function stopRecording() {
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
}


function retryRecording() {
    clearRecordingState(true);
    prepareRecordPreview();
    setRecordStatus('Ready to record again.');
}


function acceptRecording() {
    if (!recordedBlob) {
        setRecordStatus('No recording available to accept.', true);
        return;
    }
    recordedAccepted = true;
    setRecordStatus('Recording accepted. You can now upload it.');
}


async function loadUploadOptions() {
    try {
        const res = await fetch('/api/upload-options');
        if (!res.ok) {
            throw new Error(`Cannot load upload options (${res.status})`);
        }

        const payload = await res.json();
        if (!document.getElementById('labelOptionsList')) {
            return;
        }

        availableLabels = payload.labels || [];
        renderLabelOptions('');

        maxUploadFiles = payload.max_upload_files || 30;
    } catch (error) {
        console.error('Upload options error:', error);
        setUploadResult('Cannot load upload options from server.', true);
    }
}


function setUploadResult(message, isError = false) {
    const resultEl = document.getElementById('uploadResult');
    if (!resultEl) {
        return;
    }
    resultEl.textContent = message;
    resultEl.className = isError ? 'upload-result error' : 'upload-result success';
}


async function handleUpload() {
    const selectedLabel = selectedExistingLabel;
    const newLabel = document.getElementById('newLabel')?.value || '';
    const fileInput = document.getElementById('videoFiles');
    const files = fileInput?.files;

    const finalLabel = isCreateNewMode ? newLabel.trim() : selectedLabel.trim();

    if (!finalLabel) {
        setUploadResult('Please choose an existing sign or create a new one.', true);
        return;
    }

    if (uploadMethod === 'file' && (!files || files.length === 0)) {
        setUploadResult('Please select at least one video file.', true);
        return;
    }

    if (uploadMethod === 'record') {
        if (!recordedBlob) {
            setUploadResult('Please record a video first.', true);
            return;
        }
        if (!recordedAccepted) {
            setUploadResult('Please review and accept your recording before uploading.', true);
            return;
        }
    }

    if (uploadMethod === 'file' && files.length > maxUploadFiles) {
        setUploadResult(`Too many files. Maximum is ${maxUploadFiles}.`, true);
        return;
    }

    const formData = new FormData();
    formData.append('selected_label', isCreateNewMode ? '' : selectedLabel);
    formData.append('new_label', isCreateNewMode ? finalLabel : '');

    if (uploadMethod === 'file') {
        for (const file of files) {
            formData.append('videos', file);
        }
    } else {
        const recordedFile = new File(
            [recordedBlob],
            `recorded_${Date.now()}.webm`,
            { type: 'video/webm' }
        );
        formData.append('videos', recordedFile);
    }

    setUploadResult('Uploading your contribution...', false);

    try {
        const response = await fetch('/api/upload-videos', {
            method: 'POST',
            body: formData,
        });
        const payload = await response.json();

        if (!response.ok || !payload.ok) {
            throw new Error(payload.message || 'Upload failed');
        }

        setUploadResult(
            `Uploaded ${payload.uploaded_count} file(s) to label "${payload.label}". ` +
            `Skipped: ${payload.skipped_count}, Failed: ${payload.failed_count}`,
            false
        );
        if (fileInput) {
            fileInput.value = '';
        }
        document.getElementById('newLabel').value = '';
        selectedExistingLabel = '';
        exitCreateNewMode();
        updateSelectedLabelText();
        if (uploadMethod === 'record') {
            clearRecordingState(true);
            prepareRecordPreview();
        }
        await loadUploadOptions();
    } catch (error) {
        console.error('Upload error:', error);
        setUploadResult(`Upload failed: ${error.message}`, true);
    }
}

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

    // Pause recognition stream while contribution modal is open.
    if (isUploadModalOpen) {
        requestAnimationFrame(captureFrames);
        return;
    }
    
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

    const openUploadBtn = document.getElementById('openUploadBtn');
    const closeUploadBtn = document.getElementById('closeUploadBtn');
    const uploadBackdrop = document.getElementById('uploadModalBackdrop');
    const uploadBtn = document.getElementById('uploadBtn');
    const toggleRecordBtn = document.getElementById('toggleRecordBtn');
    const labelDropdownTrigger = document.getElementById('labelDropdownTrigger');
    const labelSearchInput = document.getElementById('labelSearchInput');
    const labelDropdownPanel = document.getElementById('labelDropdownPanel');
    const backToListBtn = document.getElementById('backToListBtn');
    const startRecordBtn = document.getElementById('startRecordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const retryRecordBtn = document.getElementById('retryRecordBtn');
    const acceptRecordBtn = document.getElementById('acceptRecordBtn');

    if (openUploadBtn) {
        openUploadBtn.addEventListener('click', function () {
            toggleUploadModal(true);
        });
    }

    if (closeUploadBtn) {
        closeUploadBtn.addEventListener('click', function () {
            toggleUploadModal(false);
        });
    }

    if (uploadBackdrop) {
        uploadBackdrop.addEventListener('click', function () {
            toggleUploadModal(false);
        });
    }

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            toggleUploadModal(false);
        }
    });

    if (uploadBtn) {
        uploadBtn.addEventListener('click', handleUpload);
    }

    if (toggleRecordBtn) {
        toggleRecordBtn.addEventListener('click', () => {
            updateMethodUI(uploadMethod === 'record' ? 'file' : 'record');
        });
    }

    if (labelDropdownTrigger) {
        labelDropdownTrigger.addEventListener('click', () => {
            const isOpen = !labelDropdownPanel.classList.contains('hidden');
            if (isOpen) {
                closeLabelDropdown();
            } else {
                openLabelDropdown();
            }
        });
    }

    if (labelSearchInput) {
        labelSearchInput.addEventListener('input', (event) => {
            renderLabelOptions(event.target.value || '');
        });
    }

    if (backToListBtn) {
        backToListBtn.addEventListener('click', () => {
            exitCreateNewMode();
            openLabelDropdown();
        });
    }

    document.addEventListener('click', (event) => {
        const selector = document.getElementById('labelSelector');
        if (!selector || selector.contains(event.target)) {
            return;
        }
        closeLabelDropdown();
    });

    if (startRecordBtn) {
        startRecordBtn.addEventListener('click', startRecording);
    }
    if (stopRecordBtn) {
        stopRecordBtn.addEventListener('click', stopRecording);
    }
    if (retryRecordBtn) {
        retryRecordBtn.addEventListener('click', retryRecording);
    }
    if (acceptRecordBtn) {
        acceptRecordBtn.addEventListener('click', acceptRecording);
    }
    
    initializeCamera();
    updateSelectedLabelText();
});

window.addEventListener('beforeunload', function () {
    stopPreviewStream();
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
});
