/**
 * Phase Annotation Tool — Canvas-based timeline with phase overlay.
 *
 * Renders: speed (tool1/tool2), jaw angle, inter-tool distance
 * Overlays: algorithm-predicted phase bars with draggable boundaries
 * Interactions: mark correct/incorrect, adjust boundaries, add phases
 */

// ── State ──────────────────────────────────────────────────────────────────
let templates = [];
let currentIdx = -1;
let templateData = null;   // { template, signals }
let selectedPhaseIdx = -1;
let dirty = false;

// Canvas state
let canvas, ctx;
const PADDING = { top: 10, bottom: 30, left: 55, right: 20 };
const SIGNAL_HEIGHT = 80;
const SIGNAL_GAP = 6;
const PHASE_BAR_HEIGHT = 28;
const PHASE_BAR_GAP = 4;
let zoom = 1.0;
let panOffset = 0;  // in pixels
let isDragging = false;
let dragType = null;  // 'pan', 'boundary-start', 'boundary-end'
let dragPhaseIdx = -1;
let dragStartX = 0;
let dragStartPan = 0;

// Phase colors
const PHASE_COLORS = {
    reach:    { bg: '#2E7D32', fg: '#A5D6A7', label: 'Reach' },
    nudge:    { bg: '#E65100', fg: '#FFCC80', label: 'Nudge' },
    grasp:    { bg: '#F9A825', fg: '#FFF59D', label: 'Grasp' },
    transfer: { bg: '#E65100', fg: '#FFCC80', label: 'Transfer' },
    place:    { bg: '#6A1B9A', fg: '#CE93D8', label: 'Place' },
    return:   { bg: '#1565C0', fg: '#90CAF9', label: 'Return' },
};

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    canvas = document.getElementById('timelineCanvas');
    ctx = canvas.getContext('2d');

    // Build legend
    const legendEl = document.getElementById('legend');
    for (const [phase, colors] of Object.entries(PHASE_COLORS)) {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<div class="legend-swatch" style="background:${colors.bg}"></div>${colors.label}`;
        legendEl.appendChild(item);
    }

    // Load template list
    const resp = await fetch('/api/phase/templates');
    templates = await resp.json();

    const sel = document.getElementById('templateSelect');
    templates.forEach((t, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        const status = t.complete ? ' [DONE]' : ` [${t.n_validated}/${t.n_predictions}]`;
        opt.textContent = `${t.trial_key}${status}`;
        sel.appendChild(opt);
    });

    updateProgress();
    setupCanvasEvents();
    setupKeyboard();
    handleResize();
    window.addEventListener('resize', handleResize);
});

function updateProgress() {
    const done = templates.filter(t => t.complete).length;
    document.getElementById('progressDone').textContent = done;
    document.getElementById('progressTotal').textContent = templates.length;
}

// ── Template loading ───────────────────────────────────────────────────────
async function loadTemplate(idx) {
    idx = parseInt(idx);
    if (isNaN(idx) || idx < 0) return;

    currentIdx = idx;
    const t = templates[idx];

    document.getElementById('loadingMsg').style.display = 'block';
    document.getElementById('loadingMsg').textContent = 'Loading kinematic signals...';
    document.getElementById('content').style.display = 'none';

    const resp = await fetch(`/api/phase/template/${encodeURIComponent(t.filename)}`);
    templateData = await resp.json();

    document.getElementById('loadingMsg').style.display = 'none';
    document.getElementById('content').style.display = 'block';

    // Info bar
    document.getElementById('infoDataset').textContent = `Dataset: ${t.dataset}`;
    document.getElementById('infoSkill').textContent = `Skill: ${t.skill_category}`;
    document.getElementById('infoFrames').textContent = `Frames: ${t.n_frames}`;
    document.getElementById('infoTime').textContent = `Time: ${t.total_time_s.toFixed(1)}s`;
    document.getElementById('infoPredictions').textContent = `Phases: ${t.n_predictions}`;

    // Nav buttons
    document.getElementById('btnPrev').disabled = (idx <= 0);
    document.getElementById('btnNext').disabled = (idx >= templates.length - 1);

    selectedPhaseIdx = -1;
    dirty = false;
    document.getElementById('saveStatus').textContent = '';
    resetZoom();
    buildValidationRows();
    drawTimeline();
}

function navTemplate(delta) {
    const newIdx = currentIdx + delta;
    if (newIdx >= 0 && newIdx < templates.length) {
        document.getElementById('templateSelect').value = newIdx;
        loadTemplate(newIdx);
    }
}

// ── Validation rows ────────────────────────────────────────────────────────
function buildValidationRows() {
    const container = document.getElementById('valRows');
    container.innerHTML = '';
    if (!templateData) return;

    const preds = templateData.template.algorithm_predictions;
    preds.forEach((p, i) => {
        const row = document.createElement('div');
        row.className = 'val-row';
        if (p.correct === true) row.classList.add('correct');
        if (p.correct === false) row.classList.add('incorrect');
        row.dataset.idx = i;
        row.onclick = () => selectPhase(i);

        const colors = PHASE_COLORS[p.phase] || { bg: '#555', label: p.phase };
        const statusText = p.correct === true ? 'Correct' : p.correct === false ? 'Wrong' : '—';
        const statusColor = p.correct === true ? '#4CAF50' : p.correct === false ? '#f44336' : '#757575';

        row.innerHTML = `
            <span style="color:#a0a0c0">${i}</span>
            <span><span class="phase-badge" style="background:${colors.bg}">${colors.label}</span></span>
            <span>${p.cycle_index}</span>
            <span>${p.start_frame}–${p.end_frame}</span>
            <span>${(p.confidence * 100).toFixed(0)}%</span>
            <span style="color:${statusColor};font-weight:600">${statusText}</span>
            <span>
                <button class="action-btn btn-correct" onclick="event.stopPropagation();markPhase(${i},true)">&#10003;</button>
                <button class="action-btn btn-incorrect" onclick="event.stopPropagation();markPhase(${i},false)">&#10007;</button>
                <button class="action-btn btn-reset" onclick="event.stopPropagation();markPhase(${i},null)">R</button>
            </span>
            <span style="font-size:11px;color:#a0a0c0">
                <button class="action-btn btn-reset" onclick="event.stopPropagation();adjustBoundary(${i},'start',-5)" title="Start -5">&larr;S</button>
                <button class="action-btn btn-reset" onclick="event.stopPropagation();adjustBoundary(${i},'start',5)" title="Start +5">S&rarr;</button>
                <button class="action-btn btn-reset" onclick="event.stopPropagation();adjustBoundary(${i},'end',-5)" title="End -5">&larr;E</button>
                <button class="action-btn btn-reset" onclick="event.stopPropagation();adjustBoundary(${i},'end',5)" title="End +5">E&rarr;</button>
            </span>
        `;
        container.appendChild(row);
    });
}

function selectPhase(idx) {
    selectedPhaseIdx = idx;
    // Update row highlights
    document.querySelectorAll('.val-row').forEach(r => r.classList.remove('selected'));
    const row = document.querySelector(`.val-row[data-idx="${idx}"]`);
    if (row) {
        row.classList.add('selected');
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    // Scroll canvas to show this phase
    if (templateData) {
        const pred = templateData.template.algorithm_predictions[idx];
        const nFrames = templateData.template.n_frames;
        const drawWidth = (canvas.width - PADDING.left - PADDING.right) * zoom;
        const frameX = (pred.start_frame / nFrames) * drawWidth;
        const visibleWidth = canvas.width - PADDING.left - PADDING.right;
        if (frameX + panOffset < 0 || frameX + panOffset > visibleWidth) {
            panOffset = -frameX + visibleWidth * 0.2;
            panOffset = Math.min(0, Math.max(panOffset, -(drawWidth - visibleWidth)));
        }
    }
    drawTimeline();
}

function markPhase(idx, value) {
    if (!templateData) return;
    templateData.template.algorithm_predictions[idx].correct = value;
    dirty = true;
    document.getElementById('saveStatus').textContent = '(unsaved changes)';
    buildValidationRows();
    if (selectedPhaseIdx === idx) selectPhase(idx);
    drawTimeline();
    // Update local template list status
    const preds = templateData.template.algorithm_predictions;
    templates[currentIdx].n_validated = preds.filter(p => p.correct !== null).length;
    templates[currentIdx].complete = templates[currentIdx].n_validated === preds.length;
    updateProgress();
}

function markAllCorrect() {
    if (!templateData) return;
    templateData.template.algorithm_predictions.forEach(p => {
        if (p.correct === null) p.correct = true;
    });
    dirty = true;
    document.getElementById('saveStatus').textContent = '(unsaved changes)';
    buildValidationRows();
    drawTimeline();
    const preds = templateData.template.algorithm_predictions;
    templates[currentIdx].n_validated = preds.length;
    templates[currentIdx].complete = true;
    updateProgress();
}

function adjustBoundary(idx, which, delta) {
    if (!templateData) return;
    const pred = templateData.template.algorithm_predictions[idx];
    const maxFrame = templateData.template.n_frames - 1;
    if (which === 'start') {
        pred.start_frame = Math.max(0, Math.min(pred.end_frame - 1, pred.start_frame + delta));
    } else {
        pred.end_frame = Math.max(pred.start_frame + 1, Math.min(maxFrame, pred.end_frame + delta));
    }
    dirty = true;
    document.getElementById('saveStatus').textContent = '(unsaved changes)';
    buildValidationRows();
    drawTimeline();
}

// ── Save ───────────────────────────────────────────────────────────────────
async function saveTemplate() {
    if (!templateData || currentIdx < 0) return;
    const btn = document.getElementById('saveBtn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

    const payload = {
        algorithm_predictions: templateData.template.algorithm_predictions,
        manual_corrections: templateData.template.manual_corrections || [],
    };

    // Also save to localStorage as backup
    backupToLocalStorage();

    const resp = await fetch(`/api/phase/template/${encodeURIComponent(templates[currentIdx].filename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    const result = await resp.json();
    btn.disabled = false;
    btn.textContent = 'Save';

    if (result.status === 'saved') {
        dirty = false;
        document.getElementById('saveStatus').textContent = 'Saved successfully';
        setTimeout(() => {
            if (!dirty) document.getElementById('saveStatus').textContent = '';
        }, 3000);
    } else {
        document.getElementById('saveStatus').textContent = `Error: ${result.error || 'unknown'}`;
    }

    // Update dropdown text
    const sel = document.getElementById('templateSelect');
    const opt = sel.options[currentIdx + 1]; // +1 for placeholder
    const t = templates[currentIdx];
    const status = t.complete ? ' [DONE]' : ` [${t.n_validated}/${t.n_predictions}]`;
    opt.textContent = `${t.trial_key}${status}`;
}

// ── LocalStorage backup ─────────────────────────────────────────────────────
function getStorageKey() {
    if (currentIdx < 0 || !templates[currentIdx]) return null;
    return `phase_backup_${templates[currentIdx].filename}`;
}

function backupToLocalStorage() {
    if (!templateData || currentIdx < 0) return;
    const key = getStorageKey();
    if (!key) return;
    const backup = {
        timestamp: new Date().toISOString(),
        algorithm_predictions: templateData.template.algorithm_predictions,
        manual_corrections: templateData.template.manual_corrections || [],
    };
    try {
        localStorage.setItem(key, JSON.stringify(backup));
    } catch (e) {
        console.warn('localStorage backup failed:', e);
    }
}

function restoreFromLocalStorage() {
    if (!templateData || currentIdx < 0) return;
    const key = getStorageKey();
    if (!key) return;
    const raw = localStorage.getItem(key);
    if (!raw) {
        document.getElementById('saveStatus').textContent = 'No backup found for this trial';
        setTimeout(() => { document.getElementById('saveStatus').textContent = ''; }, 3000);
        return;
    }
    const backup = JSON.parse(raw);
    const when = new Date(backup.timestamp).toLocaleString();
    if (!confirm(`Restore backup from ${when}? This will overwrite current annotations.`)) return;

    templateData.template.algorithm_predictions = backup.algorithm_predictions;
    templateData.template.manual_corrections = backup.manual_corrections || [];
    dirty = true;
    document.getElementById('saveStatus').textContent = `Restored from backup (${when})`;
    buildValidationRows();
    drawTimeline();
    const preds = templateData.template.algorithm_predictions;
    templates[currentIdx].n_validated = preds.filter(p => p.correct !== null).length;
    templates[currentIdx].complete = templates[currentIdx].n_validated === preds.length;
    updateProgress();
}

// ── Clipboard copy/paste ────────────────────────────────────────────────────
function copyAnnotations() {
    if (!templateData || currentIdx < 0) return;
    const payload = {
        trial_key: templates[currentIdx].trial_key,
        algorithm_predictions: templateData.template.algorithm_predictions,
        manual_corrections: templateData.template.manual_corrections || [],
    };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2)).then(() => {
        document.getElementById('saveStatus').textContent = 'Copied to clipboard';
        setTimeout(() => { if (!dirty) document.getElementById('saveStatus').textContent = ''; }, 3000);
    }).catch(err => {
        document.getElementById('saveStatus').textContent = 'Copy failed: ' + err;
    });
}

async function pasteAnnotations() {
    if (!templateData || currentIdx < 0) return;
    try {
        const text = await navigator.clipboard.readText();
        const data = JSON.parse(text);
        if (!data.algorithm_predictions || !Array.isArray(data.algorithm_predictions)) {
            document.getElementById('saveStatus').textContent = 'Paste failed: invalid format (missing algorithm_predictions)';
            return;
        }
        if (!confirm(`Paste ${data.algorithm_predictions.length} phases from clipboard? This will overwrite current annotations.`)) return;

        templateData.template.algorithm_predictions = data.algorithm_predictions;
        templateData.template.manual_corrections = data.manual_corrections || [];
        dirty = true;
        document.getElementById('saveStatus').textContent = 'Pasted from clipboard (unsaved)';
        buildValidationRows();
        drawTimeline();
        const preds = templateData.template.algorithm_predictions;
        templates[currentIdx].n_validated = preds.filter(p => p.correct !== null).length;
        templates[currentIdx].complete = templates[currentIdx].n_validated === preds.length;
        updateProgress();
    } catch (err) {
        document.getElementById('saveStatus').textContent = 'Paste failed: ' + err.message;
    }
}

// ── Add phase modal ────────────────────────────────────────────────────────
function showAddPhaseModal() {
    document.getElementById('addPhaseModal').classList.add('active');
    if (templateData) {
        document.getElementById('newPhaseEnd').value = templateData.template.n_frames;
    }
}

function hideAddPhaseModal() {
    document.getElementById('addPhaseModal').classList.remove('active');
}

function addPhase() {
    if (!templateData) return;
    const phase = document.getElementById('newPhaseType').value;
    const cycle = parseInt(document.getElementById('newPhaseCycle').value) || 0;
    const start = parseInt(document.getElementById('newPhaseStart').value) || 0;
    const end = parseInt(document.getElementById('newPhaseEnd').value) || 100;

    const newPred = {
        phase,
        cycle_index: cycle,
        start_frame: Math.min(start, end),
        end_frame: Math.max(start, end),
        confidence: 0.0,
        correct: null,
        manual: true,
    };

    // Add to manual_corrections
    if (!templateData.template.manual_corrections) {
        templateData.template.manual_corrections = [];
    }
    templateData.template.manual_corrections.push(newPred);

    // Also add to algorithm_predictions for display
    templateData.template.algorithm_predictions.push(newPred);

    // Sort by start_frame
    templateData.template.algorithm_predictions.sort((a, b) => a.start_frame - b.start_frame);

    dirty = true;
    document.getElementById('saveStatus').textContent = '(unsaved changes)';
    hideAddPhaseModal();
    buildValidationRows();
    drawTimeline();
}

// ── Canvas drawing ─────────────────────────────────────────────────────────
function handleResize() {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const nSignals = 3;
    const totalHeight = PADDING.top + nSignals * (SIGNAL_HEIGHT + SIGNAL_GAP) + PHASE_BAR_HEIGHT + PHASE_BAR_GAP + PADDING.bottom;

    canvas.width = rect.width * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.height = totalHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTimeline();
}

function frameToX(frame) {
    if (!templateData) return 0;
    const nFrames = templateData.template.n_frames;
    const drawWidth = (canvas.width / (window.devicePixelRatio || 1) - PADDING.left - PADDING.right) * zoom;
    return PADDING.left + (frame / nFrames) * drawWidth + panOffset;
}

function xToFrame(x) {
    if (!templateData) return 0;
    const nFrames = templateData.template.n_frames;
    const drawWidth = (canvas.width / (window.devicePixelRatio || 1) - PADDING.left - PADDING.right) * zoom;
    const frame = ((x - PADDING.left - panOffset) / drawWidth) * nFrames;
    return Math.max(0, Math.min(nFrames - 1, Math.round(frame)));
}

function drawTimeline() {
    if (!canvas || !ctx) return;
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, w, h);

    if (!templateData || !templateData.signals) {
        ctx.fillStyle = '#a0a0c0';
        ctx.font = '13px sans-serif';
        ctx.fillText('No kinematic signals available', w / 2 - 80, h / 2);
        return;
    }

    const sig = templateData.signals;
    const nFrames = templateData.template.n_frames;

    // Draw signals
    const signals = [
        { label: 'Speed (mm/s)', data: [sig.tool1_speed, sig.tool2_speed], colors: ['#2196F3', '#F44336'], dual: true },
        { label: 'Jaw (rad)', data: [sig.tool1_jaw, sig.tool2_jaw], colors: ['#2196F3', '#F44336'], dual: true },
        { label: 'Inter-tool (mm)', data: [sig.inter_tool_dist], colors: ['#4CAF50'], dual: false },
    ];

    signals.forEach((s, si) => {
        const yTop = PADDING.top + si * (SIGNAL_HEIGHT + SIGNAL_GAP);
        drawSignalPanel(s, yTop, w);
    });

    // Draw phase bars
    const phaseY = PADDING.top + 3 * (SIGNAL_HEIGHT + SIGNAL_GAP);
    drawPhaseBars(phaseY, w);

    // Draw time axis
    drawTimeAxis(h - PADDING.bottom + 5, w, nFrames);

    // Draw cursor for selected phase
    if (selectedPhaseIdx >= 0 && selectedPhaseIdx < templateData.template.algorithm_predictions.length) {
        const pred = templateData.template.algorithm_predictions[selectedPhaseIdx];
        const x1 = frameToX(pred.start_frame);
        const x2 = frameToX(pred.end_frame);
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, PADDING.top);
        ctx.lineTo(x1, h - PADDING.bottom);
        ctx.moveTo(x2, PADDING.top);
        ctx.lineTo(x2, h - PADDING.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function drawSignalPanel(signal, yTop, canvasW) {
    // Background
    ctx.fillStyle = '#0a1020';
    ctx.fillRect(PADDING.left, yTop, canvasW - PADDING.left - PADDING.right, SIGNAL_HEIGHT);

    // Label
    ctx.fillStyle = '#a0a0c0';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(signal.label, PADDING.left - 4, yTop + 12);
    ctx.textAlign = 'left';

    // Compute y-range across all data arrays
    let allMin = Infinity, allMax = -Infinity;
    signal.data.forEach(arr => {
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] < allMin) allMin = arr[i];
            if (arr[i] > allMax) allMax = arr[i];
        }
    });
    if (allMax <= allMin) { allMax = allMin + 1; }
    const range = allMax - allMin;

    // Draw each trace
    signal.data.forEach((arr, di) => {
        ctx.strokeStyle = signal.colors[di];
        ctx.lineWidth = 1;
        ctx.globalAlpha = signal.dual ? 0.7 : 1.0;
        ctx.beginPath();

        const step = templateData.signals.step || 1;
        for (let i = 0; i < arr.length; i++) {
            const frame = (templateData.signals.frame_indices ? templateData.signals.frame_indices[i] : i * step);
            const x = frameToX(frame);
            if (x < PADDING.left - 2 || x > canvasW - PADDING.right + 2) continue;
            const y = yTop + SIGNAL_HEIGHT - ((arr[i] - allMin) / range) * (SIGNAL_HEIGHT - 4) - 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    });

    // Y-axis ticks
    ctx.fillStyle = '#606080';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(allMax.toFixed(0), PADDING.left - 4, yTop + 10 + 12);
    ctx.fillText(allMin.toFixed(0), PADDING.left - 4, yTop + SIGNAL_HEIGHT - 2);
}

function drawPhaseBars(yTop, canvasW) {
    if (!templateData) return;
    const preds = templateData.template.algorithm_predictions;

    preds.forEach((p, i) => {
        const x1 = frameToX(p.start_frame);
        const x2 = frameToX(p.end_frame);
        if (x2 < PADDING.left || x1 > canvasW - PADDING.right) return;

        const colors = PHASE_COLORS[p.phase] || { bg: '#555', fg: '#aaa' };
        const alpha = p.correct === true ? 1.0 : p.correct === false ? 0.4 : 0.7;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = colors.bg;
        ctx.fillRect(
            Math.max(PADDING.left, x1),
            yTop,
            Math.min(canvasW - PADDING.right, x2) - Math.max(PADDING.left, x1),
            PHASE_BAR_HEIGHT
        );

        // Phase label if wide enough
        const barWidth = x2 - x1;
        if (barWidth > 20) {
            ctx.fillStyle = colors.fg;
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                `${colors.label || p.phase}`,
                (x1 + x2) / 2,
                yTop + PHASE_BAR_HEIGHT / 2 + 3
            );
        }

        // Selected highlight
        if (i === selectedPhaseIdx) {
            ctx.strokeStyle = '#e94560';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                Math.max(PADDING.left, x1),
                yTop,
                Math.min(canvasW - PADDING.right, x2) - Math.max(PADDING.left, x1),
                PHASE_BAR_HEIGHT
            );
        }

        // Correct/incorrect marker
        if (p.correct === true) {
            ctx.fillStyle = '#4CAF50';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('✓', (x1 + x2) / 2, yTop - 2);
        } else if (p.correct === false) {
            ctx.fillStyle = '#f44336';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('✗', (x1 + x2) / 2, yTop - 2);
        }

        ctx.globalAlpha = 1.0;

        // Boundary handles (small rectangles for dragging)
        if (i === selectedPhaseIdx) {
            ctx.fillStyle = '#e94560';
            ctx.fillRect(x1 - 3, yTop, 6, PHASE_BAR_HEIGHT);
            ctx.fillRect(x2 - 3, yTop, 6, PHASE_BAR_HEIGHT);
        }
    });
}

function drawTimeAxis(y, canvasW, nFrames) {
    if (!templateData) return;
    const fps = templateData.template.fps || 13;

    ctx.fillStyle = '#606080';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';

    // Compute tick interval based on zoom
    const totalSeconds = nFrames / fps;
    let tickInterval = 10; // seconds
    if (totalSeconds * zoom > 500) tickInterval = 5;
    if (totalSeconds * zoom > 1000) tickInterval = 2;
    if (totalSeconds * zoom > 2000) tickInterval = 1;

    for (let t = 0; t <= totalSeconds; t += tickInterval) {
        const frame = Math.round(t * fps);
        const x = frameToX(frame);
        if (x < PADDING.left || x > canvasW - PADDING.right) continue;

        ctx.fillText(`${t.toFixed(0)}s`, x, y + 10);
        ctx.fillStyle = '#303050';
        ctx.fillRect(x, PADDING.top, 1, y - PADDING.top - 5);
        ctx.fillStyle = '#606080';
    }
}

// ── Zoom / Pan ─────────────────────────────────────────────────────────────
function zoomIn() { setZoom(zoom * 1.5); }
function zoomOut() { setZoom(zoom / 1.5); }
function resetZoom() { zoom = 1.0; panOffset = 0; document.getElementById('zoomInfo').textContent = '100%'; drawTimeline(); }

function setZoom(newZoom) {
    zoom = Math.max(1.0, Math.min(50.0, newZoom));
    const visibleWidth = canvas.width / (window.devicePixelRatio || 1) - PADDING.left - PADDING.right;
    const drawWidth = visibleWidth * zoom;
    panOffset = Math.min(0, Math.max(panOffset, -(drawWidth - visibleWidth)));
    document.getElementById('zoomInfo').textContent = `${(zoom * 100).toFixed(0)}%`;
    drawTimeline();
}

// ── Canvas events ──────────────────────────────────────────────────────────
function setupCanvasEvents() {
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('mouseleave', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
}

function onMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on a phase bar boundary handle
    if (templateData && selectedPhaseIdx >= 0) {
        const pred = templateData.template.algorithm_predictions[selectedPhaseIdx];
        const x1 = frameToX(pred.start_frame);
        const x2 = frameToX(pred.end_frame);
        const phaseY = PADDING.top + 3 * (SIGNAL_HEIGHT + SIGNAL_GAP);

        if (y >= phaseY && y <= phaseY + PHASE_BAR_HEIGHT) {
            if (Math.abs(x - x1) < 8) {
                isDragging = true;
                dragType = 'boundary-start';
                dragPhaseIdx = selectedPhaseIdx;
                return;
            }
            if (Math.abs(x - x2) < 8) {
                isDragging = true;
                dragType = 'boundary-end';
                dragPhaseIdx = selectedPhaseIdx;
                return;
            }
        }
    }

    // Check if clicking on a phase bar to select it
    if (templateData) {
        const phaseY = PADDING.top + 3 * (SIGNAL_HEIGHT + SIGNAL_GAP);
        if (y >= phaseY && y <= phaseY + PHASE_BAR_HEIGHT) {
            const preds = templateData.template.algorithm_predictions;
            for (let i = preds.length - 1; i >= 0; i--) {
                const x1 = frameToX(preds[i].start_frame);
                const x2 = frameToX(preds[i].end_frame);
                if (x >= x1 && x <= x2) {
                    selectPhase(i);
                    return;
                }
            }
        }
    }

    // Pan mode
    isDragging = true;
    dragType = 'pan';
    dragStartX = x;
    dragStartPan = panOffset;
}

function onMouseMove(e) {
    if (!isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    if (dragType === 'pan') {
        const dx = x - dragStartX;
        const visibleWidth = canvas.width / (window.devicePixelRatio || 1) - PADDING.left - PADDING.right;
        const drawWidth = visibleWidth * zoom;
        panOffset = Math.min(0, Math.max(dragStartPan + dx, -(drawWidth - visibleWidth)));
        drawTimeline();
    } else if (dragType === 'boundary-start' || dragType === 'boundary-end') {
        const frame = xToFrame(x);
        const pred = templateData.template.algorithm_predictions[dragPhaseIdx];
        if (dragType === 'boundary-start') {
            pred.start_frame = Math.max(0, Math.min(pred.end_frame - 1, frame));
        } else {
            pred.end_frame = Math.max(pred.start_frame + 1, Math.min(templateData.template.n_frames - 1, frame));
        }
        dirty = true;
        document.getElementById('saveStatus').textContent = '(unsaved changes)';
        buildValidationRows();
        drawTimeline();
    }
}

function onMouseUp() {
    isDragging = false;
    dragType = null;
}

function onWheel(e) {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
        // Zoom
        const factor = e.deltaY > 0 ? 0.85 : 1.18;
        setZoom(zoom * factor);
    } else {
        // Pan
        const visibleWidth = canvas.width / (window.devicePixelRatio || 1) - PADDING.left - PADDING.right;
        const drawWidth = visibleWidth * zoom;
        panOffset = Math.min(0, Math.max(panOffset - e.deltaY * 2, -(drawWidth - visibleWidth)));
        drawTimeline();
    }
}

// ── Keyboard ───────────────────────────────────────────────────────────────
function setupKeyboard() {
    document.addEventListener('keydown', (e) => {
        // Don't handle if in a form field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        if (!templateData) return;

        const preds = templateData.template.algorithm_predictions;
        switch (e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                if (selectedPhaseIdx > 0) selectPhase(selectedPhaseIdx - 1);
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (selectedPhaseIdx < preds.length - 1) selectPhase(selectedPhaseIdx + 1);
                break;
            case 'c': case 'C':
                if (selectedPhaseIdx >= 0) markPhase(selectedPhaseIdx, true);
                break;
            case 'x': case 'X':
                if (selectedPhaseIdx >= 0) markPhase(selectedPhaseIdx, false);
                break;
            case 'r':
                if (selectedPhaseIdx >= 0) markPhase(selectedPhaseIdx, null);
                break;
            case 's': case 'S':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    saveTemplate();
                } else {
                    saveTemplate();
                }
                break;
            case '[':
                if (selectedPhaseIdx >= 0) adjustBoundary(selectedPhaseIdx, 'start', -5);
                break;
            case ']':
                if (selectedPhaseIdx >= 0) adjustBoundary(selectedPhaseIdx, 'start', 5);
                break;
            case '{':
                if (selectedPhaseIdx >= 0) adjustBoundary(selectedPhaseIdx, 'end', -5);
                break;
            case '}':
                if (selectedPhaseIdx >= 0) adjustBoundary(selectedPhaseIdx, 'end', 5);
                break;
        }
    });
}
