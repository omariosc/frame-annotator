/**
 * Surgical Image Annotation Tool
 * Main JavaScript Application
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    // Current selection
    dataset: null,
    trialId: null,
    frameIdx: null,
    sampledFrames: [],
    allFrames: [],
    frameRate: 100,  // Sample rate: 25, 50, 100, or 'all'

    // Annotation data
    currentAnnotation: null,
    priorAnnotation: null,
    kinematics: null,

    // Tool state
    currentTool: null,
    isDrawing: false,
    tempPoints: [],
    selectedElement: null,

    // Canvas state
    zoom: 1.0,
    imageWidth: 0,
    imageHeight: 0,

    // Options
    showPose: false,
    showPhaseStrip: false,

    // Undo stack
    undoStack: [],
    maxUndo: 20,

    // Progress data
    datasetsProgress: [],

    // Edge selection (Features 5, 6)
    edgeSelectionMode: false,
    hoveredEdgeIndex: null,
    hoveredEdgeTool: null,
    selectedEdges: [],           // Array of {edgeIndex, maskKey, p1, p2} for multi-select

    useEdgeSelection: true,      // Toggle: snap lines to mask edges vs direct placement

    // Vertex editing (Feature 7)
    editMode: false,
    selectedVertexIndex: null,
    selectedVertexTool: null,     // 'tool1_mask' or 'tool2_mask'
    isDraggingVertex: false,
    hoveredVertexIndex: null,
    hoveredVertexTool: null,
    dragStartPos: null,

    // Box selection for vertices
    isBoxSelecting: false,
    boxSelectStart: null,         // [x, y] in image coords
    boxSelectEnd: null,           // [x, y] in image coords
    selectedVertices: [],         // [{maskKey, vertexIndex}, ...]

    // Draggable line endpoints (Feature 7b)
    isDraggingLineEndpoint: false,
    draggedLineInfo: null,        // {linesKey, lineType, endpointIndex}

    // Pan & zoom (Feature 3) — transform-based
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    panStartPanX: 0,
    panStartPanY: 0,
    panX: 0,
    panY: 0,
    spaceHeld: false,

    // Mouse tracking for click detection
    mouseDownPos: null,
    mouseDownButton: -1,

    // SAM segmentation (AI-assisted)
    samMode: false,
    samAvailable: null,    // null=unknown, true/false after check
    samLoading: false,
    samPoints: [],         // [{x, y, label}] accumulated prompts
    samMaskProposals: null, // {masks, scores, best_idx}
    samSelectedMask: 0,    // which proposal to use

    // SAM Apply Existing mode (pre-computed segments)
    samApplyMode: false,
    precomputedSegments: null,  // {polygons, scores, areas, bboxes} from API
    hoveredSegmentIdx: null,
    selectedSegmentIndices: new Set(),  // Multi-select: toggle-based selection

    // SAM availability per frame: { frameIdx: true/false }
    samFrameAvailability: {},

    // Per-frame completion status: { frameIdx: 'completed'|'skipped'|'partial'|'none'|'negative' }
    frameStatus: {},

    // Excluded frame count (from server progress, orthogonal to status)
    excludedCount: 0,

    // Auto-advance: automatically progress to next tool after completing an annotation
    autoAdvance: true,

    // Manual mask mode: disables SAM auto-apply, includes masks in Tab progression
    manualMaskMode: false,

    // Keypoint dragging in edit mode
    isDraggingKeypoint: false,
    draggedKeypointKey: null,
    selectedKeypointKey: null,

    // Dirty flag: true when annotation modified since last load/save
    annotationDirty: false,

    // SSE stream reference for progressive trial loading
    _activeSSE: null,

    // SSE stream reference for Phase 4 full annotation background load
    _fullLoadSSE: null,

    // --- Phase & Peg Annotation ---
    annotationMode: 'tools',   // 'tools' or 'pegs'
    phaseDefinitions: null,    // loaded from /api/phase_definitions
    phaseLabels: {},           // {frameIdx: {coarse, fine, cycle_index, active_tool, events}}
    selectedPegId: null,       // 1-6 or null
    pegDrawingTool: null,      // 'bbox' or 'mask' or null
    pegBboxStart: null,        // [x,y] first corner of bbox being drawn
    pegboardTool: null,        // 'post' or 'post_keypoint' or 'outline' or null
    activePostTarget: null,    // {type: 'source'|'target', index: 0-5} — which post we're drawing
    pegKeypointIdx: null,      // 0, 1, or 2 — which keypoint to place next
    isDraggingPegHandle: false,
    draggedPegHandle: null,    // {pegId, handleType: 'keypoint'|'bbox_corner'|'mask_vertex', index}
    pegEditMode: false,        // Edit All mode for repositioning pegs + pegboard
    pegEditPhase: null,        // null | 'selecting' | 'selected' — rect-select then drag
    pegEditRect: null,         // {x1,y1,x2,y2} selection rectangle in canvas coords
    pegEditSelectedItems: [],  // [{type,points:[...refs]}] items inside selection rect
    pegEditDragStart: null,    // {x,y} drag origin for translating selection
    postMaskMode: null,        // null | 'select_post' | 'drawing' (legacy, kept for compat)
    postMaskTarget: null,      // {type: 'source'|'target', index: 0-5} (legacy)
};

// Tool progression order for auto-advance
const TOOL_PROGRESSION = {
    tool1: ['tool1_mask', 'tool1_top', 'tool1_bottom', 'tool1_middle', 'tool1_joint', 'tool1_ee_tip', 'tool1_ee_left', 'tool1_ee_right'],
    tool2: ['tool2_mask', 'tool2_top', 'tool2_bottom', 'tool2_middle', 'tool2_joint', 'tool2_ee_tip', 'tool2_ee_left', 'tool2_ee_right'],
    full: ['tool1_mask', 'tool1_top', 'tool1_bottom', 'tool1_middle', 'tool1_joint', 'tool1_ee_tip', 'tool1_ee_left', 'tool1_ee_right',
           'tool2_mask', 'tool2_top', 'tool2_bottom', 'tool2_middle', 'tool2_joint', 'tool2_ee_tip', 'tool2_ee_left', 'tool2_ee_right']
};

// ============================================================================
// Loading Bar
// ============================================================================

function showLoading(message = 'Loading...') {
    const bar = document.getElementById('loading-bar');
    const text = bar.querySelector('.loading-bar-text');
    text.textContent = message;
    bar.classList.remove('hidden');
}

function hideLoading() {
    const bar = document.getElementById('loading-bar');
    bar.classList.add('hidden');
    resetLoadingBarMode();
}

/**
 * Show determinate loading progress (e.g. "Loading annotations: 45/200").
 * Switches the loading bar from indeterminate animation to explicit width.
 * @param {string} message - Label prefix
 * @param {number} loaded - Items loaded so far
 * @param {number} total - Total items to load
 */
function showLoadingProgress(message, loaded, total) {
    const bar = document.getElementById('loading-bar');
    const fill = bar.querySelector('.loading-bar-fill');
    const text = bar.querySelector('.loading-bar-text');

    bar.classList.remove('hidden');
    text.textContent = `${message}: ${loaded}/${total}`;

    const percent = total > 0 ? (loaded / total) * 100 : 0;
    // Switch to determinate mode: disable animation, set explicit width
    fill.style.animation = 'none';
    fill.style.marginLeft = '0';
    fill.style.width = `${percent}%`;
}

/**
 * Reset loading bar to indeterminate animation mode.
 * Called by hideLoading() to restore default behavior for next use.
 */
function resetLoadingBarMode() {
    const bar = document.getElementById('loading-bar');
    const fill = bar.querySelector('.loading-bar-fill');
    fill.style.animation = '';
    fill.style.width = '';
    fill.style.marginLeft = '';
}

/**
 * Get the next tool in the annotation progression.
 * Skips masks (done via SAM) and tools marked as "Out" (-1 visibility).
 * @param {string|null} currentTool - Current tool name, or null to start from beginning
 * @returns {string|null} - Next tool name, or null if at end
 */
function getNextTool(currentTool) {
    const progression = TOOL_PROGRESSION.full;
    const currentIdx = currentTool ? progression.indexOf(currentTool) : -1;

    // Find next tool that is not marked as Out
    for (let i = currentIdx + 1; i < progression.length; i++) {
        const nextTool = progression[i];
        const toolNum = nextTool.startsWith('tool1') ? 1 : 2;
        const part = nextTool.replace(/^tool[12]_/, '');

        // Skip masks - they're done via SAM segmentation (unless manual mask mode)
        if (part === 'mask' && !state.manualMaskMode) continue;

        // Skip midlines - they're optional manual-only, not part of Tab flow
        if (part === 'middle') continue;

        // Map part name to visibility key
        let visKey = part;
        if (part === 'top' || part === 'bottom') visKey = 'lines';

        const vis = state.currentAnnotation?.[`tool${toolNum}_visibility`]?.[visKey];

        // Skip if marked as Out (-1)
        if (vis !== -1) {
            return nextTool;
        }
    }

    return null; // End of progression
}

/**
 * Format tool name for display.
 * @param {string} toolId - Tool ID like 'tool1_mask'
 * @returns {string} - Formatted name like 'T1 Mask'
 */
function formatToolName(toolId) {
    if (!toolId) return 'None';
    const toolNum = toolId.startsWith('tool1') ? '1' : '2';
    const part = toolId.replace(/^tool[12]_/, '');
    const partNames = {
        'mask': 'Mask',
        'top': 'Top Line',
        'bottom': 'Bottom Line',
        'middle': 'Mid Line',
        'joint': 'Joint',
        'ee_tip': 'EE Tip',
        'ee_left': 'EE Left',
        'ee_right': 'EE Right'
    };
    return `T${toolNum} ${partNames[part] || part}`;
}

/**
 * Advance to the next tool in the progression (auto-advance helper).
 * Called after completing an annotation when auto-advance is enabled.
 * @param {string} currentTool - The tool that was just completed
 */
function advanceToNextTool(currentTool) {
    if (!state.autoAdvance) return;

    const nextTool = getNextTool(currentTool);
    if (nextTool) {
        // Save is now awaited before calling this function, so no delay needed
        selectTool(nextTool);
        showToast(`Next: ${formatToolName(nextTool)}`);
    } else {
        showToast('All annotations complete!');
        state.currentTool = null;
        updateToolIndicator();
        updateInstructions();
    }
}

// Color scheme
const COLORS = {
    tool1: {
        main: '#4a9eff',
        fill: 'rgba(74, 158, 255, 0.3)',
        stroke: '#4a9eff'
    },
    tool2: {
        main: '#4ae066',
        fill: 'rgba(74, 224, 102, 0.3)',
        stroke: '#4ae066'
    },
    lineStyles: {
        top: { dash: [], width: 3 },
        middle: { dash: [10, 5], width: 3 },  // Auto-computed midline (dotted)
        bottom: { dash: [3, 3], width: 3 }
    },
    highlight: '#ffcc00',
    temp: 'rgba(255, 255, 255, 0.8)',
    pose: {
        tool1: 'rgba(74, 158, 255, 0.7)',
        tool2: 'rgba(74, 224, 102, 0.7)'
    }
};

const EDGE_BORDER = 25;  // px border around image for edge-click snapping

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    datasetSelect: document.getElementById('dataset-select'),
    trialSelect: document.getElementById('trial-select'),
    prevTrialBtn: document.getElementById('prev-trial-btn'),
    nextTrialBtn: document.getElementById('next-trial-btn'),
    progressPanel: document.getElementById('progress-panel'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    completedCount: document.getElementById('completed-count'),
    skippedCount: document.getElementById('skipped-count'),
    excludedCount: document.getElementById('excluded-count'),
    negativeCount: document.getElementById('negative-count'),
    navigationPanel: document.getElementById('navigation-panel'),
    currentFrameNum: document.getElementById('current-frame-num'),
    frameIdx: document.getElementById('frame-idx'),
    totalFrames: document.getElementById('total-frames'),
    prevBtn: document.getElementById('prev-btn'),
    nextBtn: document.getElementById('next-btn'),
    skipBtn: document.getElementById('skip-btn'),
    unskipBtn: document.getElementById('unskip-btn'),
    frameJumpSelect: document.getElementById('frame-jump-select'),
    frameRateSelect: document.getElementById('frame-rate-select'),
    optionsPanel: document.getElementById('options-panel'),
    phaseStripToggle: document.getElementById('phase-strip-toggle'),
    toolsPanel: document.getElementById('tools-panel'),
    statusPanel: document.getElementById('status-panel'),
    canvasContainer: document.getElementById('canvas-container'),
    imageCanvas: document.getElementById('image-canvas'),
    annotationCanvas: document.getElementById('annotation-canvas'),
    placeholder: document.getElementById('canvas-placeholder'),
    zoomIn: document.getElementById('zoom-in'),
    zoomOut: document.getElementById('zoom-out'),
    zoomLevel: document.getElementById('zoom-level'),
    zoomFit: document.getElementById('zoom-fit'),
    currentToolIndicator: document.getElementById('current-tool-indicator'),
    kinematicsInfo: document.getElementById('kinematics-info'),
    kinematicsText: document.getElementById('kinematics-text'),
    instructions: document.getElementById('instructions'),
    clearToolBtn: document.getElementById('clear-tool-btn'),
    usePriorBtn: document.getElementById('use-prior-btn'),

    // SAM panel elements
    samPanel: document.getElementById('sam-panel'),
    samApplyBtn: document.getElementById('sam-apply-btn'),
    samNewBtn: document.getElementById('sam-new-btn'),
    samStatusText: document.getElementById('sam-status-text'),
    samInstructions: document.getElementById('sam-instructions'),
    samToolAssign: document.getElementById('sam-tool-assign'),
    samAssignTool1: document.getElementById('sam-assign-tool1'),
    samAssignTool2: document.getElementById('sam-assign-tool2'),

    // JSON Viewer panel elements
    jsonViewerPanel: document.getElementById('json-viewer-panel'),
    jsonViewerContent: document.getElementById('json-viewer-content'),
    copyJsonBtn: document.getElementById('copy-json-btn'),

    // Manual mask toggle
    manualMaskCheckbox: document.getElementById('manual-mask-checkbox'),

    // Reload button
    reloadBtn: document.getElementById('reload-btn')
};

const imageCtx = elements.imageCanvas.getContext('2d');
const annotationCtx = elements.annotationCanvas.getContext('2d');

// ============================================================================
// SAM Multi-Select Helpers
// ============================================================================

function toggleSegmentSelection(idx) {
    if (state.selectedSegmentIndices.has(idx)) {
        state.selectedSegmentIndices.delete(idx);
    } else {
        state.selectedSegmentIndices.add(idx);
    }
    drawAllAnnotations();
    updateSamSelectionUI();
}

function clearSegmentSelection() {
    state.selectedSegmentIndices.clear();
    drawAllAnnotations();
    updateSamSelectionUI();
}

function getSelectedSegmentCount() {
    return state.selectedSegmentIndices.size;
}

// ---------------------------------------------------------------------------
// Polygon Merge Utilities (for multi-select SAM segments)
// ---------------------------------------------------------------------------

/**
 * Calculate squared distance between two points.
 */
function distSq(p1, p2) {
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    return dx * dx + dy * dy;
}

/**
 * Find nearest vertices between two polygons.
 * Returns {i1, i2, dist} - indices into poly1 and poly2, and distance.
 */
function findNearestVertices(poly1, poly2) {
    let minDist = Infinity;
    let bestI1 = 0, bestI2 = 0;

    for (let i = 0; i < poly1.length; i++) {
        for (let j = 0; j < poly2.length; j++) {
            const d = distSq(poly1[i], poly2[j]);
            if (d < minDist) {
                minDist = d;
                bestI1 = i;
                bestI2 = j;
            }
        }
    }
    return { i1: bestI1, i2: bestI2, dist: Math.sqrt(minDist) };
}

/**
 * Merge two polygons by connecting at their nearest vertices.
 * Creates a single polygon that traces both outlines with a bridge.
 */
function mergeTwoPolygons(poly1, poly2) {
    if (!poly1 || poly1.length < 3) return poly2;
    if (!poly2 || poly2.length < 3) return poly1;

    const { i1, i2, dist } = findNearestVertices(poly1, poly2);

    // If polygons share a vertex (dist ≈ 0), simpler merge
    if (dist < 1) {
        // Rotate both to start at shared point, concatenate
        const rotated1 = [...poly1.slice(i1), ...poly1.slice(0, i1)];
        const rotated2 = [...poly2.slice(i2), ...poly2.slice(0, i2)];
        return [...rotated1, ...rotated2];
    }

    // Bridge connection: trace poly1 → bridge → poly2 → bridge back
    // Rotate poly1 to start at nearest point
    const rotated1 = [...poly1.slice(i1), ...poly1.slice(0, i1)];
    // Rotate poly2 to start at nearest point
    const rotated2 = [...poly2.slice(i2), ...poly2.slice(0, i2)];

    // Merged polygon: poly1 (from i1 around) → poly2 (from i2 around) → back to start
    return [...rotated1, ...rotated2];
}

/**
 * Merge multiple polygons into a single polygon.
 * Iteratively merges closest pairs until one remains.
 */
function mergePolygons(polygons) {
    if (!polygons || polygons.length === 0) return [];
    if (polygons.length === 1) return polygons[0].slice();

    // Copy polygons array
    let remaining = polygons.map(p => p.slice());

    // Iteratively merge closest pair
    while (remaining.length > 1) {
        // Find closest pair of polygons
        let minDist = Infinity;
        let mergeI = 0, mergeJ = 1;

        for (let i = 0; i < remaining.length; i++) {
            for (let j = i + 1; j < remaining.length; j++) {
                const { dist } = findNearestVertices(remaining[i], remaining[j]);
                if (dist < minDist) {
                    minDist = dist;
                    mergeI = i;
                    mergeJ = j;
                }
            }
        }

        // Merge the closest pair
        const merged = mergeTwoPolygons(remaining[mergeI], remaining[mergeJ]);

        // Remove the two, add the merged one
        remaining = remaining.filter((_, idx) => idx !== mergeI && idx !== mergeJ);
        remaining.push(merged);
    }

    return remaining[0];
}

/**
 * Extract the top edge from a polygon (two highest points, sorted left-to-right).
 * Uses the points closest to minY within a tolerance band.
 */
function extractTopEdge(polygon) {
    if (!polygon || polygon.length < 2) return null;

    // Find min Y (top of polygon)
    const minY = Math.min(...polygon.map(p => p[1]));
    const tolerance = 20; // pixels - points within this band count as "top"

    // Get all points near the top
    const topPoints = polygon.filter(p => p[1] <= minY + tolerance);

    if (topPoints.length < 2) {
        // Fallback: just get the two points with smallest Y
        const sorted = [...polygon].sort((a, b) => a[1] - b[1]);
        return [sorted[0], sorted[1]].sort((a, b) => a[0] - b[0]);
    }

    // Find leftmost and rightmost among top points
    const leftmost = topPoints.reduce((a, b) => a[0] < b[0] ? a : b);
    const rightmost = topPoints.reduce((a, b) => a[0] > b[0] ? a : b);

    return [[...leftmost], [...rightmost]];
}

/**
 * Extract the bottom edge from a polygon (two lowest points, sorted left-to-right).
 * Uses the points closest to maxY within a tolerance band.
 */
function extractBottomEdge(polygon) {
    if (!polygon || polygon.length < 2) return null;

    // Find max Y (bottom of polygon)
    const maxY = Math.max(...polygon.map(p => p[1]));
    const tolerance = 20; // pixels

    // Get all points near the bottom
    const bottomPoints = polygon.filter(p => p[1] >= maxY - tolerance);

    if (bottomPoints.length < 2) {
        // Fallback: just get the two points with largest Y
        const sorted = [...polygon].sort((a, b) => b[1] - a[1]);
        return [sorted[0], sorted[1]].sort((a, b) => a[0] - b[0]);
    }

    // Find leftmost and rightmost among bottom points
    const leftmost = bottomPoints.reduce((a, b) => a[0] < b[0] ? a : b);
    const rightmost = bottomPoints.reduce((a, b) => a[0] > b[0] ? a : b);

    return [[...leftmost], [...rightmost]];
}

/**
 * Get the centroid of a polygon (average of all vertices).
 */
function getPolygonCentroid(polygon) {
    if (!polygon || polygon.length === 0) return null;

    const sumX = polygon.reduce((sum, p) => sum + p[0], 0);
    const sumY = polygon.reduce((sum, p) => sum + p[1], 0);

    return [sumX / polygon.length, sumY / polygon.length];
}

/**
 * Get the base point for middle line from selected segments.
 * Uses the centroid of the merged polygon's "base" area (bottom portion).
 */
function getMiddleLineBase(polygon) {
    if (!polygon || polygon.length < 3) return null;

    // Find the bottom edge midpoint as the base
    const bottomEdge = extractBottomEdge(polygon);
    if (!bottomEdge) return getPolygonCentroid(polygon);

    // Midpoint of bottom edge
    return [
        (bottomEdge[0][0] + bottomEdge[1][0]) / 2,
        (bottomEdge[0][1] + bottomEdge[1][1]) / 2
    ];
}

function getCombinedBoundingBox() {
    if (state.selectedSegmentIndices.size === 0) return null;
    if (!state.precomputedSegments || !state.precomputedSegments.bboxes) return null;

    const bboxes = state.precomputedSegments.bboxes;
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const idx of state.selectedSegmentIndices) {
        const bbox = bboxes[idx];
        if (!bbox) continue;
        const [x1, y1, x2, y2] = bbox;
        minX = Math.min(minX, x1);
        minY = Math.min(minY, y1);
        maxX = Math.max(maxX, x2);
        maxY = Math.max(maxY, y2);
    }

    if (minX === Infinity) return null;
    return [minX, minY, maxX, maxY];
}

function updateSamSelectionUI() {
    const count = state.selectedSegmentIndices.size;
    const infoEl = document.getElementById('sam-selection-info');
    const countEl = document.getElementById('sam-selection-count');
    const assignEl = elements.samToolAssign;
    const labelEl = document.getElementById('sam-assign-label');

    if (infoEl && countEl) {
        if (count > 0) {
            infoEl.style.display = 'flex';
            countEl.textContent = `${count} selected`;
        } else {
            infoEl.style.display = 'none';
        }
    }

    if (assignEl && labelEl) {
        if (count > 0) {
            assignEl.style.display = 'flex';
            labelEl.textContent = count > 1 ? 'Assign combined to:' : 'Assign to:';
        } else {
            assignEl.style.display = 'none';
        }
    }

    // Update instructions
    if (elements.samInstructions && state.precomputedSegments) {
        if (count > 0) {
            const noun = count === 1 ? 'segment' : 'segments';
            elements.samInstructions.textContent =
                `${count} ${noun} selected. Keys: 1/6=mask, 2/7=top, 3/8=mid, 4/9=bot`;
        } else {
            elements.samInstructions.textContent =
                `${state.precomputedSegments.polygons.length} segments. Click to select, click again to deselect.`;
        }
    }
}

// ============================================================================
// API Functions
// ============================================================================

async function api(endpoint, options = {}) {
    const response = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    return response.json();
}

async function loadDatasets() {
    // Phase 1: Fast — just dataset names + trial lists (no progress I/O)
    const datasets = await api('/datasets');
    state.datasetsRaw = datasets;
    state.datasetsProgress = null;

    elements.datasetSelect.innerHTML = '<option value="">Select Dataset...</option>';
    datasets.forEach(ds => {
        const opt = document.createElement('option');
        opt.value = ds.name;
        opt.textContent = ds.name;
        // Store minimal trial info (name-only, no progress yet)
        const trialStubs = ds.trials.map(name => ({
            trial_id: `${ds.name}/${name}`,
            trial_name: name,
            total: 0, completed: 0, skipped: 0, negative: 0,
            remaining: 0, percentage: 0
        }));
        opt.dataset.trials = JSON.stringify(trialStubs);
        elements.datasetSelect.appendChild(opt);
    });
}

async function loadProgressAsync() {
    // Phase 2: Fetch progress in background — updates UI when ready
    showLoading('Loading progress data...');
    try {
        const progressData = await api('/datasets/progress');
        state.datasetsProgress = progressData;

        // Update dataset dropdown labels with percentages
        const options = elements.datasetSelect.options;
        for (let i = 1; i < options.length; i++) {
            const ds = progressData.find(d => d.name === options[i].value);
            if (ds) {
                options[i].textContent = `${ds.name} (${ds.percentage}% done)`;
                options[i].dataset.trials = JSON.stringify(ds.trials);
            }
        }

        // If a dataset is currently selected, update the trial dropdown too
        if (state.dataset) {
            const dsData = progressData.find(d => d.name === state.dataset);
            if (dsData) {
                const currentTrialId = state.trialId;
                elements.trialSelect.innerHTML = '<option value="">Select Trial...</option>';
                dsData.trials.forEach(trial => {
                    const opt = document.createElement('option');
                    opt.value = trial.trial_id;
                    applyTrialProgress(opt, trial.trial_name, trial);
                    if (trial.trial_id === currentTrialId) opt.selected = true;
                    elements.trialSelect.appendChild(opt);
                });
            }
        }
    } catch (err) {
        console.warn('Progress load failed (non-blocking):', err);
    } finally {
        hideLoading();
    }
}

// Backward-compat wrapper used by refresh logic elsewhere
async function loadDatasetsWithProgress() {
    await loadDatasets();
    await loadProgressAsync();
}

/**
 * Background-refresh progress for a single dataset from disk.
 * Calls the server to recompute _progress.json for every trial in the dataset,
 * then updates the trial dropdown labels and dataset dropdown label.
 * Non-blocking: errors just log a warning.
 * @param {string} datasetName - Dataset name (e.g. "7DOF2024")
 */
async function refreshDatasetProgress(datasetName) {
    try {
        const data = await api(`/datasets/${encodeURIComponent(datasetName)}/refresh_progress`);
        console.log(`Refreshed progress for ${datasetName}: ${data.percentage}% done`);

        // Update each trial option label
        const trialSelect = elements.trialSelect;
        for (let i = 1; i < trialSelect.options.length; i++) {
            const opt = trialSelect.options[i];
            const trial = data.trials.find(t => t.trial_id === opt.value);
            if (trial && trial.total > 0) {
                // Extract base trial name (everything before parenthesis)
                const text = opt.textContent;
                const match = text.match(/^(.+?)\s*\(/);
                const name = opt.dataset.trialName || (match ? match[1] : text.trim());
                applyTrialProgress(opt, name, trial);
            }
        }

        // Update dataset dropdown label
        const dsSelect = elements.datasetSelect;
        for (let i = 0; i < dsSelect.options.length; i++) {
            if (dsSelect.options[i].value === datasetName) {
                dsSelect.options[i].textContent = `${datasetName} (${data.percentage}% done)`;
                // Also update stored trials JSON for future dataset changes
                dsSelect.options[i].dataset.trials = JSON.stringify(data.trials);
                break;
            }
        }

        // Update state
        if (state.datasetsProgress) {
            const idx = state.datasetsProgress.findIndex(d => d.name === datasetName);
            if (idx >= 0) {
                state.datasetsProgress[idx] = data;
            }
        }
    } catch (err) {
        console.warn(`Background progress refresh failed for ${datasetName}:`, err);
    }
}

/**
 * Build the trial option label from a progress record.
 * First slot mirrors the active annotation mode (tool segmentation vs. pegs);
 * the second slot is always phase-annotation coverage so both are visible.
 * @param {string} name - Trial name (text before the parentheses)
 * @param {Object} trial - Trial progress record from the server
 * @param {string} mode - 'tools' or 'pegs'
 * @returns {string} Formatted option text
 */
function formatTrialOptionText(name, trial, mode) {
    if (!trial || !(trial.total > 0)) return name;

    let firstDone, firstTotal, firstPct;
    if (mode === 'pegs') {
        firstDone = trial.peg_completed ?? 0;
        firstTotal = trial.peg_total ?? trial.total;
        firstPct = (trial.peg_percentage != null)
            ? trial.peg_percentage
            : (firstTotal > 0 ? (firstDone / firstTotal * 100) : 0);
    } else {
        firstDone = (trial.completed ?? 0) + (trial.skipped ?? 0);
        firstTotal = trial.total;
        firstPct = trial.percentage ?? (firstTotal > 0 ? (firstDone / firstTotal * 100) : 0);
    }

    let label = `${name} (${firstDone}/${firstTotal} ${Number(firstPct).toFixed(1)}%)`;

    const phaseTotal = trial.phase_total ?? 0;
    if (phaseTotal > 0) {
        const phaseDone = trial.phase_completed ?? 0;
        const phasePct = trial.phase_percentage ?? (phaseDone / phaseTotal * 100);
        label += ` (${phaseDone}/${phaseTotal} ${Number(phasePct).toFixed(1)}%)`;
    }
    return label;
}

/**
 * Cache the latest progress record on the option and render its label.
 * Keeps a JSON blob in `opt.dataset.progress` so mode toggles can re-render
 * without re-hitting the server.
 * @param {HTMLOptionElement} opt - Option element to update
 * @param {string} name - Trial name (text before any parentheses)
 * @param {Object} trial - Trial progress record
 */
function applyTrialProgress(opt, name, trial) {
    opt.dataset.trialName = name;
    opt.dataset.progress = JSON.stringify(trial || {});
    opt.textContent = formatTrialOptionText(name, trial, state.annotationMode);
    styleTrialOption(opt, trial);
}

/**
 * Re-render every trial option label using its cached progress data.
 * Called when the annotation mode toggles so the first slot swaps between
 * tool and peg progress without a server round trip.
 */
function rerenderTrialOptionsForMode() {
    const sel = elements.trialSelect;
    if (!sel) return;
    for (let i = 1; i < sel.options.length; i++) {
        const opt = sel.options[i];
        const name = opt.dataset.trialName;
        const raw = opt.dataset.progress;
        if (!name || !raw) continue;
        try {
            const trial = JSON.parse(raw);
            opt.textContent = formatTrialOptionText(name, trial, state.annotationMode);
        } catch (e) { /* leave label as-is */ }
    }
}

/**
 * Style a trial <option> element based on completion percentage.
 * Green = 100% done, Orange = 50%+, Gray = not started, Default = in progress.
 * @param {HTMLOptionElement} opt - The option element to style
 * @param {Object} trial - Trial progress object with completed, skipped, total fields
 */
function styleTrialOption(opt, trial) {
    if (!trial || trial.total <= 0) return;
    const done = trial.completed + trial.skipped;
    const pct = done / trial.total;
    if (pct >= 1.0) {
        opt.style.color = '#2ecc71';       // Green: 100% done
        opt.style.fontWeight = 'bold';
    } else if (pct >= 0.5) {
        opt.style.color = '#f39c12';       // Orange: 50%+ done
        opt.style.fontWeight = '';
    } else if (done > 0) {
        opt.style.color = '';              // Default: in progress
        opt.style.fontWeight = '';
    } else {
        opt.style.color = '#999';          // Gray: not started
        opt.style.fontWeight = '';
    }
}

/**
 * Background full annotation load for a trial via SSE.
 * Loads ALL annotation files (not just sampled) so navigating to any frame
 * finds real data in the cache instead of blanks.
 * No text indicators — progress is fully managed by recalculateProgress().
 * @param {string} trialId - Trial identifier
 */
function loadFullTrialAnnotations(trialId) {
    // Cancel any previous Phase 4 SSE
    if (state._fullLoadSSE) {
        state._fullLoadSSE.close();
        state._fullLoadSSE = null;
    }

    const url = `/api/trials/load_stream?trial_id=${encodeURIComponent(trialId)}&full=true`;
    const es = new EventSource(url);
    state._fullLoadSSE = es;

    // Safety timeout: 60 seconds
    const timeout = setTimeout(() => {
        console.warn('Full annotation load timeout — closing');
        es.close();
        if (state._fullLoadSSE === es) state._fullLoadSSE = null;
    }, 60000);

    es.addEventListener('done', () => {
        es.close();
        clearTimeout(timeout);
        if (state._fullLoadSSE === es) state._fullLoadSSE = null;
        console.log(`Full annotations loaded for ${trialId}`);
        // Refresh progress with the now-complete annotation cache
        if (state.trialId === trialId) {
            fetchFrameStatus(trialId);
        }
    });

    es.onerror = () => {
        console.warn('Full annotation load SSE error');
        es.close();
        clearTimeout(timeout);
        if (state._fullLoadSSE === es) state._fullLoadSSE = null;
    };
}

/**
 * Lightweight trial frame loader — gets sampled frames WITHOUT loading annotations.
 * Used in Phase 1 of progressive trial loading to show first frame quickly.
 * @param {string} trialId - Trial identifier
 * @returns {object} Response data with sampled_frames, progress, annotation_file_count
 */
async function loadTrialFramesLite(trialId) {
    const data = await api(`/trials/${trialId}/frames_lite`);
    state.sampledFrames = data.sampled_frames;
    // Show approximate progress from _progress.json immediately.
    // Phase 3 will overwrite with accurate counts.
    if (data.progress) {
        const p = data.progress;
        const total = data.sampled_frames.length;
        updateProgress({
            total: total,
            completed: p.completed || 0,
            skipped: p.skipped || 0,
            negative: p.negative || 0,
            excluded: 0,
            partial: p.partial || 0,
            broken: p.broken || 0,
            remaining: total - (p.completed || 0) - (p.skipped || 0) - (p.negative || 0) - (p.broken || 0)
        });
    }
    updateFrameJumpDropdown();
    return data;
}

async function loadTrialFrames(trialId) {
    const data = await api(`/trials/${trialId}/frames`);
    state.sampledFrames = data.sampled_frames;
    updateProgress(data.progress);
    updateFrameJumpDropdown();
    fetchSamAvailability(trialId);       // cosmetic, keep fire-and-forget
    await fetchFrameStatus(trialId);     // await: accurate counts via recalculateProgress()
    return data;
}

async function fetchSamAvailability(trialId, frames) {
    try {
        let url = `/sam/availability/${trialId}`;
        if (frames && frames.length > 0) {
            url += `?frames=${frames.join(',')}`;
        }
        const availability = await api(url);
        state.samFrameAvailability = availability;
        updateFrameJumpDropdown();
    } catch (e) {
        console.warn('Failed to fetch SAM availability:', e);
    }
}

async function fetchFrameStatus(trialId) {
    try {
        const status = await api(`/trials/${trialId}/frame_status`);
        state.frameStatus = status;
        recalculateProgress();
        updateFrameJumpDropdown();
    } catch (e) {
        console.warn('Failed to fetch frame status:', e);
    }
}

function isCurrentFrameComplete() {
    if (!state.currentAnnotation) return false;
    const ann = state.currentAnnotation;
    migrateVisibility(ann);
    const t1v = ann.tool1_visibility || {};
    const t2v = ann.tool2_visibility || {};

    // Helper: check if component needs annotation (visibility === 1)
    const needsAnnotation = (v) => v === 1;

    // Visibility: 1=visible (requires annotation), 0=occluded (optional), -1=out
    // Tool 1
    if (needsAnnotation(t1v.mask) && !(ann.tool1_mask && ann.tool1_mask.length >= 3)) return false;
    if (needsAnnotation(t1v.lines) && !['top', 'bottom'].every(
        lt => ann.tool1_lines?.[lt] && ann.tool1_lines[lt].length === 2
    )) return false;
    // Tool 1 keypoints
    if (needsAnnotation(t1v.joint) && !(ann.tool1_joint && ann.tool1_joint.length === 2)) return false;
    if (needsAnnotation(t1v.ee_tip) && !(ann.tool1_ee_tip && ann.tool1_ee_tip.length === 2)) return false;
    if (needsAnnotation(t1v.ee_left) && !(ann.tool1_ee_left && ann.tool1_ee_left.length === 2)) return false;
    if (needsAnnotation(t1v.ee_right) && !(ann.tool1_ee_right && ann.tool1_ee_right.length === 2)) return false;

    // Tool 2
    if (needsAnnotation(t2v.mask) && !(ann.tool2_mask && ann.tool2_mask.length >= 3)) return false;
    if (needsAnnotation(t2v.lines) && !['top', 'bottom'].every(
        lt => ann.tool2_lines?.[lt] && ann.tool2_lines[lt].length === 2
    )) return false;
    // Tool 2 keypoints
    if (needsAnnotation(t2v.joint) && !(ann.tool2_joint && ann.tool2_joint.length === 2)) return false;
    if (needsAnnotation(t2v.ee_tip) && !(ann.tool2_ee_tip && ann.tool2_ee_tip.length === 2)) return false;
    if (needsAnnotation(t2v.ee_left) && !(ann.tool2_ee_left && ann.tool2_ee_left.length === 2)) return false;
    if (needsAnnotation(t2v.ee_right) && !(ann.tool2_ee_right && ann.tool2_ee_right.length === 2)) return false;

    return true;
}

function isCurrentFrameNegative() {
    if (!state.currentAnnotation) return false;
    const ann = state.currentAnnotation;
    const t1v = ann.tool1_visibility || {};
    const t2v = ann.tool2_visibility || {};
    const VIS_KEYS = ['mask', 'lines', 'joint', 'ee_tip', 'ee_left', 'ee_right'];
    const t1Out = VIS_KEYS.every(k => t1v[k] === -1);
    const t2Out = VIS_KEYS.every(k => t2v[k] === -1);
    return t1Out && t2Out;
}

function recalculateProgress() {
    // Broken frames are gone — exclude them from active frames and totals
    const activeFrames = getActiveFrames().filter(
        f => state.frameStatus[String(f)] !== 'broken'
    );
    let completed = 0;
    let skipped = 0;
    let negative = 0;
    let extraCompleted = 0;
    let extraSkipped = 0;
    let extraNegative = 0;

    // Count from frameStatus for active frames
    activeFrames.forEach(frame => {
        const status = state.frameStatus[String(frame)];
        if (status === 'completed') completed++;
        else if (status === 'skipped') skipped++;
        else if (status === 'negative') negative++;
    });

    // Also count completed/negative off-rate frames (not skipped/partial/broken)
    const activeSet = new Set(activeFrames.map(String));
    Object.entries(state.frameStatus).forEach(([frameStr, status]) => {
        if (!activeSet.has(frameStr) && status !== 'broken') {
            if (status === 'completed') extraCompleted++;
            else if (status === 'negative') extraNegative++;
        }
    });

    const totalNeg = negative + extraNegative;
    const total = activeFrames.length;
    const effectiveTotal = total - negative;
    const done = completed + skipped;
    const percent = effectiveTotal > 0 ? Math.min((done / effectiveTotal) * 100, 100) : 0;
    const extraTotal = extraCompleted + extraSkipped + extraNegative;

    elements.progressFill.style.width = `${percent}%`;
    const extraText = extraTotal > 0 ? ` +${extraTotal}` : '';
    const negText = totalNeg > 0 ? `, ${totalNeg} neg` : '';
    elements.progressText.textContent = `${done} / ${effectiveTotal}${extraText}${negText} (${percent.toFixed(1)}%)`;
    elements.completedCount.textContent = `${completed + extraCompleted} completed`;
    elements.skippedCount.textContent = `${skipped + extraSkipped} skipped`;
    elements.excludedCount.textContent = `${state.excludedCount} excluded`;
    elements.negativeCount.textContent = `${totalNeg} out`;

    // Update trial dropdown percentage for current trial
    updateTrialDropdownPercentage(done, effectiveTotal, percent);

    // Update dataset dropdown percentage (recompute from trial dropdown entries)
    updateDatasetDropdownPercentage();
}

/**
 * Update the trial dropdown to show current progress count and percentage.
 * @param {number} done - Number of completed + skipped frames
 * @param {number} total - Total effective frames (excluding negatives)
 * @param {number} percent - Current completion percentage
 */
function updateTrialDropdownPercentage(done, total, percent) {
    if (!state.trialId) return;

    const trialSelect = elements.trialSelect;
    for (let i = 0; i < trialSelect.options.length; i++) {
        const opt = trialSelect.options[i];
        if (opt.value !== state.trialId) continue;

        // Extract trial name (everything before the parenthesis, or full text if none)
        const text = opt.textContent;
        const match = text.match(/^(.+?)\s*\(/);
        const name = opt.dataset.trialName || (match ? match[1] : text.trim());

        // Merge live tool counts into the cached progress so the phase slot
        // (and peg slot, if previously known) survive the in-place update.
        let trial = {};
        try { trial = JSON.parse(opt.dataset.progress || '{}'); } catch (e) { /* reset */ }
        trial.completed = done;
        trial.skipped = 0;
        trial.total = total;
        trial.percentage = Number(percent.toFixed(1));
        applyTrialProgress(opt, name, trial);
        break;
    }
}

/**
 * Recompute and update the dataset dropdown percentage from trial dropdown entries.
 * Parses "(done/total X%)" from each trial option, sums across all trials,
 * and updates the current dataset option text.
 */
function updateDatasetDropdownPercentage() {
    if (!state.dataset) return;

    // Parse done/total from each trial option in the trial dropdown
    let datasetDone = 0;
    let datasetTotal = 0;
    const trialSelect = elements.trialSelect;
    for (let i = 1; i < trialSelect.options.length; i++) {
        const text = trialSelect.options[i].textContent;
        const match = text.match(/\((\d+)\/(\d+)\s/);
        if (match) {
            datasetDone += parseInt(match[1], 10);
            datasetTotal += parseInt(match[2], 10);
        }
    }

    // Update the dataset dropdown entry
    const datasetPct = datasetTotal > 0 ? (datasetDone / datasetTotal * 100) : 0;
    const dsSelect = elements.datasetSelect;
    for (let i = 0; i < dsSelect.options.length; i++) {
        if (dsSelect.options[i].value === state.dataset) {
            dsSelect.options[i].textContent = `${state.dataset} (${datasetPct.toFixed(1)}% done)`;
            break;
        }
    }
}

async function loadAllFrames(trialId) {
    const data = await api(`/trials/${trialId}/all_frames`);
    state.allFrames = data.all_frames;
    updateFrameJumpDropdown();
    // Refresh SAM availability for sampled frames only (not all frames)
    fetchSamAvailability(trialId);
    return data;
}

/**
 * Stream annotation loading progress via SSE.
 * Opens an EventSource to /api/trials/load_stream and updates the loading bar
 * with determinate progress. Resolves when loading is complete.
 * @param {string} trialId - Trial identifier
 * @param {number} annotationFileCount - Total annotation files to load
 * @returns {Promise<void>} Resolves when loading completes or is cancelled
 */
function streamAnnotationLoad(trialId, annotationFileCount) {
    return new Promise((resolve) => {
        const url = `/api/trials/load_stream?trial_id=${encodeURIComponent(trialId)}`;
        const es = new EventSource(url);
        state._activeSSE = es;

        // Safety timeout: 30 seconds
        const timeout = setTimeout(() => {
            console.warn('SSE timeout — closing stream');
            es.close();
            state._activeSSE = null;
            resolve();
        }, 30000);

        es.addEventListener('progress', (event) => {
            // Guard: bail if user switched trials
            if (state.trialId !== trialId) {
                es.close();
                state._activeSSE = null;
                clearTimeout(timeout);
                resolve();
                return;
            }
            try {
                const data = JSON.parse(event.data);
                showLoadingProgress('Loading annotations', data.loaded, data.total);
            } catch (e) {
                console.warn('SSE parse error:', e);
            }
        });

        es.addEventListener('done', (event) => {
            es.close();
            state._activeSSE = null;
            clearTimeout(timeout);
            try {
                const data = JSON.parse(event.data);
                showLoadingProgress('Loading annotations', data.loaded, data.total);
            } catch (e) { /* ignore */ }
            // Hide loading bar after a brief moment so user sees 100%
            setTimeout(() => hideLoading(), 600);
            resolve();
        });

        es.onerror = () => {
            console.warn('SSE connection error — falling back');
            es.close();
            state._activeSSE = null;
            clearTimeout(timeout);
            resolve();
        };
    });
}

async function loadFrameImage(trialId, frameIdx) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = `/api/frames/${trialId}/${frameIdx}/image`;
    });
}

async function loadFrameAnnotations(trialId, frameIdx) {
    return api(`/frames/${trialId}/${frameIdx}/annotations`);
}

/**
 * Load annotation for a single frame without triggering full trial load.
 * Returns annotation + kinematics but no prior (requires full cache).
 * @param {string} trialId - Trial identifier
 * @param {number} frameIdx - Frame index
 * @returns {object} Response with annotation, prior:null, kinematics
 */
async function loadFrameAnnotationSingle(trialId, frameIdx) {
    return api(`/frames/${trialId}/${frameIdx}/annotation_single`);
}

async function saveAnnotations() {
    if (!state.trialId || state.frameIdx === null || !state.currentAnnotation) return;

    // Ensure all fields have valid values (not undefined) - undefined values are omitted by JSON.stringify
    const ann = state.currentAnnotation;
    const data = {
        tool1_mask: ann.tool1_mask || [],
        tool2_mask: ann.tool2_mask || [],
        tool1_lines: {top: [], bottom: [], middle: [], ...(ann.tool1_lines || {})},
        tool2_lines: {top: [], bottom: [], middle: [], ...(ann.tool2_lines || {})},
        tool1_joint: ann.tool1_joint || [],
        tool1_ee_tip: ann.tool1_ee_tip || [],
        tool1_ee_left: ann.tool1_ee_left || [],
        tool1_ee_right: ann.tool1_ee_right || [],
        tool2_joint: ann.tool2_joint || [],
        tool2_ee_tip: ann.tool2_ee_tip || [],
        tool2_ee_left: ann.tool2_ee_left || [],
        tool2_ee_right: ann.tool2_ee_right || [],
        tool1_visibility: ann.tool1_visibility || {mask: 1, lines: 1, joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1},
        tool2_visibility: ann.tool2_visibility || {mask: 1, lines: 1, joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1},
        pegs: ann.pegs || [],
        pegboard: ann.pegboard || {},
        phase: ann.phase || {},
    };

    try {
        await api(`/frames/${state.trialId}/${state.frameIdx}/annotations`, {
            method: 'POST',
            body: JSON.stringify(data)
        });

        state.annotationDirty = false;
        updateStatus();
        updateToolButtonStatus();
        updateJsonViewer();

        // Update frame status for progress tracking
        if (isCurrentFrameNegative()) {
            state.frameStatus[String(state.frameIdx)] = 'negative';
        } else if (isCurrentFrameComplete()) {
            state.frameStatus[String(state.frameIdx)] = 'completed';
        } else {
            state.frameStatus[String(state.frameIdx)] = 'partial';
        }
        recalculateProgress();
        updateFrameJumpDropdown();

        showToast('Saved');
    } catch (error) {
        console.error('Save failed:', error);
        showToast('Save failed! Click Force Save to retry.', true);
    }
}

async function skipFrame() {
    if (!state.trialId || state.frameIdx === null) return;

    await api(`/frames/${state.trialId}/${state.frameIdx}/skip`, {
        method: 'POST'
    });

    state.frameStatus[String(state.frameIdx)] = 'skipped';
    if (state.currentAnnotation) state.currentAnnotation.skipped = true;
    recalculateProgress();
    updateFrameJumpDropdown();
    updateSkipButtons();

    showToast('Frame skipped');
    navigateNext();
}

async function unskipFrame() {
    if (!state.trialId || state.frameIdx === null) return;

    await api(`/frames/${state.trialId}/${state.frameIdx}/unskip`, {
        method: 'POST'
    });

    state.frameStatus[String(state.frameIdx)] = 'partial';
    if (state.currentAnnotation) state.currentAnnotation.skipped = false;
    recalculateProgress();
    updateFrameJumpDropdown();
    updateSkipButtons();
    updateStatus();
    updateToolButtonStatus();

    showToast('Frame unskipped');
}

async function reloadFrame() {
    if (!state.trialId || state.frameIdx === null) return;

    try {
        const result = await api(`/frames/${state.trialId}/${state.frameIdx}/ensure_init`, {
            method: 'POST'
        });

        if (result.initialized) {
            showToast('Annotation file initialized');
        }

        await navigateToFrame(state.frameIdx);
        showToast('Frame reloaded');
    } catch (error) {
        console.error('Reload failed:', error);
        showToast('Reload failed', true);
    }
}

function updateSkipButtons() {
    const isSkipped = state.currentAnnotation && state.currentAnnotation.skipped;
    elements.skipBtn.style.display = isSkipped ? 'none' : '';
    elements.unskipBtn.style.display = isSkipped ? '' : 'none';
}

// ============================================================================
// Quaternion to Rotation Matrix and Axes
// ============================================================================

function quaternionToRotationMatrix(qw, qx, qy, qz) {
    // Convert quaternion to 3x3 rotation matrix
    // Each column represents an axis direction (X, Y, Z)
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;

    return {
        // X axis (Right) - first column
        xAxis: {
            x: 1 - 2 * (yy + zz),
            y: 2 * (xy + wz),
            z: 2 * (xz - wy)
        },
        // Y axis (Up) - second column
        yAxis: {
            x: 2 * (xy - wz),
            y: 1 - 2 * (xx + zz),
            z: 2 * (yz + wx)
        },
        // Z axis (Forward) - third column
        zAxis: {
            x: 2 * (xz + wy),
            y: 2 * (yz - wx),
            z: 1 - 2 * (xx + yy)
        }
    };
}

// ============================================================================
// Geometry Helpers (Features 4-7)
// ============================================================================

/**
 * Compute distance from point (px,py) to line segment (ax,ay)-(bx,by).
 * Returns {distance, projectedPoint: [x,y], t} where t is the parameter [0,1].
 */
function pointToSegmentDistance(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) {
        // Degenerate segment (a == b)
        const d = Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
        return { distance: d, projectedPoint: [ax, ay], t: 0 };
    }

    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const projX = ax + t * dx;
    const projY = ay + t * dy;
    const distance = Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);

    return { distance, projectedPoint: [projX, projY], t };
}

/**
 * Find the nearest polygon edge to point (px, py) within maxDist.
 * Returns {edgeIndex, distance, projectedPoint, maskKey} or null.
 */
function findNearestEdge(px, py, maxDist = 10) {
    if (!state.currentAnnotation) return null;

    let best = null;

    for (const maskKey of ['tool1_mask', 'tool2_mask']) {
        const polygon = state.currentAnnotation[maskKey];
        if (!polygon || polygon.length < 3) continue;

        for (let i = 0; i < polygon.length; i++) {
            const j = (i + 1) % polygon.length;
            const a = polygon[i];
            const b = polygon[j];
            const result = pointToSegmentDistance(px, py, a[0], a[1], b[0], b[1]);

            if (result.distance < maxDist && (!best || result.distance < best.distance)) {
                best = {
                    edgeIndex: i,
                    distance: result.distance,
                    projectedPoint: result.projectedPoint,
                    maskKey,
                    t: result.t
                };
            }
        }
    }

    return best;
}

/**
 * Find the nearest polygon vertex to (px, py) within maxDist.
 * Returns {maskKey, vertexIndex, distance, point} or null.
 */
function findNearestVertex(px, py, maxDist = 10) {
    if (!state.currentAnnotation) return null;

    let best = null;

    for (const maskKey of ['tool1_mask', 'tool2_mask']) {
        const polygon = state.currentAnnotation[maskKey];
        if (!polygon || polygon.length < 3) continue;

        for (let i = 0; i < polygon.length; i++) {
            const d = Math.sqrt((px - polygon[i][0]) ** 2 + (py - polygon[i][1]) ** 2);
            if (d < maxDist && (!best || d < best.distance)) {
                best = { maskKey, vertexIndex: i, distance: d, point: polygon[i] };
            }
        }
    }

    return best;
}

/**
 * Find the nearest line endpoint to (px, py) within maxDist.
 * Returns {linesKey, lineType, endpointIndex, distance, point} or null.
 */
function findNearestLineEndpoint(px, py, maxDist = 12) {
    if (!state.currentAnnotation) return null;

    let best = null;

    for (const linesKey of ['tool1_lines', 'tool2_lines']) {
        const lines = state.currentAnnotation[linesKey];
        if (!lines) continue;

        for (const lineType of ['top', 'bottom', 'middle']) {
            const pts = lines[lineType];
            if (!pts || pts.length !== 2) continue;

            for (let i = 0; i < 2; i++) {
                const d = Math.sqrt((px - pts[i][0]) ** 2 + (py - pts[i][1]) ** 2);
                if (d < maxDist && (!best || d < best.distance)) {
                    best = { linesKey, lineType, endpointIndex: i, distance: d, point: pts[i] };
                }
            }
        }
    }

    return best;
}

/**
 * Get the mask key corresponding to the current tool.
 * E.g. 'tool1_top' -> 'tool1_mask', 'tool2_middle' -> 'tool2_mask'
 */
function getMaskKeyForTool(tool) {
    if (!tool) return null;
    return tool.startsWith('tool1') ? 'tool1_mask' : 'tool2_mask';
}

/**
 * Check if the current tool's corresponding mask exists (has >= 3 vertices).
 */
function currentToolMaskExists() {
    if (!state.currentTool || !state.currentAnnotation) return false;
    const maskKey = getMaskKeyForTool(state.currentTool);
    const mask = state.currentAnnotation[maskKey];
    return mask && mask.length >= 3;
}

// ============================================================================
// Drawing Functions
// ============================================================================

function clearAnnotationCanvas() {
    annotationCtx.save();
    annotationCtx.setTransform(1, 0, 0, 1, 0, 0);
    annotationCtx.clearRect(0, 0, elements.annotationCanvas.width, elements.annotationCanvas.height);
    annotationCtx.restore();
}

function drawAllAnnotations() {
    clearAnnotationCanvas();

    // Draw edge-click border
    const bw = EDGE_BORDER;
    const w = state.imageWidth;
    const h = state.imageHeight;
    if (w > 0 && h > 0) {
        const isDrawingMask = state.isDrawing && state.currentTool?.includes('mask');
        annotationCtx.fillStyle = isDrawingMask
            ? 'rgba(255, 165, 0, 0.4)'    // Orange highlight when actively drawing mask
            : 'rgba(100, 100, 255, 0.15)'; // Subtle blue when idle
        // Top strip
        annotationCtx.fillRect(-bw, -bw, w + 2 * bw, bw);
        // Bottom strip
        annotationCtx.fillRect(-bw, h, w + 2 * bw, bw);
        // Left strip
        annotationCtx.fillRect(-bw, 0, bw, h);
        // Right strip
        annotationCtx.fillRect(w, 0, bw, h);
    }

    // Draw pose indicators if enabled
    if (state.showPose && state.kinematics) {
        drawPoseIndicators();
    }

    if (!state.currentAnnotation) return;

    const ann = state.currentAnnotation;

    // Draw masks
    drawPolygon(ann.tool1_mask, COLORS.tool1, 'tool1_mask');
    drawPolygon(ann.tool2_mask, COLORS.tool2, 'tool2_mask');

    // Draw lines (top and bottom only; midline drawn separately below)
    ['top', 'bottom'].forEach(lineType => {
        drawLine(ann.tool1_lines?.[lineType], COLORS.tool1, lineType, 'tool1');
        drawLine(ann.tool2_lines?.[lineType], COLORS.tool2, lineType, 'tool2');
    });

    // Draw midline: prefer manual midline, fall back to auto-computed from top/bottom
    for (const toolNum of [1, 2]) {
        const lines = ann[`tool${toolNum}_lines`];
        const colors = toolNum === 1 ? COLORS.tool1 : COLORS.tool2;
        if (lines?.middle?.length === 2) {
            // Manual midline exists - draw it
            drawLine(lines.middle, colors, 'middle', `tool${toolNum}`);
        } else if (lines?.top?.length === 2 && lines?.bottom?.length === 2) {
            // Auto-compute midline from top/bottom
            const midline = computeMidlineFromLines(lines.top, lines.bottom);
            if (midline) {
                drawLine(midline, colors, 'middle', `tool${toolNum}`);
            }
        }
    }

    // Draw keypoints with connecting dotted lines
    drawKeypoints(ann, 1, COLORS.tool1);
    drawKeypoints(ann, 2, COLORS.tool2);

    // Draw temp points while drawing
    if (state.isDrawing && state.tempPoints.length > 0) {
        if (state.currentTool?.includes('mask')) {
            drawTempPolygon();
        } else if (state.currentTool?.includes('top') ||
                   state.currentTool?.includes('bottom') ||
                   state.currentTool?.includes('middle')) {
            drawTempLine();
        }
    }

    // Draw edge hover highlight (Features 5, 6)
    if (state.edgeSelectionMode && state.hoveredEdgeIndex !== null && state.hoveredEdgeTool) {
        drawEdgeHighlight(state.hoveredEdgeTool, state.hoveredEdgeIndex);
    }

    // Draw selected edges for multi-select (top/bottom lines)
    if (state.selectedEdges.length > 0) {
        drawSelectedEdges();
    }

    // Draw vertex hover/selection highlights (Feature 7)
    if (state.editMode) {
        drawEditModeOverlays();
    }

    // Draw pre-computed SAM overlays in apply-existing mode
    if (state.samApplyMode && state.precomputedSegments) {
        drawPrecomputedOverlays();
    }

    // Draw peg annotations (visible in both modes, highlighted in pegs mode)
    drawPegAnnotations(ann);
}

/**
 * Draw peg bboxes, masks, and pegboard overlays.
 */
function drawPegAnnotations(ann) {
    const PEG_COLORS = ['#f97316', '#a855f7', '#06b6d4', '#eab308', '#ec4899', '#84cc16'];
    const pegs = ann.pegs || [];
    const ctx = annotationCtx;
    const inPegsMode = state.annotationMode === 'pegs';
    const alpha = inPegsMode ? 0.6 : 0.2;

    pegs.forEach(peg => {
        const color = PEG_COLORS[(peg.id - 1) % PEG_COLORS.length];
        const isSelected = inPegsMode && state.selectedPegId === peg.id;

        // Draw bbox
        if (peg.bbox && peg.bbox.length === 4) {
            const [x, y, w, h] = peg.bbox;
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 3 : 1.5;
            ctx.setLineDash(isSelected ? [] : [4, 4]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = color;
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(`P${peg.id}`, x + 2, y - 4);

            // Draw corner handles when in pegs mode
            if (inPegsMode) {
                const corners = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
                corners.forEach(([cx, cy]) => {
                    ctx.fillStyle = isSelected ? color : '#fff';
                    ctx.fillRect(cx - 3, cy - 3, 6, 6);
                    ctx.strokeStyle = isSelected ? '#fff' : color;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(cx - 3, cy - 3, 6, 6);
                });
            }
        }

        // Draw mask
        if (peg.mask && peg.mask.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(peg.mask[0][0], peg.mask[0][1]);
            for (let i = 1; i < peg.mask.length; i++) {
                ctx.lineTo(peg.mask[i][0], peg.mask[i][1]);
            }
            ctx.closePath();
            ctx.fillStyle = color + (isSelected ? '50' : '25');
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 2.5 : 1;
            ctx.stroke();
        }

        // Draw keypoints and connecting triangle lines
        if (peg.keypoints) {
            const validKps = peg.keypoints.filter(kp => kp !== null);
            // Draw connecting lines (triangle) if 2+ keypoints
            if (validKps.length >= 2) {
                ctx.beginPath();
                ctx.moveTo(validKps[0][0], validKps[0][1]);
                for (let i = 1; i < validKps.length; i++) {
                    ctx.lineTo(validKps[i][0], validKps[i][1]);
                }
                if (validKps.length === 3) ctx.closePath();
                ctx.strokeStyle = color;
                ctx.lineWidth = isSelected ? 2 : 1.5;
                ctx.stroke();
            }
            // Draw keypoint dots with labels
            peg.keypoints.forEach((kp, kpIdx) => {
                if (!kp) return;
                ctx.beginPath();
                ctx.arc(kp[0], kp[1], isSelected ? 5 : 4, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                // Label
                if (inPegsMode) {
                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 10px sans-serif';
                    ctx.fillText(`${kpIdx + 1}`, kp[0] + 7, kp[1] - 5);
                }
            });
        }
    });

    // Draw pegboard
    const pegboard = ann.pegboard || {};
    if (pegboard.source_posts?.length > 0 || pegboard.target_posts?.length > 0 || pegboard.board_mask?.length >= 3 || pegboard.source_post_masks?.some(m => m?.length >= 3) || pegboard.target_post_masks?.some(m => m?.length >= 3)) {
        // Source posts (blue — centroid marker + label)
        (pegboard.source_posts || []).forEach((pt, i) => {
            if (!pt) return;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
            ctx.fillStyle = '#3b82f6';
            ctx.fill();
            ctx.fillStyle = '#3b82f6';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`S${i + 1}`, pt[0], pt[1] - 10);
            ctx.textAlign = 'left';
        });

        // Target posts (green — centroid marker + label)
        (pegboard.target_posts || []).forEach((pt, i) => {
            if (!pt) return;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
            ctx.fillStyle = '#10b981';
            ctx.fill();
            ctx.fillStyle = '#10b981';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`T${i + 1}`, pt[0], pt[1] - 10);
            ctx.textAlign = 'left';
        });

        // Board outline
        if (pegboard.board_mask?.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(pegboard.board_mask[0][0], pegboard.board_mask[0][1]);
            for (let i = 1; i < pegboard.board_mask.length; i++) {
                ctx.lineTo(pegboard.board_mask[i][0], pegboard.board_mask[i][1]);
            }
            ctx.closePath();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = 'rgba(255,255,255,0.4)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw vertex handles in edit mode
            if (state.pegEditMode && inPegsMode) {
                pegboard.board_mask.forEach(pt => {
                    ctx.fillStyle = '#fff';
                    ctx.fillRect(pt[0] - 3, pt[1] - 3, 6, 6);
                    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(pt[0] - 3, pt[1] - 3, 6, 6);
                });
            }
        }

        // Draw post masks
        const drawPostMasks = (masks, color) => {
            if (!masks) return;
            masks.forEach(mask => {
                if (!mask || mask.length < 3) return;
                ctx.beginPath();
                ctx.moveTo(mask[0][0], mask[0][1]);
                for (let i = 1; i < mask.length; i++) ctx.lineTo(mask[i][0], mask[i][1]);
                ctx.closePath();
                ctx.fillStyle = color + '30';
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                // Draw vertices in edit mode
                if (state.pegEditMode && inPegsMode) {
                    mask.forEach(pt => {
                        ctx.fillStyle = color;
                        ctx.fillRect(pt[0] - 2, pt[1] - 2, 4, 4);
                    });
                }
            });
        };
        drawPostMasks(pegboard.source_post_masks, '#3b82f6');
        drawPostMasks(pegboard.target_post_masks, '#10b981');

        // Draw post keypoints (+ cross markers)
        const drawPostKeypoints = (keypoints, color) => {
            if (!keypoints) return;
            keypoints.forEach((kp, i) => {
                if (!kp) return;
                const sz = 5;
                ctx.beginPath();
                ctx.moveTo(kp[0] - sz, kp[1]); ctx.lineTo(kp[0] + sz, kp[1]);
                ctx.moveTo(kp[0], kp[1] - sz); ctx.lineTo(kp[0], kp[1] + sz);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        };
        drawPostKeypoints(pegboard.source_post_keypoints, '#3b82f6');
        drawPostKeypoints(pegboard.target_post_keypoints, '#10b981');

        // Draw post drag handles in edit mode
        if (state.pegEditMode && inPegsMode) {
            const drawPostHandle = (pt, color) => {
                ctx.beginPath();
                ctx.arc(pt[0], pt[1], 10, 0, Math.PI * 2);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
            };
            (pegboard.source_posts || []).forEach(pt => drawPostHandle(pt, '#3b82f6'));
            (pegboard.target_posts || []).forEach(pt => drawPostHandle(pt, '#10b981'));
        }
    }

    // Draw bbox being drawn
    // Draw bbox being drawn
    if (state.pegDrawingTool === 'bbox' && state.pegBboxStart && state.tempPoints.length > 0) {
        const [x1, y1] = state.pegBboxStart;
        const [x2, y2] = state.tempPoints[0];
        const selColor = PEG_COLORS[(state.selectedPegId - 1) % PEG_COLORS.length];
        ctx.strokeStyle = selColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        ctx.setLineDash([]);
    }

    // Draw peg mask polygon in progress
    if (state.pegDrawingTool === 'mask' && state.isDrawing && state.tempPoints.length > 0) {
        const selColor = PEG_COLORS[(state.selectedPegId - 1) % PEG_COLORS.length];
        ctx.beginPath();
        ctx.moveTo(state.tempPoints[0][0], state.tempPoints[0][1]);
        for (let i = 1; i < state.tempPoints.length; i++) {
            ctx.lineTo(state.tempPoints[i][0], state.tempPoints[i][1]);
        }
        ctx.strokeStyle = selColor;
        ctx.lineWidth = 2;
        ctx.stroke();
        // Draw vertices
        state.tempPoints.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
            ctx.fillStyle = selColor;
            ctx.fill();
        });
    }

    // Draw post polygon in progress
    if (state.pegboardTool === 'post' && state.isDrawing && state.tempPoints.length > 0) {
        const postColor = state.activePostTarget?.type === 'source' ? '#3b82f6' : '#10b981';
        ctx.beginPath();
        ctx.moveTo(state.tempPoints[0][0], state.tempPoints[0][1]);
        for (let i = 1; i < state.tempPoints.length; i++) {
            ctx.lineTo(state.tempPoints[i][0], state.tempPoints[i][1]);
        }
        ctx.strokeStyle = postColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        state.tempPoints.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
            ctx.fillStyle = postColor;
            ctx.fill();
        });
    }

    // Draw pegboard outline polygon in progress
    if (state.pegboardTool === 'outline' && state.isDrawing && state.tempPoints.length > 0) {
        ctx.beginPath();
        ctx.moveTo(state.tempPoints[0][0], state.tempPoints[0][1]);
        for (let i = 1; i < state.tempPoints.length; i++) {
            ctx.lineTo(state.tempPoints[i][0], state.tempPoints[i][1]);
        }
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        state.tempPoints.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        });
    }
}

function drawPoseIndicators() {
    if (!state.kinematics) return;

    const tools = [
        { key: 'tool1', data: state.kinematics.tool1 },
        { key: 'tool2', data: state.kinematics.tool2 }
    ];

    // Axis colors: X=Red, Y=Green, Z=Blue (standard RGB convention)
    const axisColors = {
        xAxis: '#ff4444',  // Red
        yAxis: '#44ff44',  // Green
        zAxis: '#4444ff'   // Blue
    };

    tools.forEach(({ key, data }) => {
        if (!data || !data.Rotation || data.Rotation.length < 4) return;

        const rotation = data.Rotation;
        const rotMatrix = quaternionToRotationMatrix(rotation[0], rotation[1], rotation[2], rotation[3]);

        // Get EE tip position if available, otherwise use center
        const eeTip = state.currentAnnotation?.[`${key}_ee_tip`];
        let startX, startY;

        if (eeTip) {
            startX = eeTip[0];
            startY = eeTip[1];
        } else {
            // Use center of image as fallback
            startX = state.imageWidth / 2;
            startY = state.imageHeight / 2;
        }

        // Draw all three axes
        const arrowLength = 60;
        const axes = [
            { axis: rotMatrix.xAxis, color: axisColors.xAxis, label: 'X' },
            { axis: rotMatrix.yAxis, color: axisColors.yAxis, label: 'Y' },
            { axis: rotMatrix.zAxis, color: axisColors.zAxis, label: 'Z' }
        ];

        axes.forEach(({ axis, color, label }) => {
            // Project 3D axis onto 2D image plane (use X and Y components)
            const endX = startX + axis.x * arrowLength;
            const endY = startY + axis.y * arrowLength;

            // Draw arrow line
            annotationCtx.beginPath();
            annotationCtx.moveTo(startX, startY);
            annotationCtx.lineTo(endX, endY);
            annotationCtx.strokeStyle = color;
            annotationCtx.lineWidth = 3;
            annotationCtx.stroke();

            // Draw arrowhead
            const angle = Math.atan2(axis.y, axis.x);
            const headLength = 12;
            annotationCtx.beginPath();
            annotationCtx.moveTo(endX, endY);
            annotationCtx.lineTo(
                endX - headLength * Math.cos(angle - Math.PI / 6),
                endY - headLength * Math.sin(angle - Math.PI / 6)
            );
            annotationCtx.lineTo(
                endX - headLength * Math.cos(angle + Math.PI / 6),
                endY - headLength * Math.sin(angle + Math.PI / 6)
            );
            annotationCtx.closePath();
            annotationCtx.fillStyle = color;
            annotationCtx.fill();

            // Draw axis label
            annotationCtx.font = 'bold 12px Arial';
            annotationCtx.fillStyle = color;
            annotationCtx.fillText(label, endX + 5, endY + 5);
        });

        // Draw origin circle
        const toolColor = key === 'tool1' ? COLORS.tool1.main : COLORS.tool2.main;
        annotationCtx.beginPath();
        annotationCtx.arc(startX, startY, 6, 0, Math.PI * 2);
        annotationCtx.fillStyle = toolColor;
        annotationCtx.fill();
        annotationCtx.strokeStyle = '#fff';
        annotationCtx.lineWidth = 2;
        annotationCtx.stroke();
    });
}

function drawPolygon(points, colors, id) {
    if (!points || points.length < 3) return;

    annotationCtx.beginPath();
    annotationCtx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        annotationCtx.lineTo(points[i][0], points[i][1]);
    }
    annotationCtx.closePath();

    annotationCtx.fillStyle = colors.fill;
    annotationCtx.fill();

    annotationCtx.strokeStyle = colors.stroke;
    annotationCtx.lineWidth = 2;
    annotationCtx.stroke();

    // Draw vertices
    points.forEach((p, i) => {
        annotationCtx.beginPath();
        annotationCtx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        annotationCtx.fillStyle = colors.main;
        annotationCtx.fill();
    });
}

function drawLine(points, colors, lineType, toolPrefix) {
    if (!points || points.length !== 2) return;

    const style = COLORS.lineStyles[lineType];

    annotationCtx.beginPath();
    annotationCtx.moveTo(points[0][0], points[0][1]);
    annotationCtx.lineTo(points[1][0], points[1][1]);

    annotationCtx.strokeStyle = colors.stroke;
    annotationCtx.lineWidth = style.width;
    annotationCtx.setLineDash(style.dash);
    annotationCtx.stroke();
    annotationCtx.setLineDash([]);

    // Draw endpoints
    points.forEach(p => {
        annotationCtx.beginPath();
        annotationCtx.arc(p[0], p[1], 6, 0, Math.PI * 2);
        annotationCtx.fillStyle = colors.main;
        annotationCtx.fill();
        annotationCtx.strokeStyle = '#fff';
        annotationCtx.lineWidth = 1;
        annotationCtx.stroke();
    });
}

function drawPoint(point, colors, id) {
    if (!point) return;

    // Outer circle
    annotationCtx.beginPath();
    annotationCtx.arc(point[0], point[1], 10, 0, Math.PI * 2);
    annotationCtx.fillStyle = colors.fill;
    annotationCtx.fill();
    annotationCtx.strokeStyle = colors.stroke;
    annotationCtx.lineWidth = 3;
    annotationCtx.stroke();

    // Inner dot
    annotationCtx.beginPath();
    annotationCtx.arc(point[0], point[1], 4, 0, Math.PI * 2);
    annotationCtx.fillStyle = colors.main;
    annotationCtx.fill();

    // Crosshair
    annotationCtx.beginPath();
    annotationCtx.moveTo(point[0] - 15, point[1]);
    annotationCtx.lineTo(point[0] + 15, point[1]);
    annotationCtx.moveTo(point[0], point[1] - 15);
    annotationCtx.lineTo(point[0], point[1] + 15);
    annotationCtx.strokeStyle = colors.stroke;
    annotationCtx.lineWidth = 1;
    annotationCtx.stroke();
}

/**
 * Draw keypoints for a tool with connecting dotted lines.
 * Connections: midline → joint, joint → ee_tip, joint → ee_left, joint → ee_right
 */
function drawKeypoints(ann, toolNum, colors) {
    const prefix = `tool${toolNum}`;
    const joint = ann[`${prefix}_joint`];
    const eeTip = ann[`${prefix}_ee_tip`];
    const eeLeft = ann[`${prefix}_ee_left`];
    const eeRight = ann[`${prefix}_ee_right`];
    const lines = ann[`${prefix}_lines`];
    const vis = ann[`${prefix}_visibility`] || {};

    // Get closest point on midline to joint (for connecting line)
    let midlineClosest = null;
    if (lines?.top?.length === 2 && lines?.bottom?.length === 2) {
        const midline = computeMidlineFromLines(lines.top, lines.bottom);
        if (midline && joint?.length === 2) {
            midlineClosest = closestPointOnSegment(midline[0], midline[1], joint);
        }
    }

    // Draw connecting dotted lines
    // Midline → Joint
    if (midlineClosest && joint?.length === 2 && vis.joint !== -1) {
        drawDottedLine(midlineClosest, joint, colors.stroke);
    }

    // Joint → EE Tip
    if (joint?.length === 2 && eeTip?.length === 2 && vis.ee_tip !== -1) {
        drawDottedLine(joint, eeTip, colors.stroke);
    }

    // Joint → EE Left
    if (joint?.length === 2 && eeLeft?.length === 2 && vis.ee_left !== -1) {
        drawDottedLine(joint, eeLeft, colors.stroke);
    }

    // Joint → EE Right
    if (joint?.length === 2 && eeRight?.length === 2 && vis.ee_right !== -1) {
        drawDottedLine(joint, eeRight, colors.stroke);
    }

    // Draw keypoint markers
    if (joint?.length === 2 && vis.joint !== -1) {
        drawKeypointMarker(joint, colors, 'J', vis.joint === 0);
    }
    if (eeTip?.length === 2 && vis.ee_tip !== -1) {
        drawKeypointMarker(eeTip, colors, 'T', vis.ee_tip === 0);
    }
    if (eeLeft?.length === 2 && vis.ee_left !== -1) {
        drawKeypointMarker(eeLeft, colors, 'L', vis.ee_left === 0);
    }
    if (eeRight?.length === 2 && vis.ee_right !== -1) {
        drawKeypointMarker(eeRight, colors, 'R', vis.ee_right === 0);
    }
}

/**
 * Find the closest point on a line segment to a given point.
 * @param {Array} segStart - [x, y] start of segment
 * @param {Array} segEnd - [x, y] end of segment
 * @param {Array} point - [x, y] the point to find closest to
 * @returns {Array} [x, y] closest point on segment
 */
function closestPointOnSegment(segStart, segEnd, point) {
    const dx = segEnd[0] - segStart[0];
    const dy = segEnd[1] - segStart[1];
    const lengthSq = dx * dx + dy * dy;

    if (lengthSq === 0) return segStart; // Degenerate segment

    // Project point onto line, clamped to [0, 1]
    let t = ((point[0] - segStart[0]) * dx + (point[1] - segStart[1]) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));

    return [segStart[0] + t * dx, segStart[1] + t * dy];
}

/**
 * Draw a dotted line between two points.
 */
function drawDottedLine(from, to, color) {
    annotationCtx.beginPath();
    annotationCtx.moveTo(from[0], from[1]);
    annotationCtx.lineTo(to[0], to[1]);
    annotationCtx.strokeStyle = color;
    annotationCtx.lineWidth = 2;
    annotationCtx.setLineDash([6, 4]);
    annotationCtx.stroke();
    annotationCtx.setLineDash([]);
}

/**
 * Draw a keypoint marker with label.
 */
function drawKeypointMarker(point, colors, label, isOccluded = false) {
    const radius = 8;
    const x = point[0];
    const y = point[1];

    // Outer circle
    annotationCtx.beginPath();
    annotationCtx.arc(x, y, radius, 0, Math.PI * 2);
    annotationCtx.fillStyle = isOccluded ? 'rgba(255, 152, 0, 0.5)' : colors.fill;
    annotationCtx.fill();
    annotationCtx.strokeStyle = isOccluded ? '#ff9800' : colors.stroke;
    annotationCtx.lineWidth = 2;
    annotationCtx.stroke();

    // Inner dot
    annotationCtx.beginPath();
    annotationCtx.arc(x, y, 3, 0, Math.PI * 2);
    annotationCtx.fillStyle = isOccluded ? '#ff9800' : colors.main;
    annotationCtx.fill();

    // Label
    annotationCtx.font = 'bold 10px sans-serif';
    annotationCtx.fillStyle = '#fff';
    annotationCtx.textAlign = 'center';
    annotationCtx.textBaseline = 'middle';
    annotationCtx.fillText(label, x, y - radius - 8);
    annotationCtx.textAlign = 'left';
    annotationCtx.textBaseline = 'alphabetic';
}

/**
 * Find nearest keypoint/tooltip to a point.
 * Returns {toolPrefix, keypointType, distance, point} or null.
 */
function findNearestTooltip(px, py, maxDist = 15) {
    if (!state.currentAnnotation) return null;

    const keypointTypes = ['joint', 'ee_tip', 'ee_left', 'ee_right'];
    let best = null;

    for (const toolNum of [1, 2]) {
        const prefix = `tool${toolNum}`;
        for (const kpType of keypointTypes) {
            const point = state.currentAnnotation[`${prefix}_${kpType}`];
            if (!point || point.length !== 2) continue;

            const d = Math.sqrt((px - point[0]) ** 2 + (py - point[1]) ** 2);
            if (d < maxDist && (!best || d < best.distance)) {
                best = {
                    toolPrefix: prefix,
                    keypointType: kpType,
                    keypointKey: `${prefix}_${kpType}`,
                    distance: d,
                    point: point
                };
            }
        }
    }

    return best;
}

function drawTempPolygon() {
    if (state.tempPoints.length === 0) return;

    annotationCtx.beginPath();
    annotationCtx.moveTo(state.tempPoints[0][0], state.tempPoints[0][1]);
    for (let i = 1; i < state.tempPoints.length; i++) {
        annotationCtx.lineTo(state.tempPoints[i][0], state.tempPoints[i][1]);
    }

    annotationCtx.strokeStyle = COLORS.temp;
    annotationCtx.lineWidth = 2;
    annotationCtx.setLineDash([5, 5]);
    annotationCtx.stroke();
    annotationCtx.setLineDash([]);

    // Draw vertices
    state.tempPoints.forEach((p, i) => {
        annotationCtx.beginPath();
        annotationCtx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        annotationCtx.fillStyle = i === 0 ? COLORS.highlight : COLORS.temp;
        annotationCtx.fill();
    });
}

function drawTempLine() {
    if (state.tempPoints.length === 0) return;

    state.tempPoints.forEach(p => {
        annotationCtx.beginPath();
        annotationCtx.arc(p[0], p[1], 6, 0, Math.PI * 2);
        annotationCtx.fillStyle = COLORS.temp;
        annotationCtx.fill();
    });
}

// ============================================================================
// Tool Handling
// ============================================================================

function selectTool(toolId) {
    // Deselect all tool buttons
    document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => btn.classList.remove('active'));

    // Cancel any in-progress drawing
    cancelDrawing();

    // Exit edit mode when selecting a drawing tool
    if (state.editMode) {
        state.editMode = false;
        state.selectedVertices = [];
        const editBtn = document.getElementById('edit-mode-btn');
        if (editBtn) editBtn.classList.remove('active');
    }

    // Exit SAM modes when manually selecting a drawing tool
    if (state.samMode) {
        exitSamMode();
    }
    if (state.samApplyMode) {
        exitApplyExistingMode();
    }

    if (toolId === state.currentTool) {
        // Deselect
        state.currentTool = null;
        elements.currentToolIndicator.textContent = 'No tool selected';
        elements.instructions.textContent = 'Select a tool from the sidebar to begin annotating';
        updateCursorClasses();
        return;
    }

    state.currentTool = toolId;

    // Activate button
    const btn = document.querySelector(`[data-tool="${toolId}"]`);
    if (btn) btn.classList.add('active');

    // Check if we should auto-enter edge selection mode (only for top/bottom lines, not midlines)
    if ((toolId.includes('top') || toolId.includes('bottom')) && !toolId.includes('middle') && currentToolMaskExists()) {
        enterEdgeSelectionMode();
    }

    // Update UI
    updateToolIndicator();
    updateInstructions();
}

function updateToolIndicator() {
    const tool = state.currentTool;
    if (!tool) {
        elements.currentToolIndicator.textContent = 'No tool selected';
        return;
    }

    const toolNum = tool.startsWith('tool1') ? '1' : '2';
    const color = tool.startsWith('tool1') ? COLORS.tool1.main : COLORS.tool2.main;

    let type = 'Unknown';
    if (tool.includes('mask')) type = 'Mask';
    else if (tool.includes('top')) type = 'Top Line';
    else if (tool.includes('middle')) type = 'Mid Line';
    else if (tool.includes('bottom')) type = 'Bottom Line';
    else if (tool.includes('joint')) type = 'Joint';
    else if (tool.includes('ee_tip')) type = 'EE Tip';
    else if (tool.includes('ee_left')) type = 'EE Left';
    else if (tool.includes('ee_right')) type = 'EE Right';

    elements.currentToolIndicator.innerHTML =
        `<span style="color: ${color}">Tool ${toolNum}</span>: ${type}`;

    // Update cursor style
    updateCursorClasses();
}

function updateInstructions() {
    if (state.editMode) {
        elements.instructions.textContent =
            'Drag vertices/keypoints/endpoints to move. Click to select, Delete to remove.';
        return;
    }

    const tool = state.currentTool;
    if (!tool) {
        elements.instructions.textContent = 'Select a tool from the sidebar to begin annotating';
        return;
    }

    // Edge selection modes
    if (state.edgeSelectionMode) {
        if (tool.includes('top') || tool.includes('bottom')) {
            if (state.useEdgeSelection) {
                // Snap mode: single edge auto-finalize
                elements.instructions.textContent =
                    'Click a mask edge to set the line. Esc to cancel.';
            } else {
                // Multi-edge mode: accumulate edges, Enter to confirm
                const n = state.selectedEdges.length;
                elements.instructions.textContent = n > 0
                    ? `${n} edge${n > 1 ? 's' : ''} selected. Click more edges or press Enter to confirm. Esc to cancel.`
                    : 'Click mask edges to select them, then press Enter to confirm. Esc to cancel.';
            }
            return;
        }
    }

    if (tool.includes('mask')) {
        if (state.isDrawing) {
            elements.instructions.textContent =
                'Click to add vertices. Click first vertex or press Enter to close polygon. Esc to cancel.';
        } else {
            elements.instructions.textContent = 'Click to start drawing polygon mask';
        }
    } else if (tool.includes('top') || tool.includes('bottom') || tool.includes('middle')) {
        if (state.tempPoints.length === 1) {
            elements.instructions.textContent = 'Click to set the second endpoint of the line';
        } else {
            elements.instructions.textContent = 'Click to set the first endpoint of the line';
        }
    } else if (tool.includes('joint') || tool.includes('ee_')) {
        elements.instructions.textContent = 'Click to place keypoint. N=skip (occluded), Tab=occlude (still place).';
    }
}

function cancelDrawing() {
    state.isDrawing = false;
    state.tempPoints = [];
    state.edgeSelectionMode = false;
    state.hoveredEdgeIndex = null;
    state.hoveredEdgeTool = null;
    state.selectedEdges = [];            // Clear multi-edge selection
    drawAllAnnotations();
    updateInstructions();
    updateCursorClasses();
}

/**
 * Compute midline from top and bottom lines.
 * Returns [[midStart], [midEnd]] or null if either line is missing.
 */
function computeMidlineFromLines(topLine, bottomLine) {
    if (!topLine || topLine.length !== 2 || !bottomLine || bottomLine.length !== 2) {
        return null;
    }

    const [top0, top1] = topLine;
    const [bot0, bot1] = bottomLine;

    // Match endpoints by proximity (find which top point is closer to which bottom)
    const d00 = Math.hypot(top0[0] - bot0[0], top0[1] - bot0[1]);
    const d01 = Math.hypot(top0[0] - bot1[0], top0[1] - bot1[1]);

    let topLeft, topRight, botLeft, botRight;
    if (d00 < d01) {
        // top0 matches bot0, top1 matches bot1
        topLeft = top0; botLeft = bot0;
        topRight = top1; botRight = bot1;
    } else {
        // top0 matches bot1, top1 matches bot0
        topLeft = top0; botLeft = bot1;
        topRight = top1; botRight = bot0;
    }

    // Midline = midpoints of matched pairs
    const midStart = [(topLeft[0] + botLeft[0]) / 2, (topLeft[1] + botLeft[1]) / 2];
    const midEnd = [(topRight[0] + botRight[0]) / 2, (topRight[1] + botRight[1]) / 2];

    return [midStart, midEnd];
}

// ============================================================================
// Canvas Event Handling
// ============================================================================

function getCanvasCoords(e) {
    const containerRect = elements.canvasContainer.getBoundingClientRect();
    const screenX = e.clientX - containerRect.left;
    const screenY = e.clientY - containerRect.top;

    // Convert screen coords to image coords, subtracting border offset
    let x = (screenX - state.panX) / state.zoom - EDGE_BORDER;
    let y = (screenY - state.panY) / state.zoom - EDGE_BORDER;

    // Clamp to image bounds (border clicks snap to nearest edge)
    x = Math.max(0, Math.min(x, state.imageWidth));
    y = Math.max(0, Math.min(y, state.imageHeight));

    return [x, y];
}

// ---------------------------------------------------------------------------
// Mouse Down
// ---------------------------------------------------------------------------

function handleCanvasMouseDown(e) {
    const [x, y] = getCanvasCoords(e);
    state.mouseDownPos = [x, y];
    state.mouseDownButton = e.button;

    // Priority 1: Pan (Space held or right-click drag)
    if (state.spaceHeld || e.button === 2) {
        state.isPanning = true;
        state.panStartX = e.clientX;
        state.panStartY = e.clientY;
        state.panStartPanX = state.panX;
        state.panStartPanY = state.panY;
        updateCursorClasses();
        e.preventDefault();
        return;
    }

    // Middle-click = cancel drawing
    if (e.button === 1) {
        e.preventDefault();
        cancelDrawing();
        return;
    }

    // Priority 1.5: Peg handle drag / Edit All rectangle-select-then-drag
    if (state.annotationMode === 'pegs' && !state.pegDrawingTool && state.pegKeypointIdx === null
        && !state.pegboardTool && state.currentAnnotation && e.button === 0) {

        // Edit All: rectangle select phase — start drawing selection rect
        if (state.pegEditMode && state.pegEditPhase === 'selecting') {
            state.pegEditRect = {x1: x, y1: y, x2: x, y2: y};
            pushUndo();
            e.preventDefault();
            return;
        }

        // Edit All: selected phase — drag selection or click outside to deselect
        if (state.pegEditMode && state.pegEditPhase === 'selected' && state.pegEditRect) {
            const r = state.pegEditRect;
            const rx1 = Math.min(r.x1, r.x2), ry1 = Math.min(r.y1, r.y2);
            const rx2 = Math.max(r.x1, r.x2), ry2 = Math.max(r.y1, r.y2);
            if (x >= rx1 && x <= rx2 && y >= ry1 && y <= ry2) {
                // Inside selection → start drag
                state.pegEditDragStart = {x, y};
                pushUndo();
                e.preventDefault();
                return;
            } else {
                // Outside selection → clear, go back to selecting
                state.pegEditPhase = 'selecting';
                state.pegEditRect = null;
                state.pegEditSelectedItems = [];
                state.pegEditDragStart = null;
                drawAllAnnotations();
                showToast('Selection cleared — draw new rectangle');
                e.preventDefault();
                return;
            }
        }

        // In Edit All mode (non-rect-select), check pegboard handles first, then peg handles
        if (state.pegEditMode) {
            const boardHandle = findNearestPegboardHandle(x, y, 12 / state.zoom);
            if (boardHandle) {
                state.isDraggingPegHandle = true;
                state.draggedPegHandle = boardHandle;
                pushUndo();
                updateCursorClasses('hover-vertex');
                e.preventDefault();
                return;
            }
        }
        const handle = findNearestPegHandle(x, y, 12 / state.zoom);
        if (handle) {
            state.isDraggingPegHandle = true;
            state.draggedPegHandle = handle;
            pushUndo();
            updateCursorClasses('hover-vertex');
            e.preventDefault();
            return;
        }
    }

    // Priority 2: Edit mode - start vertex/endpoint/keypoint drag
    if (state.editMode && state.currentAnnotation) {
        // Check keypoints first (highest priority in edit mode)
        const nearKeypoint = findNearestTooltip(x, y, 15 / state.zoom);
        if (nearKeypoint) {
            state.isDraggingKeypoint = true;
            state.draggedKeypointKey = nearKeypoint.keypointKey;
            state.dragStartPos = [x, y];
            pushUndo();
            updateCursorClasses();
            e.preventDefault();
            return;
        }

        // Check line endpoints
        const nearEndpoint = findNearestLineEndpoint(x, y, 12 / state.zoom);
        if (nearEndpoint) {
            state.isDraggingLineEndpoint = true;
            state.draggedLineInfo = nearEndpoint;
            state.dragStartPos = [x, y];
            pushUndo();
            updateCursorClasses();
            e.preventDefault();
            return;
        }

        // Check polygon vertex
        const nearVertex = findNearestVertex(x, y, 10 / state.zoom);
        if (nearVertex) {
            state.selectedVertexTool = nearVertex.maskKey;
            state.selectedVertexIndex = nearVertex.vertexIndex;
            state.selectedVertices = [];  // Clear multi-select
            state.isDraggingVertex = true;
            state.dragStartPos = [x, y];
            pushUndo();
            drawAllAnnotations();
            updateCursorClasses();
            e.preventDefault();
            return;
        }

        // No hit → start box selection
        state.isBoxSelecting = true;
        state.boxSelectStart = [x, y];
        state.boxSelectEnd = [x, y];
        state.selectedVertices = [];
        state.selectedVertexIndex = null;
        state.selectedVertexTool = null;
        e.preventDefault();
    }
}

// ---------------------------------------------------------------------------
// Mouse Move
// ---------------------------------------------------------------------------

function handleCanvasMouseMoveNew(e) {
    // Priority 1: Panning
    if (state.isPanning) {
        state.panX = state.panStartPanX + (e.clientX - state.panStartX);
        state.panY = state.panStartPanY + (e.clientY - state.panStartY);
        applyTransform();
        return;
    }

    const [x, y] = getCanvasCoords(e);

    // Priority 1.5: Dragging peg handle
    if (state.isDraggingPegHandle && state.draggedPegHandle) {
        const h = state.draggedPegHandle;
        const peg = state.currentAnnotation?.pegs?.find(p => p.id === h.pegId);
        if (peg) {
            const rx = Math.round(x), ry = Math.round(y);
            if (h.handleType === 'keypoint') {
                peg.keypoints[h.index] = [rx, ry];
            } else if (h.handleType === 'bbox_corner') {
                // Move nearest mask vertex to keep bbox and mask consistent
                const [bx, by, bw, bh] = peg.bbox;
                const corners = [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]];
                const corner = corners[h.index];
                const dx = rx - corner[0], dy = ry - corner[1];
                if (peg.mask && peg.mask.length > 0) {
                    // Find mask vertex closest to this bbox corner
                    let closestIdx = 0, closestDist = Infinity;
                    peg.mask.forEach((v, i) => {
                        const d = (v[0] - corner[0]) ** 2 + (v[1] - corner[1]) ** 2;
                        if (d < closestDist) { closestDist = d; closestIdx = i; }
                    });
                    peg.mask[closestIdx] = [peg.mask[closestIdx][0] + dx, peg.mask[closestIdx][1] + dy];
                }
                // Recompute bbox from mask
                if (peg.mask && peg.mask.length > 0) {
                    const xs = peg.mask.map(v => v[0]), ys = peg.mask.map(v => v[1]);
                    peg.bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
                } else {
                    // No mask, just resize bbox directly
                    const opposite = [[bx + bw, by + bh], [bx, by + bh], [bx + bw, by], [bx, by]][h.index];
                    peg.bbox = [Math.min(rx, opposite[0]), Math.min(ry, opposite[1]), Math.abs(rx - opposite[0]), Math.abs(ry - opposite[1])];
                }
            } else if (h.handleType === 'mask_vertex') {
                peg.mask[h.index] = [rx, ry];
            }
            drawAllAnnotations();
        } else if (h.handleType === 'source_post' || h.handleType === 'target_post') {
            const pegboard = state.currentAnnotation?.pegboard;
            if (pegboard) {
                const key = h.handleType === 'source_post' ? 'source_posts' : 'target_posts';
                const masksKey = h.handleType === 'source_post' ? 'source_post_masks' : 'target_post_masks';
                if (pegboard[key]?.[h.index]) {
                    // Compute delta to translate mask polygon too
                    const oldPt = pegboard[key][h.index];
                    const dx = rx - oldPt[0], dy = ry - oldPt[1];
                    pegboard[key][h.index] = [rx, ry];
                    // Translate the corresponding mask polygon
                    const mask = pegboard[masksKey]?.[h.index];
                    if (mask && mask.length > 0) {
                        for (let vi = 0; vi < mask.length; vi++) {
                            mask[vi] = [mask[vi][0] + dx, mask[vi][1] + dy];
                        }
                    }
                    drawAllAnnotations();
                }
            }
        } else if (h.handleType === 'board_vertex') {
            const pegboard = state.currentAnnotation?.pegboard;
            if (pegboard?.board_mask?.[h.index]) {
                pegboard.board_mask[h.index] = [rx, ry];
                drawAllAnnotations();
            }
        } else if (h.handleType === 'post_mask_vertex') {
            const pegboard = state.currentAnnotation?.pegboard;
            if (pegboard?.[h.masksKey]?.[h.maskIndex]?.[h.index]) {
                pegboard[h.masksKey][h.maskIndex][h.index] = [rx, ry];
                drawAllAnnotations();
            }
        }
        return;
    }

    // Priority 2: Dragging line endpoint
    if (state.isDraggingLineEndpoint && state.draggedLineInfo) {
        const info = state.draggedLineInfo;
        state.currentAnnotation[info.linesKey][info.lineType][info.endpointIndex] = [x, y];
        drawAllAnnotations();
        return;
    }

    // Priority 3: Dragging keypoint
    if (state.isDraggingKeypoint && state.draggedKeypointKey) {
        state.currentAnnotation[state.draggedKeypointKey] = [x, y];
        drawAllAnnotations();
        return;
    }

    // Priority 4: Dragging vertex
    if (state.isDraggingVertex && state.selectedVertexTool !== null) {
        const polygon = state.currentAnnotation[state.selectedVertexTool];
        if (polygon && state.selectedVertexIndex < polygon.length) {
            polygon[state.selectedVertexIndex] = [x, y];
            drawAllAnnotations();
        }
        return;
    }

    // Priority 4b: Box selecting vertices
    if (state.isBoxSelecting) {
        state.boxSelectEnd = [x, y];
        drawAllAnnotations();
        return;
    }

    // Priority 4c: SAM Apply Existing hover detection
    if (state.samApplyMode && state.precomputedSegments) {
        const segIdx = findSegmentAtPoint(x, y);
        if (segIdx !== state.hoveredSegmentIdx) {
            state.hoveredSegmentIdx = segIdx;
            drawAllAnnotations();
            drawPrecomputedOverlays();
        }
        return;
    }

    // Priority 5: Edit mode hover feedback
    if (state.editMode && state.currentAnnotation) {
        const nearTooltip = findNearestTooltip(x, y, 15 / state.zoom);
        const nearEndpoint = findNearestLineEndpoint(x, y, 12 / state.zoom);
        const nearVertex = findNearestVertex(x, y, 10 / state.zoom);

        if (nearTooltip || nearEndpoint) {
            state.hoveredVertexTool = null;
            state.hoveredVertexIndex = null;
            updateCursorClasses('hover-vertex');
        } else if (nearVertex) {
            state.hoveredVertexTool = nearVertex.maskKey;
            state.hoveredVertexIndex = nearVertex.vertexIndex;
            updateCursorClasses('hover-vertex');
        } else {
            state.hoveredVertexTool = null;
            state.hoveredVertexIndex = null;
            updateCursorClasses();
        }
        drawAllAnnotations();
        return;
    }

    // Priority 6: Edge selection hover
    if (state.edgeSelectionMode && state.currentAnnotation) {
        const nearEdge = findNearestEdge(x, y, 15 / state.zoom);
        if (nearEdge) {
            state.hoveredEdgeIndex = nearEdge.edgeIndex;
            state.hoveredEdgeTool = nearEdge.maskKey;
        } else {
            state.hoveredEdgeIndex = null;
            state.hoveredEdgeTool = null;
        }
        drawAllAnnotations();
        return;
    }

    // Priority 6.5: Peg bbox rubber-banding / peg mask rubber-banding
    if (state.annotationMode === 'pegs') {
        if (state.pegDrawingTool === 'bbox' && state.pegBboxStart) {
            state.tempPoints = [[x, y]];
            drawAllAnnotations();
            return;
        }
        if (state.pegDrawingTool === 'mask' && state.isDrawing && state.tempPoints.length > 0) {
            drawAllAnnotations();
            const last = state.tempPoints[state.tempPoints.length - 1];
            annotationCtx.beginPath();
            annotationCtx.moveTo(last[0], last[1]);
            annotationCtx.lineTo(x, y);
            annotationCtx.strokeStyle = COLORS.temp;
            annotationCtx.lineWidth = 1;
            annotationCtx.setLineDash([5, 5]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);
            // Snap-to-first
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(first[0], first[1], 12, 0, Math.PI * 2);
                    annotationCtx.strokeStyle = COLORS.highlight;
                    annotationCtx.lineWidth = 3;
                    annotationCtx.stroke();
                }
            }
            return;
        }
        // Post polygon drawing rubber-band
        if (state.pegboardTool === 'post' && state.isDrawing && state.tempPoints.length > 0) {
            drawAllAnnotations();
            const last = state.tempPoints[state.tempPoints.length - 1];
            const postColor = state.activePostTarget?.type === 'source' ? '#3b82f6' : '#10b981';
            annotationCtx.beginPath();
            annotationCtx.moveTo(last[0], last[1]);
            annotationCtx.lineTo(x, y);
            annotationCtx.strokeStyle = postColor;
            annotationCtx.lineWidth = 1;
            annotationCtx.setLineDash([5, 5]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);
            // Snap-to-first indicator
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(first[0], first[1], 12, 0, Math.PI * 2);
                    annotationCtx.strokeStyle = COLORS.highlight;
                    annotationCtx.lineWidth = 3;
                    annotationCtx.stroke();
                }
            }
            return;
        }
        if (state.pegboardTool === 'outline' && state.isDrawing && state.tempPoints.length > 0) {
            drawAllAnnotations();
            const last = state.tempPoints[state.tempPoints.length - 1];
            annotationCtx.beginPath();
            annotationCtx.moveTo(last[0], last[1]);
            annotationCtx.lineTo(x, y);
            annotationCtx.strokeStyle = 'rgba(255,255,255,0.5)';
            annotationCtx.lineWidth = 1;
            annotationCtx.setLineDash([5, 5]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);
            // Snap-to-first indicator
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(first[0], first[1], 12, 0, Math.PI * 2);
                    annotationCtx.strokeStyle = COLORS.highlight;
                    annotationCtx.lineWidth = 3;
                    annotationCtx.stroke();
                }
            }
            return;
        }
        // Post mask drawing rubber-band
        if (state.postMaskMode === 'drawing' && state.isDrawing && state.tempPoints.length > 0) {
            drawAllAnnotations();
            const last = state.tempPoints[state.tempPoints.length - 1];
            const postColor = state.postMaskTarget?.type === 'source' ? '#3b82f6' : '#10b981';
            annotationCtx.beginPath();
            annotationCtx.moveTo(last[0], last[1]);
            annotationCtx.lineTo(x, y);
            annotationCtx.strokeStyle = postColor;
            annotationCtx.lineWidth = 1;
            annotationCtx.setLineDash([5, 5]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);
            // Snap-to-first indicator
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(first[0], first[1], 12, 0, Math.PI * 2);
                    annotationCtx.strokeStyle = COLORS.highlight;
                    annotationCtx.lineWidth = 3;
                    annotationCtx.stroke();
                }
            }
            return;
        }
        // Edit All: rectangle selection rubber-band
        if (state.pegEditMode && state.pegEditPhase === 'selecting' && state.pegEditRect) {
            state.pegEditRect.x2 = x;
            state.pegEditRect.y2 = y;
            drawAllAnnotations();
            // Draw selection rectangle
            const r = state.pegEditRect;
            annotationCtx.beginPath();
            annotationCtx.rect(Math.min(r.x1, r.x2), Math.min(r.y1, r.y2),
                Math.abs(r.x2 - r.x1), Math.abs(r.y2 - r.y1));
            annotationCtx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
            annotationCtx.lineWidth = 2 / state.zoom;
            annotationCtx.setLineDash([6 / state.zoom, 4 / state.zoom]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);
            annotationCtx.fillStyle = 'rgba(59, 130, 246, 0.08)';
            annotationCtx.fill();
            return;
        }

        // Edit All: drag selected items
        if (state.pegEditMode && state.pegEditPhase === 'selected' && state.pegEditDragStart) {
            const dx = x - state.pegEditDragStart.x;
            const dy = y - state.pegEditDragStart.y;
            translatePegEditSelection(dx, dy);
            state.pegEditDragStart = {x, y};
            // Also shift the selection rectangle
            if (state.pegEditRect) {
                state.pegEditRect.x1 += dx; state.pegEditRect.y1 += dy;
                state.pegEditRect.x2 += dx; state.pegEditRect.y2 += dy;
            }
            drawAllAnnotations();
            drawPegEditSelectionOverlay();
            return;
        }

        // Edit All: draw selection overlay when selected (no drag)
        if (state.pegEditMode && state.pegEditPhase === 'selected') {
            drawAllAnnotations();
            drawPegEditSelectionOverlay();
            // Show move cursor inside selection
            if (state.pegEditRect) {
                const r = state.pegEditRect;
                const rx1 = Math.min(r.x1, r.x2), ry1 = Math.min(r.y1, r.y2);
                const rx2 = Math.max(r.x1, r.x2), ry2 = Math.max(r.y1, r.y2);
                if (x >= rx1 && x <= rx2 && y >= ry1 && y <= ry2) {
                    updateCursorClasses('hover-vertex');
                } else {
                    updateCursorClasses();
                }
            }
            return;
        }

        // Hover feedback for peg handles (no tool active)
        if (!state.pegDrawingTool && state.pegKeypointIdx === null && !state.pegboardTool) {
            const pegHandle = findNearestPegHandle(x, y, 12 / state.zoom);
            const boardHandle = state.pegEditMode ? findNearestPegboardHandle(x, y, 12 / state.zoom) : null;
            const handle = pegHandle || boardHandle;
            if (handle) {
                updateCursorClasses('hover-vertex');
                // Draw highlight circle around hovered handle
                drawAllAnnotations();
                annotationCtx.beginPath();
                annotationCtx.arc(handle.x, handle.y, 8 / state.zoom, 0, Math.PI * 2);
                annotationCtx.fillStyle = 'rgba(59, 130, 246, 0.3)';
                annotationCtx.fill();
                annotationCtx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
                annotationCtx.lineWidth = 2 / state.zoom;
                annotationCtx.stroke();
            } else {
                updateCursorClasses();
            }
        }
    }

    // Priority 7: Drawing - line from last point to cursor (existing behavior)
    if (state.isDrawing && state.currentTool?.includes('mask')) {
        drawAllAnnotations();
        if (state.tempPoints.length > 0) {
            const last = state.tempPoints[state.tempPoints.length - 1];
            annotationCtx.beginPath();
            annotationCtx.moveTo(last[0], last[1]);
            annotationCtx.lineTo(x, y);
            annotationCtx.strokeStyle = COLORS.temp;
            annotationCtx.lineWidth = 1;
            annotationCtx.setLineDash([5, 5]);
            annotationCtx.stroke();
            annotationCtx.setLineDash([]);

            // Snap-to-first indicator (Feature 8)
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(first[0], first[1], 12, 0, Math.PI * 2);
                    annotationCtx.strokeStyle = COLORS.highlight;
                    annotationCtx.lineWidth = 3;
                    annotationCtx.stroke();
                }
            }
        }
        return;
    }
}

// ---------------------------------------------------------------------------
// Mouse Up
// ---------------------------------------------------------------------------

function handleCanvasMouseUp(e) {
    const [x, y] = getCanvasCoords(e);

    // End panning
    if (state.isPanning) {
        state.isPanning = false;
        updateCursorClasses();
        return;
    }

    // End Edit All rectangle selection
    if (state.pegEditMode && state.pegEditPhase === 'selecting' && state.pegEditRect) {
        state.pegEditRect.x2 = x;
        state.pegEditRect.y2 = y;
        const items = findItemsInPegEditRect(state.pegEditRect);
        if (items.length > 0) {
            state.pegEditSelectedItems = items;
            state.pegEditPhase = 'selected';
            drawAllAnnotations();
            drawPegEditSelectionOverlay();
            showToast(`Selected ${items.length} element(s) — drag to move, click outside to deselect`);
        } else {
            state.pegEditRect = null;
            drawAllAnnotations();
            showToast('No annotations in selection — try again');
        }
        return;
    }

    // End Edit All drag
    if (state.pegEditMode && state.pegEditPhase === 'selected' && state.pegEditDragStart) {
        state.pegEditDragStart = null;
        state.annotationDirty = true;
        saveAnnotations();
        drawAllAnnotations();
        drawPegEditSelectionOverlay();
        return;
    }

    // End peg handle drag
    if (state.isDraggingPegHandle) {
        state.isDraggingPegHandle = false;
        state.draggedPegHandle = null;
        state.annotationDirty = true;
        saveAnnotations();
        drawAllAnnotations();
        updateCursorClasses();
        return;
    }

    // End line endpoint drag
    if (state.isDraggingLineEndpoint) {
        state.isDraggingLineEndpoint = false;
        state.draggedLineInfo = null;
        saveAnnotations();
        updateCursorClasses();
        return;
    }

    // End keypoint drag
    if (state.isDraggingKeypoint) {
        state.isDraggingKeypoint = false;
        state.draggedKeypointKey = null;
        saveAnnotations();
        drawAllAnnotations();
        updateCursorClasses();
        return;
    }

    // End vertex drag
    if (state.isDraggingVertex) {
        state.isDraggingVertex = false;
        saveAnnotations();
        drawAllAnnotations();
        updateCursorClasses();
        return;
    }

    // End box selection
    if (state.isBoxSelecting) {
        state.isBoxSelecting = false;
        const start = state.boxSelectStart;
        const end = state.boxSelectEnd;

        // Check if it was a real drag (> 3px) or just a click
        const dragDist = Math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2);
        if (dragDist >= 3 / state.zoom) {
            // Compute bounding box
            const minX = Math.min(start[0], end[0]);
            const maxX = Math.max(start[0], end[0]);
            const minY = Math.min(start[1], end[1]);
            const maxY = Math.max(start[1], end[1]);

            // Find all vertices inside the box
            const selected = [];
            for (const maskKey of ['tool1_mask', 'tool2_mask']) {
                const polygon = state.currentAnnotation?.[maskKey];
                if (!polygon || polygon.length < 3) continue;
                for (let i = 0; i < polygon.length; i++) {
                    const [vx, vy] = polygon[i];
                    if (vx >= minX && vx <= maxX && vy >= minY && vy <= maxY) {
                        selected.push({ maskKey, vertexIndex: i });
                    }
                }
            }

            state.selectedVertices = selected;
            state.boxSelectStart = null;
            state.boxSelectEnd = null;

            if (selected.length > 0) {
                showToast(`${selected.length} vertices selected`);
            }
            drawAllAnnotations();
            state.mouseDownPos = null;
            state.mouseDownButton = -1;
            return;
        }
        // If drag was too small, fall through to click handler
        state.boxSelectStart = null;
        state.boxSelectEnd = null;
    }

    // Detect click (mousedown + mouseup with < 3px movement)
    if (state.mouseDownPos && e.button === 0) {
        const dx = x - state.mouseDownPos[0];
        const dy = y - state.mouseDownPos[1];
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 3 / state.zoom) {
            handleCanvasClick_internal(x, y, e.shiftKey);
        }
    }

    state.mouseDownPos = null;
    state.mouseDownButton = -1;
}

// ---------------------------------------------------------------------------
// Internal click handler (called from mouseup if it's a click)
// ---------------------------------------------------------------------------

function handleCanvasClick_internal(x, y, shiftKey) {
    if (!state.currentAnnotation) return;

    // Priority 0a: SAM Apply Existing mode click - toggle multi-select
    if (state.samApplyMode && state.hoveredSegmentIdx !== null) {
        toggleSegmentSelection(state.hoveredSegmentIdx);
        return;
    }

    // Priority 0b: SAM interactive mode click
    if (state.samMode) {
        handleSamClick(x, y, shiftKey);
        return;
    }

    // Priority 1: Edit mode click
    if (state.editMode) {
        handleEditModeClick(x, y);
        return;
    }

    // Priority 2: Edge selection click
    if (state.edgeSelectionMode) {
        handleEdgeSelectionClick(x, y);
        return;
    }

    // Priority 2.5: Peg/Pegboard drawing (pegs mode)
    if (state.annotationMode === 'pegs') {
        handlePegCanvasClick(x, y);
        return;
    }

    // Priority 3: Normal drawing tool behavior
    if (!state.currentTool) return;

    const tool = state.currentTool;
    pushUndo();

    if (tool.includes('mask')) {
        handleMaskClick(x, y, tool);
    } else if (tool.includes('top') || tool.includes('bottom') || tool.includes('middle')) {
        // Check if edge selection should activate (not for midlines)
        if (!tool.includes('middle') && currentToolMaskExists() && state.tempPoints.length === 0) {
            enterEdgeSelectionMode();
            handleEdgeSelectionClick(x, y);
            return;
        }
        handleLineClick(x, y, tool);
    } else if (tool.includes('joint') || tool.includes('ee_')) {
        handleKeypointClick(x, y, tool);
    }
}

// ---------------------------------------------------------------------------
// Rubber-band line drawing helper
// ---------------------------------------------------------------------------

function drawRubberBandLine(from, to) {
    annotationCtx.beginPath();
    annotationCtx.moveTo(from[0], from[1]);
    annotationCtx.lineTo(to[0], to[1]);
    annotationCtx.strokeStyle = COLORS.highlight;
    annotationCtx.lineWidth = 2;
    annotationCtx.setLineDash([8, 4]);
    annotationCtx.stroke();
    annotationCtx.setLineDash([]);
}

// ---------------------------------------------------------------------------
// Cursor class management
// ---------------------------------------------------------------------------

function updateCursorClasses(extra = '') {
    const container = elements.canvasContainer;
    // Keep base class
    let classes = 'canvas-container';

    if (state.isPanning) {
        classes += ' panning';
    } else if (state.spaceHeld) {
        classes += ' pan-ready';
    } else if (state.editMode) {
        classes += ' edit-mode';
        if (extra === 'hover-vertex' || state.isDraggingVertex || state.isDraggingLineEndpoint || state.isDraggingKeypoint) {
            classes += state.isDraggingVertex || state.isDraggingLineEndpoint || state.isDraggingKeypoint
                ? ' dragging-vertex' : ' hover-vertex';
        }
    } else if (state.samApplyMode) {
        classes += ' sam-apply-mode';
    } else if (state.samMode) {
        classes += ' sam-mode';
    } else if (state.edgeSelectionMode) {
        classes += ' edge-select';
    } else if (state.currentTool) {
        if (state.currentTool.includes('mask')) {
            classes += ' drawing-mask';
        } else if (state.currentTool.includes('top') || state.currentTool.includes('bottom') || state.currentTool.includes('middle')) {
            classes += ' drawing-line';
        } else if (state.currentTool.includes('joint') || state.currentTool.includes('ee_')) {
            classes += ' drawing-point';
        }
    }

    container.className = classes;
}

// ---------------------------------------------------------------------------
// Scroll Wheel Zoom (Feature 2)
// ---------------------------------------------------------------------------

function handleWheelZoom(e) {
    e.preventDefault();

    const containerRect = elements.canvasContainer.getBoundingClientRect();
    const screenX = e.clientX - containerRect.left;
    const screenY = e.clientY - containerRect.top;

    // Point under cursor in image space (before zoom)
    const oldZoom = state.zoom;
    const imgX = (screenX - state.panX) / oldZoom;
    const imgY = (screenY - state.panY) / oldZoom;

    // Apply zoom
    if (e.deltaY < 0) {
        state.zoom = Math.min(state.zoom * 1.15, 5.0);
    } else {
        state.zoom = Math.max(state.zoom / 1.15, 0.1);
    }

    if (state.zoom === oldZoom) return;

    // Adjust pan so same image point stays under cursor
    state.panX = screenX - imgX * state.zoom;
    state.panY = screenY - imgY * state.zoom;

    updateZoom();
}

// ============================================================================
// Edge Selection Mode (Features 5, 6)
// ============================================================================

function enterEdgeSelectionMode() {
    state.edgeSelectionMode = true;
    state.hoveredEdgeIndex = null;
    state.hoveredEdgeTool = null;
    updateCursorClasses();
    updateInstructions();
}

function exitEdgeSelectionMode() {
    state.edgeSelectionMode = false;
    state.hoveredEdgeIndex = null;
    state.hoveredEdgeTool = null;
    state.selectedEdges = [];  // Clear multi-edge selection
    updateCursorClasses();
}

function handleEdgeSelectionClick(x, y) {
    const tool = state.currentTool;
    if (!tool) return;

    const maskKey = getMaskKeyForTool(tool);
    const polygon = state.currentAnnotation[maskKey];
    if (!polygon || polygon.length < 3) return;

    // Find nearest edge on the correct mask
    let bestEdge = null;
    for (let i = 0; i < polygon.length; i++) {
        const j = (i + 1) % polygon.length;
        const result = pointToSegmentDistance(x, y, polygon[i][0], polygon[i][1], polygon[j][0], polygon[j][1]);
        if (result.distance < 20 / state.zoom && (!bestEdge || result.distance < bestEdge.distance)) {
            bestEdge = { edgeIndex: i, distance: result.distance };
        }
    }

    if (!bestEdge) return; // No edge near click

    const edgeIdx = bestEdge.edgeIndex;
    const nextIdx = (edgeIdx + 1) % polygon.length;
    const p1 = polygon[edgeIdx];
    const p2 = polygon[nextIdx];

    if (tool.includes('top') || tool.includes('bottom')) {
        // Two-click workflow: click leftmost edge, then rightmost edge
        const existingIdx = state.selectedEdges.findIndex(e =>
            e.edgeIndex === edgeIdx && e.maskKey === maskKey);

        if (existingIdx >= 0) {
            // Deselect this edge (allow correction)
            state.selectedEdges.splice(existingIdx, 1);
        } else {
            // Select this edge
            state.selectedEdges.push({
                edgeIndex: edgeIdx,
                maskKey: maskKey,
                p1: [...p1],
                p2: [...p2]
            });
        }

        drawAllAnnotations();
        updateInstructions();

        // Snap mode (single-edge snap ON): auto-finalize after 1 edge
        if (state.useEdgeSelection && state.selectedEdges.length === 1) {
            finalizeLineFromEdges();
            return;
        }
    }
}

/**
 * Finalize line from multiple selected edges.
 * Finds the extreme points (leftmost/rightmost) from all selected edge endpoints
 * and creates the top/bottom line spanning those extremes.
 */
async function finalizeLineFromEdges() {
    if (state.selectedEdges.length === 0) return;

    const tool = state.currentTool;
    if (!tool || (!tool.includes('top') && !tool.includes('bottom'))) return;

    // Collect all edge endpoints
    const points = state.selectedEdges.flatMap(e => [e.p1, e.p2]);

    // Find extreme points (leftmost and rightmost)
    const minXPoint = points.reduce((a, b) => a[0] < b[0] ? a : b);
    const maxXPoint = points.reduce((a, b) => a[0] > b[0] ? a : b);

    const lineType = tool.includes('top') ? 'top' : 'bottom';
    const linesKey = tool.startsWith('tool1') ? 'tool1_lines' : 'tool2_lines';

    pushUndo();
    state.currentAnnotation[linesKey][lineType] = [[...minXPoint], [...maxXPoint]];

    const edgeCount = state.selectedEdges.length;
    state.selectedEdges = [];
    exitEdgeSelectionMode();
    state.tempPoints = [];
    await saveAnnotations();
    drawAllAnnotations();
    updateInstructions();
    showToast(`${lineType} line set from ${edgeCount} edge${edgeCount > 1 ? 's' : ''}`);

    // Auto-advance to next tool
    advanceToNextTool(tool);
}

// ============================================================================
// Edit Mode (Feature 7)
// ============================================================================

function toggleEditMode() {
    state.editMode = !state.editMode;

    const btn = document.getElementById('edit-mode-btn');
    if (btn) {
        btn.classList.toggle('active', state.editMode);
    }

    if (state.editMode) {
        // Exit SAM apply mode if active
        if (state.samApplyMode) exitApplyExistingMode();
        if (state.samMode) exitSamMode();
        // Exit any drawing tool
        if (state.currentTool) {
            selectTool(state.currentTool); // deselect
        }
        cancelDrawing();
        elements.currentToolIndicator.textContent = 'Edit Mode';
        elements.instructions.textContent = 'Drag vertices/keypoints/endpoints to move. Click to select, Delete to remove.';
    } else {
        state.selectedVertexIndex = null;
        state.selectedVertexTool = null;
        state.hoveredVertexIndex = null;
        state.hoveredVertexTool = null;
        state.selectedVertices = [];
        state.selectedKeypointKey = null;
        state.isDraggingKeypoint = false;
        state.draggedKeypointKey = null;
        state.isBoxSelecting = false;
        state.boxSelectStart = null;
        state.boxSelectEnd = null;
        elements.currentToolIndicator.textContent = 'No tool selected';
        elements.instructions.textContent = 'Select a tool from the sidebar to begin annotating';
    }

    updateCursorClasses();
    drawAllAnnotations();
}

function handleEditModeClick(x, y) {
    // Check if clicking on an existing keypoint → select it
    const nearKeypoint = findNearestTooltip(x, y, 15 / state.zoom);
    if (nearKeypoint) {
        state.selectedKeypointKey = nearKeypoint.keypointKey;
        state.selectedVertexIndex = null;
        state.selectedVertexTool = null;
        state.selectedVertices = [];
        drawAllAnnotations();
        return;
    }

    // Check if clicking on an existing vertex → select it (single)
    const nearVertex = findNearestVertex(x, y, 10 / state.zoom);
    if (nearVertex) {
        state.selectedKeypointKey = null;  // Clear keypoint selection
        state.selectedVertexTool = nearVertex.maskKey;
        state.selectedVertexIndex = nearVertex.vertexIndex;
        state.selectedVertices = [];  // Clear multi-select
        drawAllAnnotations();
        return;
    }

    // Check if clicking on an edge → insert vertex
    const nearEdge = findNearestEdge(x, y, 10 / state.zoom);
    if (nearEdge) {
        pushUndo();
        const polygon = state.currentAnnotation[nearEdge.maskKey];
        const insertAt = nearEdge.edgeIndex + 1;
        polygon.splice(insertAt, 0, [x, y]);
        state.selectedVertexTool = nearEdge.maskKey;
        state.selectedVertexIndex = insertAt;
        state.selectedVertices = [];  // Clear multi-select
        saveAnnotations();
        drawAllAnnotations();
        return;
    }

    // Click on empty space → deselect all
    state.selectedVertexIndex = null;
    state.selectedVertexTool = null;
    state.selectedVertices = [];
    state.selectedKeypointKey = null;
    drawAllAnnotations();
}

function deleteSelectedVertex() {
    if (!state.editMode || state.selectedVertexIndex === null || !state.selectedVertexTool) {
        return false;
    }

    const polygon = state.currentAnnotation[state.selectedVertexTool];
    if (!polygon || polygon.length <= 3) {
        showToast('Cannot delete: minimum 3 vertices required', true);
        return false;
    }

    pushUndo();
    polygon.splice(state.selectedVertexIndex, 1);
    state.selectedVertexIndex = null;
    state.selectedVertexTool = null;
    saveAnnotations();
    drawAllAnnotations();
    showToast('Vertex deleted');
    return true;
}

function deleteSelectedVertices() {
    if (!state.editMode || state.selectedVertices.length === 0) return false;

    // Group by maskKey
    const groups = {};
    for (const sv of state.selectedVertices) {
        if (!groups[sv.maskKey]) groups[sv.maskKey] = [];
        groups[sv.maskKey].push(sv.vertexIndex);
    }

    // Validate: each polygon must retain at least 3 vertices
    for (const [maskKey, indices] of Object.entries(groups)) {
        const polygon = state.currentAnnotation[maskKey];
        if (!polygon) continue;
        if (polygon.length - indices.length < 3) {
            showToast(`Cannot delete: ${maskKey} would have fewer than 3 vertices`, true);
            return false;
        }
    }

    pushUndo();

    // Sort indices descending and splice for each polygon
    for (const [maskKey, indices] of Object.entries(groups)) {
        const polygon = state.currentAnnotation[maskKey];
        if (!polygon) continue;
        indices.sort((a, b) => b - a);
        for (const idx of indices) {
            polygon.splice(idx, 1);
        }
    }

    const count = state.selectedVertices.length;
    state.selectedVertices = [];
    state.selectedVertexIndex = null;
    state.selectedVertexTool = null;
    saveAnnotations();
    drawAllAnnotations();
    showToast(`${count} vertices deleted`);
    return true;
}

// ============================================================================
// Canvas Click Handlers (existing, refactored)
// ============================================================================

function handleMaskClick(x, y, tool) {
    if (!state.isDrawing) {
        // Start new polygon
        state.isDrawing = true;
        state.tempPoints = [[x, y]];
    } else {
        // Check if clicking near first point to close
        const first = state.tempPoints[0];
        const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);

        if (dist < 15 / state.zoom && state.tempPoints.length >= 3) {
            // Close polygon
            finishPolygon(tool);
        } else {
            // Add vertex
            state.tempPoints.push([x, y]);
        }
    }

    drawAllAnnotations();
    updateInstructions();
}

async function finishPolygon(tool) {
    const maskKey = tool.startsWith('tool1') ? 'tool1_mask' : 'tool2_mask';
    state.currentAnnotation[maskKey] = [...state.tempPoints];
    state.isDrawing = false;
    state.tempPoints = [];
    await saveAnnotations();

    // Auto-advance to next tool
    advanceToNextTool(tool);
}

async function handleLineClick(x, y, tool) {
    if (state.tempPoints.length === 0) {
        // First point
        state.tempPoints = [[x, y]];
    } else {
        // Second point - complete line
        const lineType = tool.includes('top') ? 'top' : tool.includes('middle') ? 'middle' : 'bottom';
        const linesKey = tool.startsWith('tool1') ? 'tool1_lines' : 'tool2_lines';

        state.currentAnnotation[linesKey][lineType] = [state.tempPoints[0], [x, y]];
        state.tempPoints = [];
        await saveAnnotations();

        // Auto-advance to next tool
        advanceToNextTool(tool);
    }

    drawAllAnnotations();
    updateInstructions();
}

/**
 * Handle keypoint placement click.
 */
function handleKeypointClick(x, y, tool) {
    if (!state.currentAnnotation) return;

    // Set the keypoint
    state.currentAnnotation[tool] = [x, y];
    saveAnnotations();
    drawAllAnnotations();

    // Auto-toggle to next keypoint in sequence
    autoToggleToNextKeypoint(tool);
}

/**
 * Auto-toggle to next keypoint in sequence.
 * Sequence: joint → ee_tip → ee_left → ee_right
 * Skips keypoints marked as "Out" (-1).
 * When finishing a tool's keypoints, uses auto-advance to cross to next tool.
 */
function autoToggleToNextKeypoint(currentTool) {
    const toolPrefix = currentTool.startsWith('tool1') ? 'tool1' : 'tool2';
    const keypointType = currentTool.replace(`${toolPrefix}_`, '');
    const sequence = ['joint', 'ee_tip', 'ee_left', 'ee_right'];

    const currentIdx = sequence.indexOf(keypointType);
    if (currentIdx < 0) return;

    const vis = state.currentAnnotation[`${toolPrefix}_visibility`] || {};

    // Find next keypoint in sequence that's not Out (-1)
    for (let i = currentIdx + 1; i < sequence.length; i++) {
        const nextVis = vis[sequence[i]];
        // Skip if marked as Out (-1), but include Visible (1) and Occluded (0)
        if (nextVis !== -1) {
            const nextTool = `${toolPrefix}_${sequence[i]}`;
            selectTool(nextTool);
            return;
        }
    }

    // No more keypoints in this tool - use auto-advance to cross to next tool
    if (state.autoAdvance) {
        advanceToNextTool(currentTool);
    } else {
        // No auto-advance - just deselect
        selectTool(null);
    }
}

/**
 * Get the visibility key for a tool (maps tool names to visibility structure keys).
 * @param {string} tool - Tool name like 'tool1_mask', 'tool1_top', 'tool1_joint'
 * @returns {{toolPrefix: string, visKey: string, partKey: string, partName: string}} - Tool info
 */
function getToolVisibilityInfo(tool) {
    if (!tool) return null;

    const toolPrefix = tool.startsWith('tool1') ? 'tool1' : 'tool2';
    const part = tool.replace(`${toolPrefix}_`, '');

    // Map tool part to visibility key
    // Lines (top/bottom/middle) share 'lines' visibility
    let partKey = part;
    if (part === 'top' || part === 'bottom' || part === 'middle') {
        partKey = 'lines';
    }

    // Human-readable part name
    const partNames = {
        'mask': 'mask',
        'top': 'shaft',
        'bottom': 'shaft',
        'middle': 'midline',
        'lines': 'shaft',
        'joint': 'joint',
        'ee_tip': 'ee_tip',
        'ee_left': 'ee_left',
        'ee_right': 'ee_right'
    };

    return {
        toolPrefix,
        visKey: `${toolPrefix}_visibility`,
        partKey,
        partName: partNames[part] || part
    };
}

/**
 * Skip current tool (mark as occluded) and move to next tool in progression.
 * Works for masks, lines (shaft), and keypoints.
 */
function skipCurrentTool() {
    const tool = state.currentTool;
    if (!tool || !state.currentAnnotation) return;

    const info = getToolVisibilityInfo(tool);
    if (!info) return;

    pushUndo();

    // Mark current as occluded (0)
    migrateVisibility(state.currentAnnotation);
    state.currentAnnotation[info.visKey][info.partKey] = 0;

    saveAnnotations();
    updateVisibilitySelects();
    updateStatus();
    updateToolButtonStatus();
    updateJsonViewer();

    // Move to next tool in progression
    const nextTool = getNextTool(tool);
    if (nextTool) {
        selectTool(nextTool);
        showToast(`${info.partName} marked occluded → ${formatToolName(nextTool)}`);
    } else {
        showToast(`${info.partName} marked occluded (end of progression)`);
        selectTool(null);
    }
}

/**
 * Mark current tool as occluded without moving to next.
 * Allows user to still interact with the tool if desired.
 * Works for masks, lines (shaft), and keypoints.
 */
function markCurrentToolOccluded() {
    const tool = state.currentTool;
    if (!tool || !state.currentAnnotation) return;

    const info = getToolVisibilityInfo(tool);
    if (!info) return;

    pushUndo();

    migrateVisibility(state.currentAnnotation);
    state.currentAnnotation[info.visKey][info.partKey] = 0;

    saveAnnotations();
    updateVisibilitySelects();
    updateStatus();
    updateToolButtonStatus();
    drawAllAnnotations();
    updateJsonViewer();

    showToast(`${info.partName} marked occluded (click to place)`);
}

// Legacy function names for backwards compatibility
function skipToNextKeypoint() {
    skipCurrentTool();
}

function markKeypointOccluded() {
    markCurrentToolOccluded();
}

/**
 * Mark all components of a tool as Out (-1).
 */
function markToolAsOut(toolNum) {
    if (!state.currentAnnotation) return;

    pushUndo();

    const visKey = `tool${toolNum}_visibility`;
    migrateVisibility(state.currentAnnotation);

    state.currentAnnotation[visKey] = {
        mask: -1, lines: -1,
        joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
    };

    saveAnnotations();
    updateVisibilitySelects();
    updateStatus();
    updateToolButtonStatus();
    drawAllAnnotations();

    showToast(`Tool ${toolNum} marked as Out`);
}

/**
 * Mark all components of both tools as Out (-1).
 */
function markAllAsOut() {
    if (!state.currentAnnotation) return;

    pushUndo();

    for (const toolNum of [1, 2]) {
        const visKey = `tool${toolNum}_visibility`;
        state.currentAnnotation[visKey] = {
            mask: -1, lines: -1,
            joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
        };
    }

    saveAnnotations();
    updateVisibilitySelects();
    updateStatus();
    updateToolButtonStatus();
    drawAllAnnotations();

    showToast('All tools marked as Out');
}

/**
 * Draw highlighted edge on a polygon mask.
 */
function drawEdgeHighlight(maskKey, edgeIndex) {
    const polygon = state.currentAnnotation?.[maskKey];
    if (!polygon || polygon.length < 3) return;

    const nextIndex = (edgeIndex + 1) % polygon.length;
    const p1 = polygon[edgeIndex];
    const p2 = polygon[nextIndex];

    // Thick yellow edge line
    annotationCtx.beginPath();
    annotationCtx.moveTo(p1[0], p1[1]);
    annotationCtx.lineTo(p2[0], p2[1]);
    annotationCtx.strokeStyle = COLORS.highlight;
    annotationCtx.lineWidth = 4;
    annotationCtx.stroke();

    // Vertex dots at edge endpoints
    [p1, p2].forEach(p => {
        annotationCtx.beginPath();
        annotationCtx.arc(p[0], p[1], 7, 0, Math.PI * 2);
        annotationCtx.fillStyle = COLORS.highlight;
        annotationCtx.fill();
        annotationCtx.strokeStyle = '#fff';
        annotationCtx.lineWidth = 2;
        annotationCtx.stroke();
    });
}

/**
 * Draw all selected edges for multi-edge selection (top/bottom lines).
 * Uses a different color (cyan) to distinguish from hover highlight (yellow).
 */
function drawSelectedEdges() {
    if (state.selectedEdges.length === 0) return;

    // Draw each selected edge with cyan highlight
    state.selectedEdges.forEach((edge, idx) => {
        const { p1, p2 } = edge;

        // Thick cyan edge line
        annotationCtx.beginPath();
        annotationCtx.moveTo(p1[0], p1[1]);
        annotationCtx.lineTo(p2[0], p2[1]);
        annotationCtx.strokeStyle = '#00ffff';  // Cyan for selected
        annotationCtx.lineWidth = 4;
        annotationCtx.stroke();

        // Vertex dots at edge endpoints
        [p1, p2].forEach(p => {
            annotationCtx.beginPath();
            annotationCtx.arc(p[0], p[1], 6, 0, Math.PI * 2);
            annotationCtx.fillStyle = '#00ffff';
            annotationCtx.fill();
            annotationCtx.strokeStyle = '#fff';
            annotationCtx.lineWidth = 2;
            annotationCtx.stroke();
        });

        // Show edge number
        const midX = (p1[0] + p2[0]) / 2;
        const midY = (p1[1] + p2[1]) / 2;
        annotationCtx.font = 'bold 14px sans-serif';
        annotationCtx.fillStyle = '#000';
        annotationCtx.strokeStyle = '#fff';
        annotationCtx.lineWidth = 3;
        annotationCtx.strokeText(`${idx + 1}`, midX - 4, midY + 5);
        annotationCtx.fillText(`${idx + 1}`, midX - 4, midY + 5);
    });
}

/**
 * Draw edit mode overlays: hovered vertex enlargement, selected vertex highlight.
 */
function drawEditModeOverlays() {
    // Draw selected keypoint highlight (yellow ring)
    if (state.selectedKeypointKey) {
        const point = state.currentAnnotation?.[state.selectedKeypointKey];
        if (point && point.length === 2) {
            annotationCtx.beginPath();
            annotationCtx.arc(point[0], point[1], 16, 0, Math.PI * 2);
            annotationCtx.strokeStyle = COLORS.highlight;
            annotationCtx.lineWidth = 3;
            annotationCtx.stroke();
            annotationCtx.beginPath();
            annotationCtx.arc(point[0], point[1], 13, 0, Math.PI * 2);
            annotationCtx.strokeStyle = '#fff';
            annotationCtx.lineWidth = 1;
            annotationCtx.stroke();
        }
    }

    // Draw hover ring on hovered vertex
    if (state.hoveredVertexTool && state.hoveredVertexIndex !== null) {
        const polygon = state.currentAnnotation?.[state.hoveredVertexTool];
        if (polygon && state.hoveredVertexIndex < polygon.length) {
            const p = polygon[state.hoveredVertexIndex];
            annotationCtx.beginPath();
            annotationCtx.arc(p[0], p[1], 9, 0, Math.PI * 2);
            annotationCtx.strokeStyle = '#fff';
            annotationCtx.lineWidth = 2;
            annotationCtx.stroke();
        }
    }

    // Draw selected vertex highlight (hollow yellow ring for visibility)
    if (state.selectedVertexTool && state.selectedVertexIndex !== null) {
        const polygon = state.currentAnnotation?.[state.selectedVertexTool];
        if (polygon && state.selectedVertexIndex < polygon.length) {
            const p = polygon[state.selectedVertexIndex];
            // Outer yellow ring (no fill - hollow)
            annotationCtx.beginPath();
            annotationCtx.arc(p[0], p[1], 10, 0, Math.PI * 2);
            annotationCtx.strokeStyle = COLORS.highlight;
            annotationCtx.lineWidth = 3;
            annotationCtx.stroke();
            // Inner white ring for visibility on any background
            annotationCtx.beginPath();
            annotationCtx.arc(p[0], p[1], 7, 0, Math.PI * 2);
            annotationCtx.strokeStyle = '#fff';
            annotationCtx.lineWidth = 1;
            annotationCtx.stroke();
        }
    }

    // Draw multi-selected vertices (yellow highlight rings)
    if (state.selectedVertices.length > 0) {
        for (const sv of state.selectedVertices) {
            const polygon = state.currentAnnotation?.[sv.maskKey];
            if (polygon && sv.vertexIndex < polygon.length) {
                const p = polygon[sv.vertexIndex];
                annotationCtx.beginPath();
                annotationCtx.arc(p[0], p[1], 10, 0, Math.PI * 2);
                annotationCtx.strokeStyle = COLORS.highlight;
                annotationCtx.lineWidth = 3;
                annotationCtx.stroke();
                annotationCtx.beginPath();
                annotationCtx.arc(p[0], p[1], 7, 0, Math.PI * 2);
                annotationCtx.strokeStyle = '#fff';
                annotationCtx.lineWidth = 1;
                annotationCtx.stroke();
            }
        }
    }

    // Draw box selection rectangle
    if (state.isBoxSelecting && state.boxSelectStart && state.boxSelectEnd) {
        const [x1, y1] = state.boxSelectStart;
        const [x2, y2] = state.boxSelectEnd;
        annotationCtx.beginPath();
        annotationCtx.rect(
            Math.min(x1, x2), Math.min(y1, y2),
            Math.abs(x2 - x1), Math.abs(y2 - y1)
        );
        annotationCtx.strokeStyle = COLORS.highlight;
        annotationCtx.lineWidth = 1.5;
        annotationCtx.setLineDash([6, 4]);
        annotationCtx.stroke();
        annotationCtx.setLineDash([]);
        annotationCtx.fillStyle = 'rgba(255, 204, 0, 0.08)';
        annotationCtx.fill();
    }
}

// handleCanvasMouseMove is replaced by handleCanvasMouseMoveNew above

// ============================================================================
// Navigation
// ============================================================================

function getActiveFrames() {
    // If "all" rate selected, return all frames
    if (state.frameRate === 'all') {
        return state.allFrames.length > 0 ? state.allFrames : state.sampledFrames;
    }
    const rate = parseInt(state.frameRate, 10) || 100;
    // Filter by selected rate from whichever frame list is available
    if (state.allFrames.length > 0) {
        return state.allFrames.filter(f => f % rate === 0);
    }
    // Filter sampledFrames by rate too (backend gives every 25)
    return state.sampledFrames.filter(f => f % rate === 0);
}

function getNavigableFrames() {
    // Active frames plus any COMPLETED off-rate frames, sorted.
    // Broken frames are excluded entirely — they must never be navigable.
    const activeFrames = getActiveFrames().filter(
        f => state.frameStatus[String(f)] !== 'broken'
    );
    const activeSet = new Set(activeFrames);
    const extra = [];
    Object.entries(state.frameStatus).forEach(([frameStr, status]) => {
        const frameNum = parseInt(frameStr, 10);
        if (!activeSet.has(frameNum) && status === 'completed') {
            extra.push(frameNum);
        }
    });
    if (extra.length === 0) return activeFrames;
    return [...activeFrames, ...extra].sort((a, b) => a - b);
}

/**
 * Lightweight first-frame display during progressive trial loading.
 * Uses single-frame annotation endpoint (no full trial load).
 * Skips auto-save and SAM auto-apply since this is the initial frame.
 * @param {number} frameIdx - Frame index to display
 */
async function navigateToFrameQuick(frameIdx) {
    if (!state.trialId || frameIdx === null) return;

    state.selectedSegmentIndices.clear();
    state.frameIdx = frameIdx;

    try {
        // Load image
        const img = await loadFrameImage(state.trialId, frameIdx);
        state.imageWidth = img.width;
        state.imageHeight = img.height;

        // Resize canvases (image + border)
        const bw = EDGE_BORDER;
        elements.imageCanvas.width = img.width + 2 * bw;
        elements.imageCanvas.height = img.height + 2 * bw;
        elements.annotationCanvas.width = img.width + 2 * bw;
        elements.annotationCanvas.height = img.height + 2 * bw;

        // Draw image offset by border width
        imageCtx.drawImage(img, bw, bw);

        // Persistent translate so all annotation drawing uses image coords
        annotationCtx.setTransform(1, 0, 0, 1, bw, bw);

        // Load single-frame annotation (no full trial load, no prior)
        const annData = await loadFrameAnnotationSingle(state.trialId, frameIdx);
        state.currentAnnotation = annData.annotation;
        state.priorAnnotation = null;  // Not available without full cache
        state.kinematics = annData.kinematics;
        state.annotationDirty = false;

        // Update UI
        updateNavigationUI();
        zoomFit();
        drawAllAnnotations();
        updateStatus();
        updateToolButtonStatus();
        updateVisibilitySelects();
        updateKinematicsDisplay();
        updateSkipButtons();
        updateJsonViewer();

        elements.placeholder.style.display = 'none';
    } catch (error) {
        console.error('Error in quick navigate:', error);
        showToast(`Error loading frame ${frameIdx}`, true);
    }
}

async function navigateToFrame(frameIdx) {
    if (!state.trialId || frameIdx === null) return;

    // Auto-save current frame before navigating away (only if modified)
    if (state.annotationDirty && state.currentAnnotation && state.frameIdx !== null && state.frameIdx !== frameIdx) {
        await saveAnnotations();
    }

    // Clear SAM multi-select when navigating frames
    state.selectedSegmentIndices.clear();

    state.frameIdx = frameIdx;

    try {
        // Load image
        const img = await loadFrameImage(state.trialId, frameIdx);
        state.imageWidth = img.width;
        state.imageHeight = img.height;

        // Resize canvases (image + border)
        const bw = EDGE_BORDER;
        elements.imageCanvas.width = img.width + 2 * bw;
        elements.imageCanvas.height = img.height + 2 * bw;
        elements.annotationCanvas.width = img.width + 2 * bw;
        elements.annotationCanvas.height = img.height + 2 * bw;

        // Draw image offset by border width
        imageCtx.drawImage(img, bw, bw);

        // Persistent translate so all annotation drawing uses image coords
        annotationCtx.setTransform(1, 0, 0, 1, bw, bw);

        // Load annotations
        const annData = await loadFrameAnnotations(state.trialId, frameIdx);
        state.currentAnnotation = annData.annotation;
        state.priorAnnotation = annData.prior;
        state.kinematics = annData.kinematics;
        state.annotationDirty = false;

        // Update UI
        updateNavigationUI();
        zoomFit();  // Auto-fit image to canvas
        drawAllAnnotations();
        updateStatus();
        updateToolButtonStatus();
        updateVisibilitySelects();
        updateKinematicsDisplay();
        updateSkipButtons();
        updateJsonViewer();
        updatePhaseControls();
        drawPhaseStrip();
        updatePegboardCounters();
        if (state.selectedPegId) loadPegDataToUI(state.selectedPegId);

        elements.placeholder.style.display = 'none';

        // Manual mask mode: auto-select tool1_mask on frame navigate
        if (state.manualMaskMode && state.autoAdvance) {
            setTimeout(() => {
                selectTool('tool1_mask');
            }, 100);
        }

        // Auto-apply SAM mode if no masks exist and SAM data is available (skip in manual mask mode)
        if (state.autoAdvance && !state.manualMaskMode) {
            const ann = state.currentAnnotation;
            const hasTool1Mask = ann?.tool1_mask?.length > 0;
            const hasTool2Mask = ann?.tool2_mask?.length > 0;
            const samAvailable = state.samFrameAvailability[String(frameIdx)];

            if (!hasTool1Mask && !hasTool2Mask && samAvailable) {
                // No masks - auto-enter SAM apply mode
                setTimeout(() => {
                    enterApplyExistingMode();
                    showToast('No masks - SAM apply mode activated');
                }, 200);
            }
        }
    } catch (error) {
        console.error('Error navigating to frame:', error);
        showToast(`Error loading frame ${frameIdx}`, true);
    }
}

async function navigateNext() {
    const frames = getNavigableFrames();
    if (frames.length === 0) return;

    const currentIdx = frames.indexOf(state.frameIdx);
    if (currentIdx < frames.length - 1) {
        await navigateToFrame(frames[currentIdx + 1]);
    }
}

async function navigatePrev() {
    const frames = getNavigableFrames();
    if (frames.length === 0) return;

    const currentIdx = frames.indexOf(state.frameIdx);
    if (currentIdx > 0) {
        await navigateToFrame(frames[currentIdx - 1]);
    }
}

function updateNavigationUI() {
    const frames = getNavigableFrames();
    const currentIdx = frames.indexOf(state.frameIdx);

    const status = state.frameStatus[String(state.frameIdx)];
    let statusLabel = '';
    if (status === 'completed') statusLabel = ' \u2713';
    else if (status === 'skipped') statusLabel = ' \u2298';
    else if (status === 'partial') statusLabel = ' \u25D0';
    else if (status === 'negative') statusLabel = ' \u2212';
    elements.currentFrameNum.textContent = state.frameIdx + statusLabel;
    elements.frameIdx.textContent = currentIdx >= 0 ? currentIdx + 1 : '?';
    elements.totalFrames.textContent = frames.length;

    elements.prevBtn.disabled = currentIdx <= 0;
    elements.nextBtn.disabled = currentIdx >= frames.length - 1;

    // Update frame jump dropdown selection
    elements.frameJumpSelect.value = state.frameIdx;
}

function updateFrameJumpDropdown() {
    const activeFrames = getActiveFrames();
    const activeSet = new Set(activeFrames);
    const frames = getNavigableFrames();
    elements.frameJumpSelect.innerHTML = '<option value="">Jump to frame...</option>';

    frames.forEach((frameIdx, idx) => {
        const opt = document.createElement('option');
        opt.value = frameIdx;

        const status = state.frameStatus[String(frameIdx)];
        const isExtra = !activeSet.has(frameIdx);

        let indicator = '';
        if (status === 'completed') {
            indicator = ' \u2713';   // ✓
        } else if (status === 'skipped') {
            indicator = ' \u2298';   // ⊘
        } else if (status === 'negative') {
            indicator = ' \u2212';   // −
        } else if (status === 'partial') {
            indicator = ' \u25D0';   // ◐
        } else {
            indicator = ' \u25CB';   // ○ (not annotated)
        }

        const extraLabel = isExtra ? ' *' : '';
        opt.textContent = `Frame ${frameIdx} (${idx + 1}/${frames.length})${extraLabel}${indicator}`;

        if (status === 'completed') {
            opt.style.color = '#2ecc71';  // green
        } else if (status === 'partial') {
            opt.style.color = '#f39c12';  // orange
        } else if (status === 'skipped' || status === 'negative') {
            opt.style.color = '#95a5a6';  // gray
        }

        elements.frameJumpSelect.appendChild(opt);
    });
}

function updateKinematicsDisplay() {
    if (!state.kinematics || !state.showPose) {
        elements.kinematicsInfo.style.display = 'none';
        return;
    }

    const t1 = state.kinematics.tool1;
    const t2 = state.kinematics.tool2;

    let html = '';
    if (t1 && t1.Position) {
        html += `<span class="kin-tool1">T1: [${t1.Position.map(v => v?.toFixed(2) || 'N/A').join(', ')}]</span>`;
    }
    if (t2 && t2.Position) {
        html += `<span class="kin-tool2">T2: [${t2.Position.map(v => v?.toFixed(2) || 'N/A').join(', ')}]</span>`;
    }

    elements.kinematicsText.innerHTML = html;
    elements.kinematicsInfo.style.display = html ? 'block' : 'none';
}

// ============================================================================
// Prior Annotations
// ============================================================================

function usePriorAnnotation() {
    if (!state.priorAnnotation || !state.currentAnnotation) {
        showToast('No prior annotation available', true);
        return;
    }

    pushUndo();

    const defaultVis = {
        mask: 1, lines: 1,
        joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1
    };

    // Copy all annotations from prior
    state.currentAnnotation.tool1_mask = state.priorAnnotation.tool1_mask ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_mask)) : [];
    state.currentAnnotation.tool2_mask = state.priorAnnotation.tool2_mask ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_mask)) : [];
    state.currentAnnotation.tool1_lines = state.priorAnnotation.tool1_lines ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_lines)) :
        { top: [], bottom: [], middle: [] };
    state.currentAnnotation.tool2_lines = state.priorAnnotation.tool2_lines ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_lines)) :
        { top: [], bottom: [], middle: [] };

    // Copy keypoints
    state.currentAnnotation.tool1_joint = state.priorAnnotation.tool1_joint ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_joint)) : [];
    state.currentAnnotation.tool1_ee_tip = state.priorAnnotation.tool1_ee_tip ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_ee_tip)) : [];
    state.currentAnnotation.tool1_ee_left = state.priorAnnotation.tool1_ee_left ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_ee_left)) : [];
    state.currentAnnotation.tool1_ee_right = state.priorAnnotation.tool1_ee_right ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_ee_right)) : [];
    state.currentAnnotation.tool2_joint = state.priorAnnotation.tool2_joint ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_joint)) : [];
    state.currentAnnotation.tool2_ee_tip = state.priorAnnotation.tool2_ee_tip ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_ee_tip)) : [];
    state.currentAnnotation.tool2_ee_left = state.priorAnnotation.tool2_ee_left ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_ee_left)) : [];
    state.currentAnnotation.tool2_ee_right = state.priorAnnotation.tool2_ee_right ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_ee_right)) : [];

    // Copy visibility
    state.currentAnnotation.tool1_visibility = state.priorAnnotation.tool1_visibility ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool1_visibility)) :
        { ...defaultVis };
    state.currentAnnotation.tool2_visibility = state.priorAnnotation.tool2_visibility ?
        JSON.parse(JSON.stringify(state.priorAnnotation.tool2_visibility)) :
        { ...defaultVis };

    saveAnnotations();
    drawAllAnnotations();
    updateVisibilitySelects();
    showToast('Prior annotations applied');
}

// ============================================================================
// Undo
// ============================================================================

function pushUndo() {
    if (!state.currentAnnotation) return;

    state.annotationDirty = true;
    const snapshot = JSON.stringify(state.currentAnnotation);
    state.undoStack.push(snapshot);

    if (state.undoStack.length > state.maxUndo) {
        state.undoStack.shift();
    }
}

function undo() {
    if (state.undoStack.length === 0) {
        showToast('Nothing to undo', true);
        return;
    }

    const snapshot = state.undoStack.pop();
    state.currentAnnotation = JSON.parse(snapshot);
    state.annotationDirty = true;
    saveAnnotations();
    drawAllAnnotations();
    updateJsonViewer();
    showToast('Undone');
}

// ============================================================================
// Clear Tool
// ============================================================================

function clearCurrentTool() {
    if (!state.currentTool || !state.currentAnnotation) return;

    pushUndo();
    const tool = state.currentTool;

    if (tool.includes('mask')) {
        const key = tool.startsWith('tool1') ? 'tool1_mask' : 'tool2_mask';
        state.currentAnnotation[key] = [];
    } else if (tool.includes('top') || tool.includes('bottom') || tool.includes('middle')) {
        const lineType = tool.includes('top') ? 'top' : tool.includes('middle') ? 'middle' : 'bottom';
        const linesKey = tool.startsWith('tool1') ? 'tool1_lines' : 'tool2_lines';
        state.currentAnnotation[linesKey][lineType] = [];
    } else if (tool.includes('joint') || tool.includes('ee_')) {
        // Clear keypoint
        state.currentAnnotation[tool] = [];
    }

    cancelDrawing();
    saveAnnotations();
    drawAllAnnotations();
    updateStatus();
    updateToolButtonStatus();
    showToast('Cleared');
}

function clearAllAnnotations() {
    if (!state.currentAnnotation) return;

    pushUndo();

    const defaultVis = {
        mask: 1, lines: 1,
        joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1
    };

    // Clear all annotations
    state.currentAnnotation.tool1_mask = [];
    state.currentAnnotation.tool2_mask = [];
    state.currentAnnotation.tool1_lines = { top: [], bottom: [], middle: [] };
    state.currentAnnotation.tool2_lines = { top: [], bottom: [], middle: [] };
    state.currentAnnotation.tool1_joint = [];
    state.currentAnnotation.tool1_ee_tip = [];
    state.currentAnnotation.tool1_ee_left = [];
    state.currentAnnotation.tool1_ee_right = [];
    state.currentAnnotation.tool2_joint = [];
    state.currentAnnotation.tool2_ee_tip = [];
    state.currentAnnotation.tool2_ee_left = [];
    state.currentAnnotation.tool2_ee_right = [];
    state.currentAnnotation.tool1_visibility = { ...defaultVis };
    state.currentAnnotation.tool2_visibility = { ...defaultVis };

    cancelDrawing();
    saveAnnotations();
    drawAllAnnotations();
    updateStatus();
    updateToolButtonStatus();
    updateVisibilitySelects();
    showToast('All annotations cleared');
}

// ============================================================================
// Zoom
// ============================================================================

function applyTransform() {
    const wrapper = document.getElementById('canvas-wrapper');
    if (wrapper) {
        wrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    }
    elements.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function updateZoom() {
    applyTransform();
    drawAllAnnotations();
}

function zoomIn() {
    const container = elements.canvasContainer;
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    const imgX = (centerX - state.panX) / state.zoom;
    const imgY = (centerY - state.panY) / state.zoom;

    state.zoom = Math.min(state.zoom * 1.2, 5.0);

    state.panX = centerX - imgX * state.zoom;
    state.panY = centerY - imgY * state.zoom;
    updateZoom();
}

function zoomOut() {
    const container = elements.canvasContainer;
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    const imgX = (centerX - state.panX) / state.zoom;
    const imgY = (centerY - state.panY) / state.zoom;

    state.zoom = Math.max(state.zoom / 1.2, 0.1);

    state.panX = centerX - imgX * state.zoom;
    state.panY = centerY - imgY * state.zoom;
    updateZoom();
}

function zoomFit() {
    const container = elements.canvasContainer;
    const containerW = container.clientWidth;
    const containerH = container.clientHeight;
    const canvasW = state.imageWidth + 2 * EDGE_BORDER;
    const canvasH = state.imageHeight + 2 * EDGE_BORDER;

    if (canvasW > 0 && canvasH > 0) {
        state.zoom = Math.min(containerW / canvasW, containerH / canvasH, 1.0);
        state.zoom = Math.max(state.zoom, 0.1);
    }

    state.panX = (containerW - canvasW * state.zoom) / 2;
    state.panY = (containerH - canvasH * state.zoom) / 2;
    updateZoom();
}

// ============================================================================
// Status & Progress
// ============================================================================

function updateStatus() {
    if (!state.currentAnnotation) return;

    const ann = state.currentAnnotation;
    migrateVisibility(ann);
    const t1v = ann.tool1_visibility || {};
    const t2v = ann.tool2_visibility || {};

    // Helper to get status string based on visibility and data
    const getStatus = (vis, hasData) => {
        if (vis === -1) return 'out';
        if (vis === 0) return 'occluded';
        return hasData ? 'complete' : 'incomplete';
    };

    // Tool 1 mask
    updateStatusItem('tool1_mask', getStatus(t1v.mask, ann.tool1_mask && ann.tool1_mask.length >= 3));

    // Tool 1 lines (only top/bottom - middle is auto-computed)
    const t1LinesComplete = ['top', 'bottom'].every(
        lt => ann.tool1_lines?.[lt] && ann.tool1_lines[lt].length === 2
    );
    updateStatusItem('tool1_lines', getStatus(t1v.lines, t1LinesComplete));

    // Tool 1 keypoints
    updateStatusItem('tool1_joint', getStatus(t1v.joint, ann.tool1_joint && ann.tool1_joint.length === 2));
    updateStatusItem('tool1_ee_tip', getStatus(t1v.ee_tip, ann.tool1_ee_tip && ann.tool1_ee_tip.length === 2));
    updateStatusItem('tool1_ee_left', getStatus(t1v.ee_left, ann.tool1_ee_left && ann.tool1_ee_left.length === 2));
    updateStatusItem('tool1_ee_right', getStatus(t1v.ee_right, ann.tool1_ee_right && ann.tool1_ee_right.length === 2));

    // Tool 2 mask
    updateStatusItem('tool2_mask', getStatus(t2v.mask, ann.tool2_mask && ann.tool2_mask.length >= 3));

    // Tool 2 lines (only top/bottom - middle is auto-computed)
    const t2LinesComplete = ['top', 'bottom'].every(
        lt => ann.tool2_lines?.[lt] && ann.tool2_lines[lt].length === 2
    );
    updateStatusItem('tool2_lines', getStatus(t2v.lines, t2LinesComplete));

    // Tool 2 keypoints
    updateStatusItem('tool2_joint', getStatus(t2v.joint, ann.tool2_joint && ann.tool2_joint.length === 2));
    updateStatusItem('tool2_ee_tip', getStatus(t2v.ee_tip, ann.tool2_ee_tip && ann.tool2_ee_tip.length === 2));
    updateStatusItem('tool2_ee_left', getStatus(t2v.ee_left, ann.tool2_ee_left && ann.tool2_ee_left.length === 2));
    updateStatusItem('tool2_ee_right', getStatus(t2v.ee_right, ann.tool2_ee_right && ann.tool2_ee_right.length === 2));

    // Also update peg status if in pegs mode
    if (state.annotationMode === 'pegs') updatePegStatus();

    // Sort: incomplete items first, complete items last
    const statusList = document.querySelector('#status-panel .status-list');
    if (statusList) {
        const items = Array.from(statusList.children);
        items.sort((a, b) => {
            const aDone = (a.classList.contains('complete') || a.classList.contains('occluded') || a.classList.contains('out')) ? 1 : 0;
            const bDone = (b.classList.contains('complete') || b.classList.contains('occluded') || b.classList.contains('out')) ? 1 : 0;
            return aDone - bDone;
        });
        items.forEach(item => statusList.appendChild(item));
    }
}

/**
 * Update peg annotation status checklist (shown in pegs mode).
 * Uses same color scheme: green=complete, orange=occluded, grey=incomplete.
 */
function updatePegStatus() {
    if (!state.currentAnnotation) return;
    const ann = state.currentAnnotation;
    const pegs = ann.pegs || [];
    const pb = ann.pegboard || {};

    // P1-P6: mask and keypoints
    for (let i = 1; i <= 6; i++) {
        const peg = pegs.find(p => p.id === i);
        const hasMask = peg?.mask && peg.mask.length >= 3;
        const allKps = peg?.keypoints && peg.keypoints.filter(kp => kp !== null).length === 3;
        updateStatusItem(`p${i}_mask`, hasMask ? 'complete' : 'incomplete');
        updateStatusItem(`p${i}_kps`, allKps ? 'complete' : 'incomplete');
    }

    // S1-S6, T1-T6: post mask + keypoint
    for (let i = 0; i < 6; i++) {
        const sMask = pb.source_post_masks?.[i]?.length >= 3;
        const sKp = pb.source_post_keypoints?.[i] !== null && pb.source_post_keypoints?.[i] !== undefined;
        updateStatusItem(`s${i + 1}`, (sMask && sKp) ? 'complete' : sMask ? 'occluded' : 'incomplete');

        const tMask = pb.target_post_masks?.[i]?.length >= 3;
        const tKp = pb.target_post_keypoints?.[i] !== null && pb.target_post_keypoints?.[i] !== undefined;
        updateStatusItem(`t${i + 1}`, (tMask && tKp) ? 'complete' : tMask ? 'occluded' : 'incomplete');
    }

    // Board outline
    updateStatusItem('board', pb.board_mask?.length >= 3 ? 'complete' : 'incomplete');
}

function updateStatusItem(checkName, stateOrBool) {
    const item = document.querySelector(`[data-check="${checkName}"]`);
    if (!item) return;

    const icon = item.querySelector('.status-icon');
    item.classList.remove('complete', 'occluded', 'out');

    // Accept boolean for backward compat or string state
    const st = (typeof stateOrBool === 'boolean')
        ? (stateOrBool ? 'complete' : 'incomplete')
        : stateOrBool;

    if (st === 'complete') {
        item.classList.add('complete');
        icon.innerHTML = '&#9745;';
    } else if (st === 'occluded') {
        item.classList.add('occluded');
        icon.innerHTML = '&#9745;';
    } else if (st === 'out') {
        item.classList.add('out');
        icon.innerHTML = '&#10006;';  // X mark for out
    } else {
        icon.innerHTML = '&#9744;';
    }
}

function updateToolButtonStatus() {
    if (!state.currentAnnotation) return;

    const ann = state.currentAnnotation;
    migrateVisibility(ann);
    const t1v = ann.tool1_visibility || {};
    const t2v = ann.tool2_visibility || {};

    // Helper: check if component is done (not visible or has data)
    const isDone = (vis, hasData) => vis !== 1 || hasData;

    // Update each tool button's status indicator
    const statusMap = {
        'tool1_mask': isDone(t1v.mask, ann.tool1_mask && ann.tool1_mask.length >= 3),
        'tool1_top': isDone(t1v.lines, ann.tool1_lines?.top && ann.tool1_lines.top.length === 2),
        'tool1_bottom': isDone(t1v.lines, ann.tool1_lines?.bottom && ann.tool1_lines.bottom.length === 2),
        'tool1_middle': ann.tool1_lines?.middle && ann.tool1_lines.middle.length === 2,
        'tool1_joint': isDone(t1v.joint, ann.tool1_joint && ann.tool1_joint.length === 2),
        'tool1_ee_tip': isDone(t1v.ee_tip, ann.tool1_ee_tip && ann.tool1_ee_tip.length === 2),
        'tool1_ee_left': isDone(t1v.ee_left, ann.tool1_ee_left && ann.tool1_ee_left.length === 2),
        'tool1_ee_right': isDone(t1v.ee_right, ann.tool1_ee_right && ann.tool1_ee_right.length === 2),
        'tool2_mask': isDone(t2v.mask, ann.tool2_mask && ann.tool2_mask.length >= 3),
        'tool2_top': isDone(t2v.lines, ann.tool2_lines?.top && ann.tool2_lines.top.length === 2),
        'tool2_bottom': isDone(t2v.lines, ann.tool2_lines?.bottom && ann.tool2_lines.bottom.length === 2),
        'tool2_middle': ann.tool2_lines?.middle && ann.tool2_lines.middle.length === 2,
        'tool2_joint': isDone(t2v.joint, ann.tool2_joint && ann.tool2_joint.length === 2),
        'tool2_ee_tip': isDone(t2v.ee_tip, ann.tool2_ee_tip && ann.tool2_ee_tip.length === 2),
        'tool2_ee_left': isDone(t2v.ee_left, ann.tool2_ee_left && ann.tool2_ee_left.length === 2),
        'tool2_ee_right': isDone(t2v.ee_right, ann.tool2_ee_right && ann.tool2_ee_right.length === 2)
    };

    Object.entries(statusMap).forEach(([toolId, done]) => {
        const statusEl = document.getElementById(`status-${toolId}`);
        if (statusEl) {
            statusEl.className = done ? 'tool-status done' : 'tool-status';
        }
    });
}

/**
 * Migrate old visibility formats to full format with keypoints.
 * Visibility: 1=visible (requires annotation), 0=occluded (optional), -1=out of scene.
 */
function migrateVisibility(ann) {
    const defaultVis = {
        mask: 1, lines: 1,
        joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1
    };

    for (const toolNum of [1, 2]) {
        const visKey = `tool${toolNum}_visibility`;
        const missingKey = `tool${toolNum}_missing`;

        if (!ann[visKey]) {
            if (ann[missingKey]) {
                const m = ann[missingKey];
                ann[visKey] = {
                    mask: m.mask ? 0 : 1,
                    lines: m.lines ? 0 : 1,
                    joint: 1, ee_tip: 1, ee_left: 1, ee_right: 1
                };
            } else {
                ann[visKey] = { ...defaultVis };
            }
        } else {
            // Ensure all keys exist with default 1
            for (const key of Object.keys(defaultVis)) {
                if (ann[visKey][key] === undefined) {
                    ann[visKey][key] = 1;
                }
            }
            // Remove old 'ee' key if present
            delete ann[visKey].ee;
        }
    }
}

function updateVisibilitySelects() {
    if (!state.currentAnnotation) return;

    const ann = state.currentAnnotation;
    migrateVisibility(ann);

    // Sync dropdowns to annotation state
    const parts = ['mask', 'lines', 'joint', 'ee_tip', 'ee_left', 'ee_right'];
    for (const toolNum of [1, 2]) {
        const vis = ann[`tool${toolNum}_visibility`];
        for (const part of parts) {
            const el = document.getElementById(`tool${toolNum}-vis-${part}`);
            if (el) {
                el.value = String(vis[part] !== undefined ? vis[part] : 1);
                updateVisSelectStyling(el);
            }
        }
    }
}

function updateVisSelectStyling(el) {
    el.classList.remove('vis-occluded', 'vis-out');
    const val = parseInt(el.value);
    if (val === 0) el.classList.add('vis-occluded');
    else if (val === -1) el.classList.add('vis-out');
}

function handleVisibilityChange(toolNum, part, value) {
    if (!state.currentAnnotation) return;

    pushUndo();

    const key = `tool${toolNum}_visibility`;
    migrateVisibility(state.currentAnnotation);
    state.currentAnnotation[key][part] = parseInt(value);

    // Update dropdown styling
    const el = document.getElementById(`tool${toolNum}-vis-${part}`);
    if (el) updateVisSelectStyling(el);

    saveAnnotations();
    updateStatus();
    updateToolButtonStatus();
}

function updateProgress(progress) {
    const negative = progress.negative || 0;
    const excluded = progress.excluded || 0;
    const total = (progress.total || 1) - negative;
    const completed = progress.completed + progress.skipped;
    const percent = total > 0 ? (completed / total) * 100 : 0;

    // Store excluded count in state for recalculateProgress
    state.excludedCount = excluded;

    elements.progressFill.style.width = `${percent}%`;
    const negText = negative > 0 ? `, ${negative} neg` : '';
    elements.progressText.textContent = `${completed} / ${total}${negText} (${percent.toFixed(1)}%)`;
    elements.completedCount.textContent = `${progress.completed} completed`;
    elements.skippedCount.textContent = `${progress.skipped} skipped`;
    elements.excludedCount.textContent = `${excluded} excluded`;
    elements.negativeCount.textContent = `${negative} out`;
}

// ============================================================================
// Toast Notification
// ============================================================================

function showToast(message, isError = false) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast${isError ? ' error' : ''}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.remove(), 2000);
}

// ============================================================================
// JSON Viewer
// ============================================================================

/**
 * Update the live JSON viewer panel with current annotation data.
 */
function updateJsonViewer() {
    if (!elements.jsonViewerContent) return;

    // Show panel if hidden and we have a trial loaded
    if (elements.jsonViewerPanel && state.trialId) {
        elements.jsonViewerPanel.style.display = '';
    }

    if (!state.currentAnnotation) {
        elements.jsonViewerContent.textContent = '// No annotation loaded';
        return;
    }

    // Format JSON with 2-space indent
    const json = JSON.stringify(state.currentAnnotation, null, 2);
    elements.jsonViewerContent.textContent = json;
}

// ============================================================================
// Toggle Functions
// ============================================================================

function togglePhaseStrip() {
    state.showPhaseStrip = !state.showPhaseStrip;
    elements.phaseStripToggle.checked = state.showPhaseStrip;
    const phaseStrip = document.getElementById('phase-strip');
    const phaseControls = document.getElementById('phase-controls');
    if (phaseStrip) phaseStrip.style.display = state.showPhaseStrip ? 'block' : 'none';
    if (phaseControls) phaseControls.style.display = state.showPhaseStrip ? 'flex' : 'none';
}


async function updateFrameRate() {
    const value = elements.frameRateSelect.value;
    state.frameRate = value === 'all' ? 'all' : parseInt(value, 10);

    // Load all frames if not already loaded (needed for any rate filtering)
    if (state.allFrames.length === 0 && state.trialId) {
        await loadAllFrames(state.trialId);
    }

    updateFrameJumpDropdown();
    recalculateProgress();
    updateNavigationUI();
}

// ============================================================================
// Event Listeners
// ============================================================================

// Trial navigation helpers
function updateTrialNavButtons() {
    const select = elements.trialSelect;
    elements.prevTrialBtn.disabled = (select.selectedIndex <= 1);
    elements.nextTrialBtn.disabled = (select.selectedIndex >= select.options.length - 1 || select.selectedIndex < 1);
}

async function switchTrial(direction) {
    const select = elements.trialSelect;
    const newIndex = select.selectedIndex + direction;
    if (newIndex < 1 || newIndex >= select.options.length) return;
    select.selectedIndex = newIndex;
    select.dispatchEvent(new Event('change'));
}

// Dataset selection
elements.datasetSelect.addEventListener('change', async (e) => {
    const dataset = e.target.value;
    state.dataset = dataset;

    elements.trialSelect.innerHTML = '<option value="">Select Trial...</option>';

    if (!dataset) {
        elements.trialSelect.disabled = true;
        return;
    }

    const option = e.target.selectedOptions[0];
    const trials = JSON.parse(option.dataset.trials);

    trials.forEach(trial => {
        const opt = document.createElement('option');
        opt.value = trial.trial_id;
        applyTrialProgress(opt, trial.trial_name, trial);
        elements.trialSelect.appendChild(opt);
    });

    elements.trialSelect.disabled = false;
    updateTrialNavButtons();

    // Background: recompute accurate progress from disk (non-blocking)
    refreshDatasetProgress(dataset);
});

// Trial selection — 3-phase progressive loading
elements.trialSelect.addEventListener('change', async (e) => {
    const trialId = e.target.value;
    if (!trialId) return;

    // Cancel any active SSE from previous trial
    if (state._activeSSE) {
        state._activeSSE.close();
        state._activeSSE = null;
    }
    if (state._fullLoadSSE) {
        state._fullLoadSSE.close();
        state._fullLoadSSE = null;
    }

    state.trialId = trialId;
    state.allFrames = [];
    state.frameRate = 100;  // Reset to default rate
    elements.frameRateSelect.value = '100';

    showLoading('Loading trial...');
    elements.trialSelect.disabled = true;

    // Show panels with placeholder progress BEFORE Phase 1
    elements.progressPanel.style.display = 'block';
    elements.navigationPanel.style.display = 'block';
    elements.optionsPanel.style.display = 'block';
    elements.toolsPanel.style.display = 'block';
    elements.statusPanel.style.display = 'block';
    elements.samPanel.style.display = 'block';
    // Show mode switcher, phase strip, phase controls
    const modeSwitcher = document.getElementById('mode-switcher');
    const phaseStrip = document.getElementById('phase-strip');
    const phaseControls = document.getElementById('phase-controls');
    if (modeSwitcher) modeSwitcher.style.display = 'flex';
    if (phaseStrip) phaseStrip.style.display = state.showPhaseStrip ? 'block' : 'none';
    if (phaseControls) phaseControls.style.display = state.showPhaseStrip ? 'flex' : 'none';
    // Reset to tools mode
    state.annotationMode = 'tools';
    updateAnnotationModeUI();
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = 'Loading...';
    elements.completedCount.textContent = '- completed';
    elements.skippedCount.textContent = '- skipped';
    elements.excludedCount.textContent = '- excluded';
    elements.negativeCount.textContent = '- out';

    try {
        // ── Phase 1: Quick start ──
        // Load frame list without loading annotations (fast)
        // Also sets approximate progress from _progress.json
        const liteData = await loadTrialFramesLite(trialId);
        const annotationFileCount = liteData.annotation_file_count || 0;

        // Fetch per-frame status early so the dropdown shows indicators
        // before Phase 2/3 complete (fast endpoint, ~29 frames)
        await fetchFrameStatus(trialId);

        // Navigate to first frame using single-frame read (no full trial load)
        if (state.sampledFrames.length > 0) {
            await navigateToFrameQuick(state.sampledFrames[0]);
        }

        updateTrialNavButtons();
        elements.trialSelect.disabled = false;  // Re-enable immediately after Phase 1

        // ── Phase 2: Background load with progress counter ──
        if (annotationFileCount > 0) {
            showLoadingProgress('Loading annotations', 0, annotationFileCount);
            await streamAnnotationLoad(trialId, annotationFileCount);
        }

        // ── Phase 3: Finalize (cached now, instant) ──
        // Guard: bail if user switched trials during Phase 2
        if (state.trialId !== trialId) return;

        // Full endpoints are now instant (annotations cached by SSE stream)
        await loadTrialFrames(trialId);
        await loadAllFrames(trialId);

        // Re-fetch current frame with full annotations (now includes prior)
        if (state.frameIdx !== null) {
            const annData = await loadFrameAnnotations(state.trialId, state.frameIdx);
            state.currentAnnotation = annData.annotation;
            state.priorAnnotation = annData.prior;
            state.kinematics = annData.kinematics;
            state.annotationDirty = false;
            drawAllAnnotations();
            updateStatus();
            updateToolButtonStatus();
            updateVisibilitySelects();
            updateKinematicsDisplay();
            updateSkipButtons();
            updateJsonViewer();
        }

        updateTrialNavButtons();

        // ── Phase 4: Background full annotation load ──
        // Load ALL annotation files (not just sampled) so off-grid frames are safe
        if (state.trialId === trialId) {
            loadFullTrialAnnotations(trialId);
            // Load phase labels for timeline strip
            loadPhaseSummary(trialId);
        }
    } finally {
        hideLoading();
        elements.trialSelect.disabled = false;
    }
});

// Trial nav buttons
elements.prevTrialBtn.addEventListener('click', () => switchTrial(-1));
elements.nextTrialBtn.addEventListener('click', () => switchTrial(+1));

// Frame jump
elements.frameJumpSelect.addEventListener('change', async (e) => {
    const frameIdx = parseInt(e.target.value);
    if (!isNaN(frameIdx)) {
        await navigateToFrame(frameIdx);
    }
});

// Frame rate selector
elements.frameRateSelect.addEventListener('change', updateFrameRate);

// Phase strip toggle
elements.phaseStripToggle.addEventListener('change', (e) => {
    state.showPhaseStrip = e.target.checked;
    const phaseStrip = document.getElementById('phase-strip');
    const phaseControls = document.getElementById('phase-controls');
    if (phaseStrip) phaseStrip.style.display = state.showPhaseStrip ? 'block' : 'none';
    if (phaseControls) phaseControls.style.display = state.showPhaseStrip ? 'flex' : 'none';
});

// Edge selection toggle (snap lines to mask edges)
const edgeToggle = document.getElementById('edge-select-toggle');
if (edgeToggle) {
    state.useEdgeSelection = localStorage.getItem('useEdgeSelection') !== 'false';
    edgeToggle.checked = state.useEdgeSelection;
    edgeToggle.addEventListener('change', () => {
        state.useEdgeSelection = edgeToggle.checked;
        localStorage.setItem('useEdgeSelection', state.useEdgeSelection);
    });
}

// Auto-advance toggle
const autoAdvanceCheckbox = document.getElementById('auto-advance-checkbox');
if (autoAdvanceCheckbox) {
    state.autoAdvance = localStorage.getItem('autoAdvance') !== 'false';
    autoAdvanceCheckbox.checked = state.autoAdvance;
    autoAdvanceCheckbox.addEventListener('change', () => {
        state.autoAdvance = autoAdvanceCheckbox.checked;
        localStorage.setItem('autoAdvance', state.autoAdvance);
        showToast(state.autoAdvance ? 'Auto-advance ON' : 'Auto-advance OFF');
    });
}

// Manual mask mode toggle
if (elements.manualMaskCheckbox) {
    state.manualMaskMode = localStorage.getItem('manualMaskMode') === 'true';
    elements.manualMaskCheckbox.checked = state.manualMaskMode;
    elements.manualMaskCheckbox.addEventListener('change', () => {
        state.manualMaskMode = elements.manualMaskCheckbox.checked;
        localStorage.setItem('manualMaskMode', state.manualMaskMode);
        showToast(state.manualMaskMode ? 'Manual masks ON (SAM auto-apply disabled)' : 'Manual masks OFF');
    });
}

// Navigation buttons
elements.prevBtn.addEventListener('click', navigatePrev);
elements.nextBtn.addEventListener('click', navigateNext);
elements.skipBtn.addEventListener('click', skipFrame);
elements.unskipBtn.addEventListener('click', unskipFrame);
elements.reloadBtn.addEventListener('click', reloadFrame);

// Tool buttons (only those with data-tool attribute)
document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
    btn.addEventListener('click', () => selectTool(btn.dataset.tool));
});

// Edit mode button
document.getElementById('edit-mode-btn').addEventListener('click', toggleEditMode);

// Clickable status items
document.querySelectorAll('.status-item.clickable').forEach(item => {
    item.addEventListener('click', () => {
        const tool = item.dataset.tool;
        if (tool) selectTool(tool);
    });
});

// Tool actions
elements.clearToolBtn.addEventListener('click', clearCurrentTool);
elements.usePriorBtn.addEventListener('click', () => {
    if (state.annotationMode === 'pegs') {
        showToast('Use peg-specific Copy Prior in Pegs panel', true);
        return;
    }
    usePriorAnnotation();
});

// Visibility select dropdowns
document.querySelectorAll('.vis-select').forEach(el => {
    el.addEventListener('change', (e) => {
        const toolNum = parseInt(e.target.dataset.tool);
        const part = e.target.dataset.part;
        handleVisibilityChange(toolNum, part, e.target.value);
    });
});

// Double-click to cycle visibility states: Visible → Occluded → Out → Visible
document.querySelectorAll('.vis-select').forEach(el => {
    el.addEventListener('dblclick', (e) => {
        const toolNum = parseInt(e.target.dataset.tool);
        const part = e.target.dataset.part;
        const current = parseInt(e.target.value);
        // Cycle: 1 → 0 → -1 → 1
        const next = current === 1 ? 0 : current === 0 ? -1 : 1;
        e.target.value = String(next);
        handleVisibilityChange(toolNum, part, next);
    });
});

// Clear all button with confirmation
document.getElementById('clear-all-btn').addEventListener('click', () => {
    if (confirm('Are you sure you want to clear ALL annotations for this frame?')) {
        clearAllAnnotations();
    }
});

// Zoom buttons
elements.zoomIn.addEventListener('click', zoomIn);
elements.zoomOut.addEventListener('click', zoomOut);
elements.zoomFit.addEventListener('click', zoomFit);

// SAM panel buttons
if (elements.samApplyBtn) {
    elements.samApplyBtn.addEventListener('click', enterApplyExistingMode);
}
if (elements.samNewBtn) {
    elements.samNewBtn.addEventListener('click', toggleSamMode);
}
if (elements.samAssignTool1) {
    elements.samAssignTool1.addEventListener('click', () => assignSegmentToTool(1));
}
if (elements.samAssignTool2) {
    elements.samAssignTool2.addEventListener('click', () => assignSegmentToTool(2));
}
// Clear selection button
const samClearBtn = document.getElementById('sam-clear-selection');
if (samClearBtn) {
    samClearBtn.addEventListener('click', clearSegmentSelection);
}

// Mark Out buttons
const markTool1OutBtn = document.getElementById('mark-tool1-out');
if (markTool1OutBtn) {
    markTool1OutBtn.addEventListener('click', () => markToolAsOut(1));
}
const markTool2OutBtn = document.getElementById('mark-tool2-out');
if (markTool2OutBtn) {
    markTool2OutBtn.addEventListener('click', () => markToolAsOut(2));
}
const markAllOutBtn = document.getElementById('mark-all-out');
if (markAllOutBtn) {
    markAllOutBtn.addEventListener('click', markAllAsOut);
}

// Force Save button
const forceSaveBtn = document.getElementById('forceSaveBtn');
if (forceSaveBtn) {
    forceSaveBtn.addEventListener('click', async () => {
        if (!state.currentAnnotation) {
            showToast('No annotation to save', true);
            return;
        }
        await saveAnnotations();
    });
}

// Backup button
const backupBtn = document.getElementById('backupBtn');
if (backupBtn) {
    backupBtn.addEventListener('click', async () => {
        if (!state.trialId) {
            showToast('No trial loaded', true);
            return;
        }
        try {
            backupBtn.disabled = true;
            backupBtn.textContent = '⏳ Backing up...';
            const data = await api(`/trials/${state.trialId}/backup`, { method: 'POST' });
            if (data.success) {
                showToast(`Backup saved: ${data.path}`);
            } else {
                showToast('Backup failed', true);
            }
        } catch (error) {
            console.error('Backup failed:', error);
            showToast('Backup failed!', true);
        } finally {
            backupBtn.disabled = false;
            backupBtn.textContent = '📦 Backup';
        }
    });
}

// Copy JSON button
if (elements.copyJsonBtn) {
    elements.copyJsonBtn.addEventListener('click', () => {
        if (!state.currentAnnotation) {
            showToast('No annotation to copy', true);
            return;
        }
        const json = JSON.stringify(state.currentAnnotation, null, 2);
        navigator.clipboard.writeText(json).then(() => {
            showToast('Copied to clipboard');
        }).catch(() => {
            showToast('Failed to copy', true);
        });
    });
}

// Canvas events (refactored for pan, edit, edge-select support)
elements.annotationCanvas.addEventListener('mousedown', handleCanvasMouseDown);
elements.annotationCanvas.addEventListener('mousemove', handleCanvasMouseMoveNew);
elements.annotationCanvas.addEventListener('mouseup', handleCanvasMouseUp);

// Wheel zoom (Feature 2)
elements.canvasContainer.addEventListener('wheel', handleWheelZoom, { passive: false });

// Prevent right-click context menu (right-click is used for pan)
elements.annotationCanvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Video mode takes priority
    if (videoState.active) {
        if (videoHandleKeydown(e)) return;
    }

    // Skip if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    // Block modifier+key combos from triggering our shortcuts
    // Exceptions: Ctrl/Cmd+Z for undo, Ctrl/Cmd+S for save
    if ((e.ctrlKey || e.metaKey || e.altKey) && e.key.length === 1) {
        if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
            // Fall through to undo handler
        } else if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
            // Ctrl+S / Cmd+S: Force save
            e.preventDefault();
            if (state.currentAnnotation) {
                saveAnnotations();
            } else {
                showToast('No annotation to save', true);
            }
            return;
        } else {
            return; // Let browser handle Ctrl+C, Cmd+A, etc.
        }
    }

    // Space for pan
    if (e.key === ' ' || e.code === 'Space') {
        if (!state.spaceHeld) {
            e.preventDefault();
            state.spaceHeld = true;
            updateCursorClasses();
        }
        return;
    }

    // Shift+key handlers for "Mark Out" (before keyMap processing)
    if (e.shiftKey) {
        if (e.key === '1' || e.key === '!') {
            e.preventDefault();
            markToolAsOut(1);
            return;
        }
        if (e.key === '2' || e.key === '@') {
            e.preventDefault();
            markToolAsOut(2);
            return;
        }
        if (e.key.toLowerCase() === 'a') {
            e.preventDefault();
            markAllAsOut();
            return;
        }
    }

    // Number keys for tool selection (mask, lines, and keypoints)
    const keyMap = {
        '1': 'tool1_mask',
        '2': 'tool1_top',
        '3': 'tool1_bottom',
        '4': 'tool2_mask',
        '5': 'tool2_top',
        '6': 'tool2_bottom',
        '7': 'tool1_joint',
        '8': 'tool1_ee_tip',
        'q': 'tool1_ee_left',
        'w': 'tool1_ee_right',
        '9': 'tool2_joint',
        '0': 'tool2_ee_tip',
        'o': 'tool2_ee_left',
        'i': 'tool2_ee_right'
    };

    // Normalize key to lowercase for letter keys
    const normalizedKey = e.key.length === 1 ? e.key.toLowerCase() : e.key;

    // Line extraction keys: 2,3 for Tool1; 5,6 for Tool2 (top/bottom only - middle is auto-computed)
    // Works with SAM selections OR existing mask polygons
    const lineKeys = {
        '2': [1, 'top'], '3': [1, 'bottom'],
        '5': [2, 'top'], '6': [2, 'bottom']
    };

    if (lineKeys[normalizedKey] && state.annotationMode !== 'pegs') {
        const [toolNum, lineType] = lineKeys[normalizedKey];

        // SAM apply mode with selected segments
        if (state.samApplyMode && state.selectedSegmentIndices.size > 0) {
            assignSegmentToLine(toolNum, lineType);
            return;
        }
        // Otherwise fall through to the general keyMap handler below,
        // which calls selectTool() — matching button click behavior.
    }

    // SAM apply mode: 1/4 or A/B for mask assignment
    if (state.samApplyMode && state.selectedSegmentIndices.size > 0) {
        if (normalizedKey === '1' || normalizedKey === 'a') { assignSegmentToTool(1); return; }
        if (normalizedKey === '4' || normalizedKey === 'b') { assignSegmentToTool(2); return; }
    }

    if (keyMap[normalizedKey] && state.annotationMode !== 'pegs') {
        selectTool(keyMap[normalizedKey]);
        return;
    }

    // Batch mode intercepts most keys when active
    if (batchState && batchState.active) {
        if (handleBatchKeydown(e)) {
            e.preventDefault();
            return;
        }
    }

    // Phase keybinds (Z, X, D, R, T, G) — only when not in batch mode
    if (PHASE_KEYBINDS[normalizedKey] && !batchState.active) {
        e.preventDefault();
        setPhaseForCurrentFrame(PHASE_KEYBINDS[normalizedKey]);
        return;
    }


    // Peg mode keybinds
    if (!batchState.active && state.annotationMode === 'pegs') {
        // 1-6 = select peg
        if ('123456'.includes(normalizedKey)) {
            selectPeg(parseInt(normalizedKey));
            return;
        }
        // B = bbox, A = mask (when peg selected)
        if (normalizedKey === 'b' && state.selectedPegId) {
            document.getElementById('peg-bbox-btn')?.click();
            return;
        }
        if (normalizedKey === 'a' && state.selectedPegId) {
            document.getElementById('peg-mask-btn')?.click();
            return;
        }
        // P = copy prior peg data
        if (normalizedKey === 'p' && state.selectedPegId) {
            document.getElementById('peg-copy-prior-btn')?.click();
            return;
        }
        // C = clear peg
        if (normalizedKey === 'c' && state.selectedPegId) {
            document.getElementById('peg-clear-btn')?.click();
            return;
        }
        // U/Y/J = peg keypoints KP1/KP2/KP3
        if ((normalizedKey === 'u' || normalizedKey === 'y' || normalizedKey === 'j') && state.selectedPegId) {
            const kpMap = {u: 0, y: 1, j: 2};
            selectPegKeypoint(kpMap[normalizedKey]);
            showToast(`Click to place KP${kpMap[normalizedKey] + 1} for Peg ${state.selectedPegId}`);
            return;
        }
        // L = board outline
        if (normalizedKey === 'l') {
            document.getElementById('pegboard-outline-btn')?.click();
            return;
        }
        // E = toggle Edit All mode
        if (normalizedKey === 'e') {
            document.getElementById('pegboard-edit-btn')?.click();
            return;
        }
        // K = board copy prior
        if (normalizedKey === 'k') {
            document.getElementById('pegboard-copy-btn')?.click();
            return;
        }
    }

    // M = toggle annotation mode (tools/pegs)
    if (normalizedKey === 'm' && !batchState.active) {
        toggleAnnotationMode();
        return;
    }

    // H = increment cycle index
    if (normalizedKey === 'h' && !batchState.active) {
        const cycleInput = document.getElementById('phase-cycle');
        if (cycleInput) {
            cycleInput.value = parseInt(cycleInput.value || 0) + 1;
            showToast(`Cycle: ${cycleInput.value}`);
        }
        return;
    }

    switch (normalizedKey) {
        case 'F5':
            e.preventDefault();
            reloadFrame();
            break;
        case 'b':
            toggleBatchMode();
            break;
        case 'ArrowLeft':
            navigatePrev();
            break;
        case 'ArrowRight':
            navigateNext();
            break;
        case 'e':
            toggleEditMode();
            break;
        case 'f':
            zoomFit();
            break;
        case 's':
            if (state.currentAnnotation && state.currentAnnotation.skipped) {
                unskipFrame();
            } else {
                skipFrame();
            }
            break;
        case 'p':
            // Only copy tool annotations in tools mode (never in pegs mode)
            if (state.annotationMode !== 'pegs') {
                usePriorAnnotation();
            }
            break;
        case 'v':
            if (videoState.active) {
                closeVideoMode();
            } else {
                openVideoMode();
            }
            break;
        case 'c':
            if (confirm('Clear ALL annotations for this frame?')) {
                clearAllAnnotations();
            }
            break;
        case 'Escape':
            // Clear Edit All selection
            if (state.pegEditMode && state.pegEditPhase === 'selected') {
                state.pegEditPhase = 'selecting';
                state.pegEditRect = null;
                state.pegEditSelectedItems = [];
                state.pegEditDragStart = null;
                drawAllAnnotations();
                showToast('Selection cleared — draw new rectangle');
                break;
            }
            if (state.annotationMode === 'pegs' && state.postMaskMode) {
                // Cancel post mask mode
                state.postMaskMode = null;
                state.postMaskTarget = null;
                state.isDrawing = false;
                state.tempPoints = [];
                document.getElementById('pegboard-postmask-btn')?.classList.remove('active');
                drawAllAnnotations();
                showToast('Post mask mode cancelled');
            } else if (state.annotationMode === 'pegs' && (state.pegDrawingTool || state.pegboardTool)) {
                // Cancel peg/pegboard drawing
                state.pegDrawingTool = null;
                state.pegboardTool = null;
                state.pegBboxStart = null;
                state.isDrawing = false;
                state.tempPoints = [];
                clearPegboardBtnActive();
                document.getElementById('peg-bbox-btn')?.classList.remove('active');
                document.getElementById('peg-mask-btn')?.classList.remove('active');
                drawAllAnnotations();
                showToast('Drawing cancelled');
            } else if (state.samApplyMode) {
                exitApplyExistingMode();
            } else if (state.samMode) {
                exitSamMode();
            } else if (state.selectedEdges.length > 0) {
                // Clear multi-edge selection
                state.selectedEdges = [];
                drawAllAnnotations();
                updateInstructions();
            } else if (state.editMode) {
                toggleEditMode();
            } else {
                cancelDrawing();
            }
            break;
        case 'Enter':
            if (state.annotationMode === 'pegs' && state.pegDrawingTool === 'mask' && state.isDrawing && state.tempPoints.length >= 3) {
                finalizePegMask();
            } else if (state.annotationMode === 'pegs' && (state.pegboardTool === 'post') && state.isDrawing && state.tempPoints.length >= 3) {
                finalizePostPlacement();
            } else if (state.annotationMode === 'pegs' && state.pegboardTool === 'outline' && state.isDrawing && state.tempPoints.length >= 3) {
                finalizePegboardOutline();
            } else if (state.annotationMode === 'pegs' && state.postMaskMode === 'drawing' && state.isDrawing && state.tempPoints.length >= 3) {
                finalizePostMask();
            } else if (state.samMode && state.samMaskProposals) {
                acceptSamMask();
            } else if (state.selectedEdges.length > 0) {
                // Finalize multi-edge selection for top/bottom lines
                finalizeLineFromEdges();
            } else if (state.isDrawing && state.currentTool?.includes('mask') && state.tempPoints.length >= 3) {
                finishPolygon(state.currentTool);
                drawAllAnnotations();
            }
            break;
        case 'n':
            // Skip current tool (mark occluded) and move to next
            // Works for all tools: mask, shaft (top/bottom), and keypoints
            if (state.currentTool) {
                e.preventDefault();
                skipCurrentTool();
            }
            break;
        case 'Tab':
            e.preventDefault();
            // Toggle occluded on current tool BUT stay on it (can still place)
            // Works for all tools: mask, shaft (top/bottom), and keypoints
            if (state.currentTool) {
                markCurrentToolOccluded();
            } else if (state.samMode && state.samMaskProposals) {
                cycleSamMask();
            } else {
                // No tool selected - start from beginning
                const firstTool = getNextTool(null) || TOOL_PROGRESSION.full[0];
                if (firstTool) {
                    selectTool(firstTool);
                    showToast(`Selected: ${formatToolName(firstTool)}`);
                }
            }
            break;
        case 'Delete':
            if (state.editMode && state.selectedKeypointKey) {
                pushUndo();
                state.currentAnnotation[state.selectedKeypointKey] = [];
                state.selectedKeypointKey = null;
                saveAnnotations();
                drawAllAnnotations();
                showToast('Keypoint deleted');
            } else if (state.editMode && state.selectedVertices.length > 0) {
                deleteSelectedVertices();
            } else if (state.editMode && state.selectedVertexIndex !== null) {
                deleteSelectedVertex();
            } else {
                clearCurrentTool();
            }
            break;
        case 'Backspace':
            clearCurrentTool();
            break;
    }

    // Ctrl/Cmd+Z for undo (handled after switch since normalizedKey already lowercased)
    if (normalizedKey === 'z' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        undo();
    }
});

// Space keyup for pan release
document.addEventListener('keyup', (e) => {
    if (e.key === ' ' || e.code === 'Space') {
        state.spaceHeld = false;
        if (!state.isPanning) {
            updateCursorClasses();
        }
    }
});

// ============================================================================
// SAM Segmentation (AI-Assisted)
// ============================================================================

async function checkSamStatus() {
    try {
        const status = await api('/sam/status');
        state.samAvailable = status.available;
        updateSamPanelStatus();
        return status;
    } catch (e) {
        state.samAvailable = false;
        updateSamPanelStatus();
        return { available: false };
    }
}

function updateSamPanelStatus() {
    if (!elements.samStatusText) return;

    if (state.samAvailable === false) {
        elements.samStatusText.textContent = 'SAM not available (pip install sam2)';
        if (elements.samNewBtn) elements.samNewBtn.disabled = true;
    } else if (state.samAvailable === true) {
        elements.samStatusText.textContent = 'SAM ready';
    } else {
        elements.samStatusText.textContent = 'Checking SAM...';
    }

    // Update button active states
    if (elements.samApplyBtn) {
        elements.samApplyBtn.classList.toggle('active', state.samApplyMode);
    }
    if (elements.samNewBtn) {
        elements.samNewBtn.classList.toggle('active', state.samMode);
    }
}

// ---------------------------------------------------------------------------
// Apply Existing Mode (pre-computed segments)
// ---------------------------------------------------------------------------

async function enterApplyExistingMode() {
    if (state.samApplyMode) {
        exitApplyExistingMode();
        return;
    }

    if (!state.trialId || state.frameIdx === null) {
        showToast('Load a frame first', true);
        return;
    }

    // Exit other modes
    if (state.samMode) exitSamMode();
    if (state.editMode) toggleEditMode();
    cancelDrawing();

    // Fetch precomputed segments
    elements.samInstructions.textContent = 'Loading pre-computed segments...';

    try {
        let result = await api(`/sam/precomputed/${state.trialId}/frame/${state.frameIdx}`);

        if (!result.available || !result.polygons || result.polygons.length === 0) {
            // No precomputed data — compute on demand
            elements.samInstructions.textContent = 'Computing SAM masks (this may take a moment)...';
            try {
                const computed = await api(`/sam/compute/${state.trialId}/frame/${state.frameIdx}`, { method: 'POST' });
                if (!computed.available || !computed.polygons || computed.polygons.length === 0) {
                    showToast('SAM computation failed or produced no segments', true);
                    elements.samInstructions.textContent = 'No segments available.';
                    return;
                }
                result = computed;
                // Update availability cache — this frame now has SAM data
                state.samFrameAvailability[String(state.frameIdx)] = true;
                updateFrameJumpDropdown();
            } catch (computeErr) {
                console.error('On-demand SAM compute failed:', computeErr);
                showToast('Failed to compute SAM masks. Is SAM2 installed?', true);
                elements.samInstructions.textContent = 'Computation failed.';
                return;
            }
        }

        state.samApplyMode = true;
        state.precomputedSegments = result;
        state.hoveredSegmentIdx = null;
        state.selectedSegmentIndices.clear();

        elements.samInstructions.textContent =
            `${result.polygons.length} segments. Click to select, click again to deselect.`;

        updateSamPanelStatus();
        updateCursorClasses();
        drawAllAnnotations();

    } catch (e) {
        console.error('Failed to load precomputed SAM:', e);
        showToast('Failed to load pre-computed segments', true);
        elements.samInstructions.textContent = 'Error loading segments.';
    }
}

function exitApplyExistingMode() {
    state.samApplyMode = false;
    state.precomputedSegments = null;
    state.hoveredSegmentIdx = null;
    state.selectedSegmentIndices.clear();
    hideToolAssignPrompt();
    updateSamPanelStatus();
    updateCursorClasses();
    drawAllAnnotations();
    elements.samInstructions.textContent = '';
}

function isPointInPolygon(x, y, polygon) {
    // Ray-casting algorithm
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function findSegmentAtPoint(x, y) {
    if (!state.precomputedSegments || !state.precomputedSegments.polygons) return null;

    // Check segments in order (largest first, since they're sorted by area desc)
    // But prefer smaller segments that contain the point (more specific)
    let bestIdx = null;
    let bestArea = Infinity;

    for (let i = 0; i < state.precomputedSegments.polygons.length; i++) {
        const poly = state.precomputedSegments.polygons[i];
        if (!poly || poly.length < 3) continue;

        if (isPointInPolygon(x, y, poly)) {
            const area = state.precomputedSegments.areas[i] || Infinity;
            if (area < bestArea) {
                bestArea = area;
                bestIdx = i;
            }
        }
    }

    return bestIdx;
}

function drawPrecomputedOverlays() {
    if (!state.precomputedSegments || !state.precomputedSegments.polygons) return;

    const ctx = annotationCtx;

    // Draw all segments
    state.precomputedSegments.polygons.forEach((poly, idx) => {
        if (!poly || poly.length < 3) return;

        const isHovered = idx === state.hoveredSegmentIdx;
        const isSelected = state.selectedSegmentIndices.has(idx);

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        if (isSelected) {
            // Selected: cyan fill + thick border
            ctx.fillStyle = 'rgba(0, 200, 255, 0.3)';
            ctx.fill();
            ctx.strokeStyle = '#00c8ff';
            ctx.lineWidth = 2.5;
            ctx.stroke();
        } else if (isHovered) {
            // Hovered: yellow highlight
            ctx.fillStyle = 'rgba(255, 200, 0, 0.2)';
            ctx.fill();
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 2;
            ctx.stroke();
        } else {
            // Default: thin semi-transparent outline
            ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });

    // Draw merged polygon preview if multiple selected
    if (state.selectedSegmentIndices.size > 1) {
        const selectedPolys = Array.from(state.selectedSegmentIndices)
            .map(idx => state.precomputedSegments.polygons[idx])
            .filter(p => p && p.length >= 3);

        if (selectedPolys.length > 1) {
            const merged = mergePolygons(selectedPolys);
            if (merged.length >= 3) {
                // Draw merged polygon outline
                ctx.beginPath();
                ctx.moveTo(merged[0][0], merged[0][1]);
                for (let i = 1; i < merged.length; i++) {
                    ctx.lineTo(merged[i][0], merged[i][1]);
                }
                ctx.closePath();

                ctx.strokeStyle = '#ff6600';  // Orange
                ctx.lineWidth = 2;
                ctx.setLineDash([8, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Label showing count (position at top-left of bounding box)
                const bbox = getCombinedBoundingBox();
                if (bbox) {
                    ctx.fillStyle = '#ff6600';
                    ctx.font = 'bold 14px sans-serif';
                    ctx.fillText(`${state.selectedSegmentIndices.size} → merged`, bbox[0] + 4, bbox[1] - 6);
                }
            }
        }
    }
}

function hideToolAssignPrompt() {
    if (elements.samToolAssign) {
        elements.samToolAssign.style.display = 'none';
    }
}

async function assignSegmentToTool(toolNum) {
    if (state.selectedSegmentIndices.size === 0 || !state.precomputedSegments || !state.currentAnnotation) return;

    const polys = state.precomputedSegments.polygons;
    const selectedIndices = Array.from(state.selectedSegmentIndices);

    let finalPolygon;

    if (selectedIndices.length === 1) {
        // Single selection: use polygon directly
        const poly = polys[selectedIndices[0]];
        if (!poly || poly.length < 3) {
            showToast('Invalid segment', true);
            return;
        }
        finalPolygon = poly;
    } else {
        // Multiple selections: merge polygons via nearest-point connection
        const selectedPolys = selectedIndices
            .map(idx => polys[idx])
            .filter(p => p && p.length >= 3);

        if (selectedPolys.length === 0) {
            showToast('No valid polygons to merge', true);
            return;
        }

        finalPolygon = mergePolygons(selectedPolys);
        if (finalPolygon.length < 3) {
            showToast('Failed to merge polygons', true);
            return;
        }
    }

    pushUndo();

    const maskKey = toolNum === 1 ? 'tool1_mask' : 'tool2_mask';
    state.currentAnnotation[maskKey] = JSON.parse(JSON.stringify(finalPolygon));

    await saveAnnotations();

    const segmentText = selectedIndices.length === 1 ? 'Segment' : `${selectedIndices.length} segments (merged)`;

    // Check if both masks are now assigned
    const ann = state.currentAnnotation;
    const bothMasksAssigned = ann.tool1_mask && ann.tool1_mask.length >= 3 &&
                              ann.tool2_mask && ann.tool2_mask.length >= 3;

    if (bothMasksAssigned) {
        // Both masks done - exit SAM mode, select last mask, enter edit mode
        showToast(`${segmentText} assigned to Tool ${toolNum} - Edit mode`);
        exitApplyExistingMode();
        selectTool(maskKey);
        if (!state.editMode) {
            toggleEditMode();
        }
        drawAllAnnotations();
    } else {
        // First mask done - clear selection, stay in SAM apply mode for next mask
        showToast(`${segmentText} assigned to Tool ${toolNum}`);
        clearSegmentSelection();
    }
}

/**
 * Assign selected SAM segments as a line annotation.
 * @param {number} toolNum - 1 or 2
 * @param {string} lineType - 'top', 'middle', or 'bottom'
 */
async function assignSegmentToLine(toolNum, lineType) {
    if (state.selectedSegmentIndices.size === 0 || !state.precomputedSegments || !state.currentAnnotation) return;

    const polys = state.precomputedSegments.polygons;
    const selectedIndices = Array.from(state.selectedSegmentIndices);

    // Get merged polygon from selected segments
    let mergedPoly;
    if (selectedIndices.length === 1) {
        mergedPoly = polys[selectedIndices[0]];
    } else {
        const selectedPolys = selectedIndices
            .map(idx => polys[idx])
            .filter(p => p && p.length >= 3);
        if (selectedPolys.length === 0) {
            showToast('No valid polygons selected', true);
            return;
        }
        mergedPoly = mergePolygons(selectedPolys);
    }

    if (!mergedPoly || mergedPoly.length < 3) {
        showToast('Invalid polygon for line extraction', true);
        return;
    }

    const linesKey = toolNum === 1 ? 'tool1_lines' : 'tool2_lines';
    const eeTipKey = toolNum === 1 ? 'tool1_ee_tip' : 'tool2_ee_tip';

    pushUndo();

    const currentTool = `tool${toolNum}_${lineType}`;

    if (lineType === 'top') {
        const topEdge = extractTopEdge(mergedPoly);
        if (!topEdge) {
            showToast('Could not extract top edge', true);
            return;
        }
        state.currentAnnotation[linesKey].top = topEdge;
        await saveAnnotations();
        showToast(`Top line assigned to Tool ${toolNum}`);

    } else if (lineType === 'bottom') {
        const bottomEdge = extractBottomEdge(mergedPoly);
        if (!bottomEdge) {
            showToast('Could not extract bottom edge', true);
            return;
        }
        state.currentAnnotation[linesKey].bottom = bottomEdge;
        await saveAnnotations();
        showToast(`Bottom line assigned to Tool ${toolNum}`);
    }

    clearSegmentSelection();
    drawAllAnnotations();

    // Auto-advance to next tool
    advanceToNextTool(currentTool);
}

/**
 * Extract a line from an existing mask polygon.
 * @param {number} toolNum - 1 or 2
 * @param {string} lineType - 'top' or 'bottom' (middle is auto-computed)
 * @returns {boolean} - true if extraction was performed
 */
function extractLineFromExistingMask(toolNum, lineType) {
    if (!state.currentAnnotation) return false;

    const maskKey = toolNum === 1 ? 'tool1_mask' : 'tool2_mask';
    const linesKey = toolNum === 1 ? 'tool1_lines' : 'tool2_lines';
    const polygon = state.currentAnnotation[maskKey];

    if (!polygon || polygon.length < 3) {
        return false;  // No mask to extract from
    }

    pushUndo();

    if (lineType === 'top') {
        const topEdge = extractTopEdge(polygon);
        if (!topEdge) {
            showToast('Could not extract top edge', true);
            return true;
        }
        state.currentAnnotation[linesKey].top = topEdge;
        saveAnnotations();
        drawAllAnnotations();
        showToast(`Top line extracted from Tool ${toolNum} mask`);
        return true;

    } else if (lineType === 'bottom') {
        const bottomEdge = extractBottomEdge(polygon);
        if (!bottomEdge) {
            showToast('Could not extract bottom edge', true);
            return true;
        }
        state.currentAnnotation[linesKey].bottom = bottomEdge;
        saveAnnotations();
        drawAllAnnotations();
        showToast(`Bottom line extracted from Tool ${toolNum} mask`);
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// New SAM Mode (interactive point-based)
// ---------------------------------------------------------------------------

function toggleSamMode() {
    if (state.samAvailable === false) {
        showToast('SAM not available. Install: pip install sam2 torch torchvision', true);
        return;
    }

    if (!state.samMode) {
        // Entering SAM mode
        if (state.samApplyMode) exitApplyExistingMode();
        cancelDrawing();

        if (state.editMode) {
            state.editMode = false;
            state.selectedVertices = [];
            const editBtn = document.getElementById('edit-mode-btn');
            if (editBtn) editBtn.classList.remove('active');
        }

        state.samMode = true;
        state.samPoints = [];
        state.samMaskProposals = null;
        state.samSelectedMask = 0;

        if (!state.currentTool || !state.currentTool.includes('mask')) {
            selectTool('tool1_mask');
        }

        elements.instructions.textContent =
            'SAM: Left-click = foreground, Shift+click = background. Enter to accept, Esc to cancel.';
        elements.samInstructions.textContent = 'Click on the tool to segment.';
        showToast('SAM mode activated');
    } else {
        exitSamMode();
    }

    updateSamPanelStatus();
    updateCursorClasses();
}

function exitSamMode() {
    state.samMode = false;
    state.samPoints = [];
    state.samMaskProposals = null;
    state.samSelectedMask = 0;
    state.samLoading = false;
    updateSamPanelStatus();
    drawAllAnnotations();
    updateInstructions();
    updateCursorClasses();
    elements.samInstructions.textContent = '';
}

async function handleSamClick(x, y, isBackground) {
    if (!state.samMode || !state.trialId || state.frameIdx === null) return;

    const label = isBackground ? 0 : 1;
    state.samPoints.push({ x, y, label });

    drawAllAnnotations();
    drawSamPoints();

    state.samLoading = true;
    elements.instructions.textContent = 'Running SAM segmentation...';

    try {
        const result = await api(
            `/sam/segment/${state.trialId}/${state.frameIdx}`,
            {
                method: 'POST',
                body: JSON.stringify({
                    point_coords: state.samPoints.map(p => [p.x, p.y]),
                    point_labels: state.samPoints.map(p => p.label),
                    multimask: state.samPoints.length === 1
                })
            }
        );

        state.samMaskProposals = result;
        state.samSelectedMask = result.best_idx;
        state.samLoading = false;

        drawAllAnnotations();
        drawSamProposal();
        drawSamPoints();

        const score = (result.scores[result.best_idx] * 100).toFixed(1);
        elements.instructions.textContent =
            `SAM mask (${score}%). Click to add points, Tab to cycle, Enter to accept, Esc to cancel.`;

    } catch (e) {
        state.samLoading = false;
        console.error('SAM segmentation failed:', e);
        showToast('SAM segmentation failed', true);
        elements.instructions.textContent =
            'SAM failed. Try again or Esc to cancel.';
    }
}

function cycleSamMask() {
    if (!state.samMaskProposals || !state.samMaskProposals.masks) return;

    const numMasks = state.samMaskProposals.masks.length;
    if (numMasks <= 1) return;

    state.samSelectedMask = (state.samSelectedMask + 1) % numMasks;
    drawAllAnnotations();
    drawSamProposal();
    drawSamPoints();

    const score = (state.samMaskProposals.scores[state.samSelectedMask] * 100).toFixed(1);
    elements.instructions.textContent =
        `SAM mask ${state.samSelectedMask + 1}/${numMasks} (${score}%). Tab to cycle, Enter to accept, Esc to cancel.`;
}

function acceptSamMask() {
    if (!state.samMaskProposals || !state.currentTool) return;

    const mask = state.samMaskProposals.masks[state.samSelectedMask];
    if (!mask || mask.length < 3) {
        showToast('No valid mask to accept', true);
        return;
    }

    pushUndo();

    const maskKey = state.currentTool.includes('tool1') ? 'tool1_mask' : 'tool2_mask';
    state.currentAnnotation[maskKey] = mask;

    saveAnnotations();
    exitSamMode();
    showToast('SAM mask applied');
}

function drawSamPoints() {
    if (!state.samMode || state.samPoints.length === 0) return;

    const ctx = annotationCtx;

    state.samPoints.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        ctx.strokeStyle = pt.label === 1 ? '#00ff00' : '#ff0000';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = pt.label === 1 ? '#00ff00' : '#ff0000';
        ctx.fill();

        ctx.font = '10px monospace';
        ctx.fillStyle = '#fff';
        ctx.fillText(pt.label === 1 ? '+' : '-', pt.x + 10, pt.y - 5);
    });
}

function drawSamProposal() {
    if (!state.samMaskProposals || !state.samMaskProposals.masks) return;

    const mask = state.samMaskProposals.masks[state.samSelectedMask];
    if (!mask || mask.length < 3) return;

    const ctx = annotationCtx;

    ctx.beginPath();
    ctx.moveTo(mask[0][0], mask[0][1]);
    for (let i = 1; i < mask.length; i++) {
        ctx.lineTo(mask[i][0], mask[i][1]);
    }
    ctx.closePath();

    ctx.fillStyle = 'rgba(255, 200, 0, 0.25)';
    ctx.fill();

    ctx.setLineDash([6, 3]);
    ctx.strokeStyle = '#ffcc00';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);

    mask.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffcc00';
        ctx.fill();
    });
}

// ============================================================================
// Phase & Peg Annotation Mode
// ============================================================================

const PHASE_COLORS = {
    'idle':     {bg: '#6b7280', text: '#fff'},
    'reach':    {bg: '#3b82f6', text: '#fff'},
    'nudge':    {bg: '#f97316', text: '#fff'},
    'grasp':    {bg: '#f59e0b', text: '#000'},
    'transfer': {bg: '#8b5cf6', text: '#fff'},
    'place':    {bg: '#10b981', text: '#fff'},
    'dropped':  {bg: '#ef4444', text: '#fff'},
};

const PHASE_KEYBINDS = {
    'z': 'idle',
    'x': 'reach',
    'c': 'nudge',
    'd': 'grasp',
    'r': 'transfer',
    't': 'place',
    'g': 'dropped',
};

/**
 * Load phase definitions from API (once on init).
 */
async function loadPhaseDefinitions() {
    try {
        state.phaseDefinitions = await api('/phase_definitions');
    } catch (e) {
        console.warn('Failed to load phase definitions:', e);
    }
}

/**
 * Load phase summary for a trial.
 */
async function loadPhaseSummary(trialId) {
    try {
        const summary = await api(`/trials/${trialId}/phase_summary`);
        state.phaseLabels = summary || {};
        drawPhaseStrip();
    } catch (e) {
        console.warn('Failed to load phase summary:', e);
        state.phaseLabels = {};
    }
}

/**
 * Toggle annotation mode between 'tools' and 'pegs'.
 */
function toggleAnnotationMode() {
    state.annotationMode = state.annotationMode === 'tools' ? 'pegs' : 'tools';
    updateAnnotationModeUI();
    rerenderTrialOptionsForMode();
    updateDatasetDropdownPercentage();
    // Cancel any active drawing
    if (state.isDrawing) cancelDrawing();
    state.pegDrawingTool = null;
    state.pegboardTool = null;
    state.pegKeypointIdx = null;
    document.querySelectorAll('.peg-kp-btn').forEach(b => b.classList.remove('active'));
    drawAllAnnotations();
}

/**
 * Update UI to reflect current annotation mode.
 */
function updateAnnotationModeUI() {
    const modeSwitcher = document.getElementById('mode-switcher');
    const toolsPanel = document.getElementById('tools-panel');
    const pegsPanel = document.getElementById('pegs-panel');
    const samPanel = document.getElementById('sam-panel');

    if (!modeSwitcher) return;

    // Update mode buttons
    modeSwitcher.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === state.annotationMode);
    });

    const phaseStrip = document.getElementById('phase-strip');
    const phaseControls = document.getElementById('phase-controls');

    const toolStatusList = document.querySelector('#status-panel .status-list:not(.peg-status-list)');
    const pegStatusList = document.querySelector('#status-panel .peg-status-list');

    if (state.annotationMode === 'tools') {
        if (toolsPanel) toolsPanel.style.display = 'block';
        if (pegsPanel) pegsPanel.style.display = 'none';
        if (samPanel) samPanel.style.display = 'block';
        if (phaseStrip) phaseStrip.style.display = state.showPhaseStrip ? 'block' : 'none';
        if (phaseControls) phaseControls.style.display = state.showPhaseStrip ? 'flex' : 'none';
        if (toolStatusList) toolStatusList.style.display = '';
        if (pegStatusList) pegStatusList.style.display = 'none';
    } else {
        if (toolsPanel) toolsPanel.style.display = 'none';
        if (pegsPanel) pegsPanel.style.display = 'block';
        if (samPanel) samPanel.style.display = 'none';
        if (phaseStrip) phaseStrip.style.display = state.showPhaseStrip ? 'block' : 'none';
        if (phaseControls) phaseControls.style.display = state.showPhaseStrip ? 'flex' : 'none';
        if (toolStatusList) toolStatusList.style.display = 'none';
        if (pegStatusList) pegStatusList.style.display = '';
        updatePegStatus();
    }
}

/**
 * Select a peg for annotation.
 */
function selectPeg(pegId) {
    state.selectedPegId = pegId;
    state.pegDrawingTool = null;
    state.pegKeypointIdx = null;
    document.querySelectorAll('.peg-kp-btn').forEach(b => b.classList.remove('active'));

    // Update peg grid buttons
    document.querySelectorAll('.peg-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.peg) === pegId);
    });

    // Show peg details
    const detailsPanel = document.getElementById('peg-details');
    const titleEl = document.getElementById('peg-detail-title');
    const PEG_COLORS = ['#f97316', '#a855f7', '#06b6d4', '#eab308', '#ec4899', '#84cc16'];
    if (detailsPanel) {
        detailsPanel.style.display = 'block';
        titleEl.textContent = `Peg ${pegId}`;
        titleEl.style.color = PEG_COLORS[pegId - 1];
    }
    // Color Copy Prior text to match selected peg
    const copyPriorBtn = document.getElementById('peg-copy-prior-btn');
    if (copyPriorBtn) {
        copyPriorBtn.style.color = PEG_COLORS[pegId - 1];
    }

    // Load existing peg data for this frame
    loadPegDataToUI(pegId);
    drawAllAnnotations();

    // Auto-activate mask drawing if peg has no mask yet
    const peg = state.currentAnnotation?.pegs?.find(p => p.id === pegId);
    if (!peg || !peg.mask || peg.mask.length === 0) {
        state.pegDrawingTool = 'mask';
        state.pegboardTool = null;
        document.getElementById('peg-mask-btn')?.classList.add('active');
        showToast(`P${pegId}: draw mask polygon`);
    }
}

/**
 * Load peg data from current annotation into UI controls.
 */
function loadPegDataToUI(pegId) {
    if (!state.currentAnnotation) return;
    const pegs = state.currentAnnotation.pegs || [];
    const peg = pegs.find(p => p.id === pegId);

    // State buttons
    const activeState = peg?.state || 'on_source_post';
    document.querySelectorAll('.peg-state-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.state === activeState);
    });
    // Post buttons
    const activePost = peg?.post_id || '';
    document.querySelectorAll('.peg-post-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.post === activePost);
    });
    updatePegKeypointStatus();
}

/**
 * Save peg data from UI controls to current annotation.
 */
function savePegDataFromUI() {
    if (!state.currentAnnotation || !state.selectedPegId) return;
    if (!state.currentAnnotation.pegs) state.currentAnnotation.pegs = [];

    const pegId = state.selectedPegId;
    let peg = state.currentAnnotation.pegs.find(p => p.id === pegId);
    if (!peg) {
        peg = {id: pegId, bbox: [], mask: [], keypoints: [null, null, null], state: 'on_source_post', post_id: null, visible: true};
        state.currentAnnotation.pegs.push(peg);
    }

    peg.state = document.querySelector('.peg-state-btn.active')?.dataset.state || 'on_source_post';
    peg.post_id = document.querySelector('.peg-post-btn.active')?.dataset.post || null;

    state.annotationDirty = true;
    saveAnnotations();
}

/**
 * Find the nearest peg handle (keypoint or bbox corner) within maxDist.
 * Returns {pegId, handleType, index, x, y} or null.
 */
function findNearestPegHandle(px, py, maxDist) {
    if (!state.currentAnnotation?.pegs) return null;
    let best = null;
    let bestDist = maxDist;

    for (const peg of state.currentAnnotation.pegs) {
        // Check keypoints
        if (peg.keypoints) {
            for (let i = 0; i < peg.keypoints.length; i++) {
                const kp = peg.keypoints[i];
                if (!kp) continue;
                const d = Math.sqrt((px - kp[0]) ** 2 + (py - kp[1]) ** 2);
                if (d < bestDist) {
                    bestDist = d;
                    best = {pegId: peg.id, handleType: 'keypoint', index: i, x: kp[0], y: kp[1]};
                }
            }
        }
        // Check bbox corners [x, y, w, h] → TL(0), TR(1), BL(2), BR(3)
        if (peg.bbox && peg.bbox.length === 4) {
            const [bx, by, bw, bh] = peg.bbox;
            const corners = [
                [bx, by],           // TL
                [bx + bw, by],      // TR
                [bx, by + bh],      // BL
                [bx + bw, by + bh], // BR
            ];
            for (let i = 0; i < 4; i++) {
                const d = Math.sqrt((px - corners[i][0]) ** 2 + (py - corners[i][1]) ** 2);
                if (d < bestDist) {
                    bestDist = d;
                    best = {pegId: peg.id, handleType: 'bbox_corner', index: i, x: corners[i][0], y: corners[i][1]};
                }
            }
        }
        // Check mask vertices
        if (peg.mask && peg.mask.length > 0) {
            for (let i = 0; i < peg.mask.length; i++) {
                const v = peg.mask[i];
                const d = Math.sqrt((px - v[0]) ** 2 + (py - v[1]) ** 2);
                if (d < bestDist) {
                    bestDist = d;
                    best = {pegId: peg.id, handleType: 'mask_vertex', index: i, x: v[0], y: v[1]};
                }
            }
        }
    }
    return best;
}

/**
 * Find the nearest pegboard handle (source/target post or board outline vertex).
 * Returns {handleType: 'source_post'|'target_post'|'board_vertex', index, x, y} or null.
 */
function findNearestPegboardHandle(px, py, maxDist) {
    const pegboard = state.currentAnnotation?.pegboard;
    if (!pegboard) return null;
    let best = null;
    let bestDist = maxDist;

    // Source posts
    (pegboard.source_posts || []).forEach((pt, i) => {
        const d = Math.sqrt((px - pt[0]) ** 2 + (py - pt[1]) ** 2);
        if (d < bestDist) { bestDist = d; best = {handleType: 'source_post', index: i, x: pt[0], y: pt[1]}; }
    });
    // Target posts
    (pegboard.target_posts || []).forEach((pt, i) => {
        const d = Math.sqrt((px - pt[0]) ** 2 + (py - pt[1]) ** 2);
        if (d < bestDist) { bestDist = d; best = {handleType: 'target_post', index: i, x: pt[0], y: pt[1]}; }
    });
    // Board outline vertices
    (pegboard.board_mask || []).forEach((pt, i) => {
        const d = Math.sqrt((px - pt[0]) ** 2 + (py - pt[1]) ** 2);
        if (d < bestDist) { bestDist = d; best = {handleType: 'board_vertex', index: i, x: pt[0], y: pt[1]}; }
    });
    // Post mask vertices
    ['source_post_masks', 'target_post_masks'].forEach(masksKey => {
        (pegboard[masksKey] || []).forEach((mask, maskIdx) => {
            if (!mask) return;
            mask.forEach((pt, vtxIdx) => {
                const d = Math.sqrt((px - pt[0]) ** 2 + (py - pt[1]) ** 2);
                if (d < bestDist) {
                    bestDist = d;
                    best = {handleType: 'post_mask_vertex', masksKey, maskIndex: maskIdx, index: vtxIdx, x: pt[0], y: pt[1]};
                }
            });
        });
    });
    return best;
}

/**
 * Find all peg/pegboard items with any vertex inside the selection rectangle.
 * Returns array of {type, ref} objects describing what to translate.
 */
function findItemsInPegEditRect(rect) {
    const rx1 = Math.min(rect.x1, rect.x2), ry1 = Math.min(rect.y1, rect.y2);
    const rx2 = Math.max(rect.x1, rect.x2), ry2 = Math.max(rect.y1, rect.y2);
    const inside = (pt) => pt && pt[0] >= rx1 && pt[0] <= rx2 && pt[1] >= ry1 && pt[1] <= ry2;
    const anyInside = (pts) => pts && pts.some(pt => pt && inside(pt));
    const items = [];

    // Peg masks, keypoints, bboxes
    (state.currentAnnotation?.pegs || []).forEach(peg => {
        if (peg.mask && anyInside(peg.mask)) {
            items.push({type: 'peg_mask', pegId: peg.id});
        }
        if (peg.keypoints && anyInside(peg.keypoints)) {
            items.push({type: 'peg_kps', pegId: peg.id});
        }
        if (peg.bbox && peg.bbox.length === 4) {
            const [bx, by, bw, bh] = peg.bbox;
            if (anyInside([[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]])) {
                items.push({type: 'peg_bbox', pegId: peg.id});
            }
        }
    });

    // Pegboard elements
    const pb = state.currentAnnotation?.pegboard;
    if (pb) {
        if (pb.board_mask && anyInside(pb.board_mask)) {
            items.push({type: 'board_mask'});
        }
        ['source', 'target'].forEach(side => {
            const posts = pb[`${side}_posts`] || [];
            const masks = pb[`${side}_post_masks`] || [];
            const kps = pb[`${side}_post_keypoints`] || [];
            for (let i = 0; i < 6; i++) {
                const hasMask = masks[i] && anyInside(masks[i]);
                const hasPost = posts[i] && inside(posts[i]);
                const hasKp = kps[i] && inside(kps[i]);
                if (hasMask || hasPost || hasKp) {
                    items.push({type: 'post', side, index: i});
                }
            }
        });
    }
    return items;
}

/**
 * Translate all selected peg-edit items by (dx, dy).
 */
function translatePegEditSelection(dx, dy) {
    const rdx = Math.round(dx), rdy = Math.round(dy);
    const movePt = (pt) => pt ? [pt[0] + rdx, pt[1] + rdy] : pt;
    const movePts = (pts) => pts ? pts.map(movePt) : pts;

    for (const item of state.pegEditSelectedItems) {
        if (item.type === 'peg_mask' || item.type === 'peg_kps' || item.type === 'peg_bbox') {
            const peg = state.currentAnnotation?.pegs?.find(p => p.id === item.pegId);
            if (!peg) continue;
            if (item.type === 'peg_mask' && peg.mask) peg.mask = movePts(peg.mask);
            if (item.type === 'peg_kps' && peg.keypoints) peg.keypoints = peg.keypoints.map(movePt);
            if (item.type === 'peg_bbox' && peg.bbox) {
                peg.bbox = [peg.bbox[0] + rdx, peg.bbox[1] + rdy, peg.bbox[2], peg.bbox[3]];
            }
        } else if (item.type === 'board_mask') {
            const pb = state.currentAnnotation?.pegboard;
            if (pb?.board_mask) pb.board_mask = movePts(pb.board_mask);
        } else if (item.type === 'post') {
            const pb = state.currentAnnotation?.pegboard;
            if (!pb) continue;
            const postsKey = `${item.side}_posts`;
            const masksKey = `${item.side}_post_masks`;
            const kpsKey = `${item.side}_post_keypoints`;
            if (pb[postsKey]?.[item.index]) pb[postsKey][item.index] = movePt(pb[postsKey][item.index]);
            if (pb[masksKey]?.[item.index]) pb[masksKey][item.index] = movePts(pb[masksKey][item.index]);
            if (pb[kpsKey]?.[item.index]) pb[kpsKey][item.index] = movePt(pb[kpsKey][item.index]);
        }
    }
}

/**
 * Draw blue dashed overlay rectangle for the current peg-edit selection.
 */
function drawPegEditSelectionOverlay() {
    if (!state.pegEditRect) return;
    const r = state.pegEditRect;
    const rx = Math.min(r.x1, r.x2), ry = Math.min(r.y1, r.y2);
    const rw = Math.abs(r.x2 - r.x1), rh = Math.abs(r.y2 - r.y1);
    annotationCtx.beginPath();
    annotationCtx.rect(rx, ry, rw, rh);
    annotationCtx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    annotationCtx.lineWidth = 2 / state.zoom;
    annotationCtx.setLineDash([6 / state.zoom, 4 / state.zoom]);
    annotationCtx.stroke();
    annotationCtx.setLineDash([]);
    annotationCtx.fillStyle = 'rgba(59, 130, 246, 0.05)';
    annotationCtx.fill();
}

/**
 * Handle canvas click in pegs mode (bbox, mask, pegboard post placement).
 */
function handlePegCanvasClick(x, y) {
    if (!state.currentAnnotation) return;

    // --- Peg BBox drawing ---
    if (state.pegDrawingTool === 'bbox' && state.selectedPegId) {
        if (!state.pegBboxStart) {
            // First corner
            state.pegBboxStart = [x, y];
            state.tempPoints = [];
            showToast('Click second corner for bbox');
        } else {
            // Second corner → finalize bbox
            const [x1, y1] = state.pegBboxStart;
            const bx = Math.min(x1, x);
            const by = Math.min(y1, y);
            const bw = Math.abs(x - x1);
            const bh = Math.abs(y - y1);

            if (!state.currentAnnotation.pegs) state.currentAnnotation.pegs = [];
            let peg = state.currentAnnotation.pegs.find(p => p.id === state.selectedPegId);
            if (!peg) {
                peg = {id: state.selectedPegId, bbox: [], mask: [], keypoints: [null, null, null], state: 'on_source_post', post_id: null, visible: true};
                state.currentAnnotation.pegs.push(peg);
            }
            peg.bbox = [Math.round(bx), Math.round(by), Math.round(bw), Math.round(bh)];

            state.pegBboxStart = null;
            state.tempPoints = [];
            state.annotationDirty = true;
            saveAnnotations();
            drawAllAnnotations();
            showToast(`Peg ${state.selectedPegId} bbox saved`);
        }
        return;
    }

    // --- Peg Mask drawing (polygon) ---
    if (state.pegDrawingTool === 'mask' && state.selectedPegId) {
        if (!state.isDrawing) {
            state.isDrawing = true;
            state.tempPoints = [[x, y]];
            showToast('Click to add points, snap to first or Enter to finish');
        } else {
            // Snap-to-first to close
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    finalizePegMask();
                    return;
                }
            }
            state.tempPoints.push([x, y]);
        }
        drawAllAnnotations();
        return;
    }

    // --- Peg Keypoint placement ---
    if (state.pegKeypointIdx !== null && state.selectedPegId) {
        if (!state.currentAnnotation.pegs) state.currentAnnotation.pegs = [];
        let peg = state.currentAnnotation.pegs.find(p => p.id === state.selectedPegId);
        if (!peg) {
            peg = {id: state.selectedPegId, bbox: [], mask: [], keypoints: [null, null, null], state: 'on_source_post', post_id: null, visible: true};
            state.currentAnnotation.pegs.push(peg);
        }
        if (!peg.keypoints) peg.keypoints = [null, null, null];
        peg.keypoints[state.pegKeypointIdx] = [Math.round(x), Math.round(y)];

        state.annotationDirty = true;
        saveAnnotations();
        drawAllAnnotations();
        updatePegKeypointStatus();

        // Auto-advance to next unset keypoint
        const nextIdx = peg.keypoints.findIndex((kp, i) => i > state.pegKeypointIdx && kp === null);
        if (nextIdx >= 0) {
            selectPegKeypoint(nextIdx);
            showToast(`KP${state.pegKeypointIdx + 1} placed — click KP${nextIdx + 1}`);
        } else {
            selectPegKeypoint(null);
            if (state.selectedPegId < 6) {
                selectPeg(state.selectedPegId + 1);
                showToast(`P${state.selectedPegId}: draw mask polygon`);
            } else {
                showToast('All 6 pegs annotated!');
            }
        }
        return;
    }

    // --- Post Mask drawing ---
    if (state.postMaskMode === 'select_post') {
        // Find nearest post to click
        const pegboard = state.currentAnnotation?.pegboard;
        if (pegboard) {
            let best = null, bestDist = 20 / state.zoom;
            (pegboard.source_posts || []).forEach((pt, i) => {
                const d = Math.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2);
                if (d < bestDist) { bestDist = d; best = {type: 'source', index: i}; }
            });
            (pegboard.target_posts || []).forEach((pt, i) => {
                const d = Math.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2);
                if (d < bestDist) { bestDist = d; best = {type: 'target', index: i}; }
            });
            if (best) {
                state.postMaskTarget = best;
                state.postMaskMode = 'drawing';
                state.isDrawing = false;
                state.tempPoints = [];
                const label = best.type === 'source' ? `S${best.index + 1}` : `T${best.index + 1}`;
                showToast(`Draw mask for post ${label}, snap to first or Enter to finish`);
            } else {
                showToast('Click on a placed post', true);
            }
        }
        return;
    }
    if (state.postMaskMode === 'drawing' && state.postMaskTarget) {
        if (!state.isDrawing) {
            state.isDrawing = true;
            state.tempPoints = [[x, y]];
            showToast('Click to add points, snap to first or Enter to finish');
        } else {
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    finalizePostMask();
                    return;
                }
            }
            state.tempPoints.push([x, y]);
        }
        drawAllAnnotations();
        return;
    }

    // --- Post keypoint placement ---
    if (state.pegboardTool === 'post_keypoint' && state.activePostTarget) {
        const pegboard = state.currentAnnotation?.pegboard;
        if (!pegboard) return;
        const {type, index} = state.activePostTarget;
        const kpKey = type === 'source' ? 'source_post_keypoints' : 'target_post_keypoints';
        if (!pegboard[kpKey]) pegboard[kpKey] = [null, null, null, null, null, null];
        pegboard[kpKey][index] = [Math.round(x), Math.round(y)];
        const label = type === 'source' ? `S${index + 1}` : `T${index + 1}`;
        state.pegboardTool = null;
        state.activePostTarget = null;
        clearPegboardBtnActive();
        state.annotationDirty = true;
        saveAnnotations();
        drawAllAnnotations();
        updatePostButtonStatus();
        showToast(`${label} keypoint placed`);
        return;
    }

    // --- Individual post mask drawing (polygon) ---
    if (state.pegboardTool === 'post' && state.activePostTarget) {
        if (!state.currentAnnotation.pegboard) {
            state.currentAnnotation.pegboard = {source_posts: [], target_posts: [], source_post_masks: [[], [], [], [], [], []], target_post_masks: [[], [], [], [], [], []], board_mask: []};
        }
        const {type, index} = state.activePostTarget;
        const label = type === 'source' ? `S${index + 1}` : `T${index + 1}`;

        if (!state.isDrawing) {
            state.isDrawing = true;
            state.tempPoints = [[x, y]];
            showToast(`Drawing ${label} mask — snap to first or Enter to finish`);
        } else {
            // Snap-to-first click
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    finalizePostPlacement();
                    return;
                }
            }
            state.tempPoints.push([x, y]);
        }
        drawAllAnnotations();
        return;
    }

    // --- Pegboard Board Outline (polygon) ---
    if (state.pegboardTool === 'outline') {
        if (!state.isDrawing) {
            state.isDrawing = true;
            state.tempPoints = [[x, y]];
            showToast('Click to add outline points, snap to first or Enter to finish');
        } else {
            if (state.tempPoints.length >= 3) {
                const first = state.tempPoints[0];
                const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2);
                if (dist < 15 / state.zoom) {
                    finalizePegboardOutline();
                    return;
                }
            }
            state.tempPoints.push([x, y]);
        }
        drawAllAnnotations();
        return;
    }
}

/**
 * Finalize peg mask polygon from tempPoints.
 */
function finalizePegMask() {
    if (!state.currentAnnotation || !state.selectedPegId || state.tempPoints.length < 3) return;
    if (!state.currentAnnotation.pegs) state.currentAnnotation.pegs = [];
    let peg = state.currentAnnotation.pegs.find(p => p.id === state.selectedPegId);
    if (!peg) {
        peg = {id: state.selectedPegId, bbox: [], mask: [], keypoints: [null, null, null], state: 'on_source_post', post_id: null, visible: true};
        state.currentAnnotation.pegs.push(peg);
    }
    peg.mask = state.tempPoints.map(pt => [Math.round(pt[0]), Math.round(pt[1])]);
    // Auto-compute bounding box from mask vertices
    if (peg.mask.length >= 3) {
        const xs = peg.mask.map(p => p[0]);
        const ys = peg.mask.map(p => p[1]);
        const minX = Math.min(...xs), minY = Math.min(...ys);
        const maxX = Math.max(...xs), maxY = Math.max(...ys);
        peg.bbox = [minX, minY, maxX - minX, maxY - minY];
    }
    state.isDrawing = false;
    state.tempPoints = [];
    state.annotationDirty = true;
    saveAnnotations();
    drawAllAnnotations();
    // Auto-flow: switch to keypoint mode after mask completion
    state.pegDrawingTool = null;
    document.getElementById('peg-mask-btn')?.classList.remove('active');
    const firstUnsetKp = peg.keypoints ? peg.keypoints.findIndex(kp => kp === null) : 0;
    if (firstUnsetKp >= 0) {
        selectPegKeypoint(firstUnsetKp);
        showToast(`Mask saved — now place KP${firstUnsetKp + 1}`);
    } else {
        showToast(`Peg ${state.selectedPegId} mask saved (${peg.mask.length} points)`);
    }
}

/**
 * Finalize pegboard board outline polygon.
 */
function finalizePegboardOutline() {
    if (!state.currentAnnotation || state.tempPoints.length < 3) return;
    if (!state.currentAnnotation.pegboard) {
        state.currentAnnotation.pegboard = {source_posts: [], target_posts: [], board_mask: []};
    }
    state.currentAnnotation.pegboard.board_mask = state.tempPoints.map(pt => [Math.round(pt[0]), Math.round(pt[1])]);
    state.isDrawing = false;
    state.tempPoints = [];
    state.pegboardTool = null;
    clearPegboardBtnActive();
    state.annotationDirty = true;
    saveAnnotations();
    drawAllAnnotations();
    updatePegStatus();
    showToast('Board outline saved');
}

/**
 * Finalize source/target post placement from polygon tempPoints.
 * Stores polygon mask and computes centroid for label position.
 */
function finalizePostPlacement() {
    if (!state.currentAnnotation || !state.activePostTarget || state.tempPoints.length < 3) return;
    const pegboard = state.currentAnnotation.pegboard;
    if (!pegboard) return;
    const {type, index} = state.activePostTarget;
    const isSource = type === 'source';
    const key = isSource ? 'source_posts' : 'target_posts';
    const masksKey = isSource ? 'source_post_masks' : 'target_post_masks';
    const label = isSource ? `S${index + 1}` : `T${index + 1}`;

    // Ensure arrays exist with correct length
    if (!pegboard[masksKey]) pegboard[masksKey] = [[], [], [], [], [], []];
    while (pegboard[key].length <= index) pegboard[key].push(null);

    // Store polygon mask
    const mask = state.tempPoints.map(pt => [Math.round(pt[0]), Math.round(pt[1])]);
    pegboard[masksKey][index] = mask;

    // Compute centroid for label position
    const cx = Math.round(mask.reduce((s, p) => s + p[0], 0) / mask.length);
    const cy = Math.round(mask.reduce((s, p) => s + p[1], 0) / mask.length);
    pegboard[key][index] = [cx, cy];

    state.isDrawing = false;
    state.tempPoints = [];
    state.annotationDirty = true;
    saveAnnotations();
    drawAllAnnotations();
    updatePostButtonStatus();

    // Auto-transition to keypoint placement
    state.pegboardTool = 'post_keypoint';
    // Keep state.activePostTarget so we know which post to place keypoint for
    showToast(`${label} mask saved — place center-top keypoint`);
}

/**
 * Finalize post mask polygon from tempPoints.
 */
function finalizePostMask() {
    if (!state.currentAnnotation || !state.postMaskTarget || state.tempPoints.length < 3) return;
    const pegboard = state.currentAnnotation.pegboard;
    if (!pegboard) return;
    const {type, index} = state.postMaskTarget;
    const masksKey = type === 'source' ? 'source_post_masks' : 'target_post_masks';
    if (!pegboard[masksKey]) pegboard[masksKey] = [[], [], [], [], [], []];
    pegboard[masksKey][index] = state.tempPoints.map(pt => [Math.round(pt[0]), Math.round(pt[1])]);
    state.isDrawing = false;
    state.tempPoints = [];
    state.postMaskMode = 'select_post';  // Stay in post mask mode for next post
    state.postMaskTarget = null;
    state.annotationDirty = true;
    saveAnnotations();
    drawAllAnnotations();
    const label = type === 'source' ? `S${index + 1}` : `T${index + 1}`;
    showToast(`Post ${label} mask saved — click another post or press Esc`);
}

/**
 * Update pegboard post counters in the UI.
 */
function updatePegboardCounters() {
    updatePostButtonStatus();
    if (state.annotationMode === 'pegs') updatePegStatus();
}

/**
 * Update individual post button status (done/not done).
 */
function updatePostButtonStatus() {
    const pegboard = state.currentAnnotation?.pegboard;
    document.querySelectorAll('.post-btn').forEach(btn => {
        const postType = btn.dataset.postType;
        const postIdx = parseInt(btn.dataset.postIdx);
        const masksKey = postType === 'source' ? 'source_post_masks' : 'target_post_masks';
        const mask = pegboard?.[masksKey]?.[postIdx];
        if (mask && mask.length >= 3) {
            btn.classList.add('done');
        } else {
            btn.classList.remove('done');
        }
    });
}

/**
 * Clear active state from pegboard and post buttons.
 */
function clearPegboardBtnActive() {
    ['pegboard-outline-btn', 'pegboard-edit-btn'].forEach(id => {
        document.getElementById(id)?.classList.remove('active');
    });
    document.querySelectorAll('.post-btn').forEach(btn => btn.classList.remove('active'));
    state.pegEditMode = false;
    state.pegEditPhase = null;
    state.pegEditRect = null;
    state.pegEditSelectedItems = [];
    state.pegEditDragStart = null;
    state.postMaskMode = null;
    state.postMaskTarget = null;
    state.activePostTarget = null;
}

/**
 * Select a peg keypoint index for placement (0, 1, 2, or null to deselect).
 */
function selectPegKeypoint(kpIdx) {
    state.pegKeypointIdx = kpIdx;
    // Deactivate other drawing tools when selecting a keypoint
    if (kpIdx !== null) {
        state.pegDrawingTool = null;
        state.pegboardTool = null;
        state.pegBboxStart = null;
        state.isDrawing = false;
        state.tempPoints = [];
        document.getElementById('peg-bbox-btn')?.classList.remove('active');
        document.getElementById('peg-mask-btn')?.classList.remove('active');
        clearPegboardBtnActive();
    }
    // Update button highlights
    document.querySelectorAll('.peg-kp-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.kp) === kpIdx);
    });
    updatePegKeypointStatus();
}

/**
 * Update peg keypoint status text showing which are set.
 */
function updatePegKeypointStatus() {
    const el = document.getElementById('peg-kp-status');
    if (!el) return;
    if (!state.selectedPegId || !state.currentAnnotation) {
        el.textContent = '';
        return;
    }
    const pegs = state.currentAnnotation.pegs || [];
    const peg = pegs.find(p => p.id === state.selectedPegId);
    const kps = peg?.keypoints || [null, null, null];
    el.innerHTML = kps.map((kp, i) => {
        const check = kp ? '\u2713' : '\u2717';
        const color = kp ? 'var(--accent-green)' : 'var(--text-secondary)';
        return `<span style="color:${color}">KP${i + 1}:${check}</span>`;
    }).join('');
}

/**
 * Set coarse phase for current frame.
 */
function setPhaseForCurrentFrame(coarsePhase) {
    if (!state.currentAnnotation) return;
    if (!state.currentAnnotation.phase) {
        state.currentAnnotation.phase = {coarse: '', fine: '', cycle_index: 0, active_tool: 0, events: []};
    }
    const ph = state.currentAnnotation.phase;
    ph.coarse = coarsePhase;
    // Mirror to per-tool fields (both tools same) so the on-disk schema stays consistent.
    ph.tool1 = coarsePhase;
    ph.tool2 = coarsePhase;

    const cycleInput = document.getElementById('phase-cycle');
    const toolSelect = document.getElementById('phase-active-tool');
    if (cycleInput) ph.cycle_index = parseInt(cycleInput.value) || 0;
    if (toolSelect) ph.active_tool = parseInt(toolSelect.value) || 0;

    state.phaseLabels[state.frameIdx] = {
        coarse: coarsePhase,
        tool1: coarsePhase,
        tool2: coarsePhase,
        cycle_index: ph.cycle_index,
    };

    state.annotationDirty = true;
    saveAnnotations();
    drawPhaseStrip();
    updatePhaseControlsHighlight(coarsePhase);
    showToast(`Phase: ${coarsePhase}`);
}

/**
 * Highlight the active phase button.
 */
function updatePhaseControlsHighlight(coarsePhase) {
    document.querySelectorAll('.phase-controls .phase-btn').forEach(btn => {
        btn.classList.toggle('active-phase', btn.dataset.phase === coarsePhase);
    });
}

/**
 * Draw the phase timeline strip canvas.
 */
function drawPhaseStrip() {
    const canvas = document.getElementById('phase-strip-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    const h = canvas.height;

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, h);

    const totalFrames = state.allFrames.length || state.sampledFrames.length;
    if (!totalFrames) return;

    // Draw phase color bars (single row, legacy `coarse`)
    if (state.phaseLabels) {
        for (const [frameIdxStr, phaseData] of Object.entries(state.phaseLabels)) {
            const coarse = (typeof phaseData === 'string')
                ? phaseData
                : (phaseData.coarse || phaseData.tool1 || phaseData.tool2 || '');
            if (!coarse) continue;
            const frameIdx = parseInt(frameIdxStr);
            const x = (frameIdx / totalFrames) * canvas.width;
            const w = Math.max(2, canvas.width / totalFrames);
            ctx.fillStyle = PHASE_COLORS[coarse]?.bg || '#555';
            ctx.fillRect(x, 0, w, h - 4);
        }
    }

    // Current frame cursor
    if (state.frameIdx !== null) {
        const x = (state.frameIdx / totalFrames) * canvas.width;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x - 1, 0, 3, h);
    }

    // Tick marks every 10% for orientation
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    for (let i = 1; i < 10; i++) {
        const x = (i / 10) * canvas.width;
        ctx.fillRect(x, h - 4, 1, 4);
    }
}

/**
 * Update phase controls from current annotation.
 */
function updatePhaseControls() {
    if (!state.currentAnnotation) return;
    const phase = state.currentAnnotation.phase || {};
    updatePhaseControlsHighlight(phase.coarse || '');

    const cycleInput = document.getElementById('phase-cycle');
    const toolSelect = document.getElementById('phase-active-tool');
    if (cycleInput) cycleInput.value = phase.cycle_index || 0;
    if (toolSelect) toolSelect.value = phase.active_tool || 0;
}

// ============================================================================
// Batch Annotation Mode
// ============================================================================

const batchState = {
    active: false,
    allFrameIndices: [],
    currentPage: 0,
    pageSize: 55,
    selectedFrames: new Set(),
    lastClickedFrame: null,
    frameVisibility: {},  // frameIdx -> {t1: {mask, lines, ...}, t2: {...}}
    frameExclude: {},     // frameIdx -> boolean (training-only flag)
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragRect: null,
    showAllFrames: true,  // Toggle between all frames vs sampled
    bookendPending: false,     // true when on bookend page with unloaded thumbnails
    unloadedThumbnails: new Set(),  // frame indices not yet loaded on current page
    // phaseMode removed — phase buttons are now always inline
    pegCopyMode: null,        // null | 'select_source' | 'select_targets'
    pegCopySource: null,      // frame index
    pegCopyTargets: new Set(),
    // Prefetch state
    prefetchCache: new Map(),      // frameIdx -> Image (loaded next-page thumbnails)
    prefetchPage: -1,              // which page was prefetched
    prefetchInFlight: 0,           // number of prefetch requests currently loading
    currentPageLoaded: 0,          // count of loaded images on current page
    currentPageTotal: 0,           // total images on current page
};

const batchElements = {
    batchView: document.getElementById('batch-view'),
    batchGrid: document.getElementById('batch-grid'),
    batchPageInfo: document.getElementById('batch-page-info'),
    batchPageSelect: document.getElementById('batch-page-select'),
    batchGotoFrame: document.getElementById('batch-goto-frame'),
    batchCount: document.getElementById('batch-count'),
    batchPrev: document.getElementById('batch-prev'),
    batchNext: document.getElementById('batch-next'),
    batchClose: document.getElementById('batch-close'),
    batchT1Out: document.getElementById('batch-t1-out'),
    batchT2Out: document.getElementById('batch-t2-out'),
    batchBothOut: document.getElementById('batch-both-out'),
    batchClearSelection: document.getElementById('batch-clear-selection'),
    batchShowAll: document.getElementById('batch-show-all'),
    batchRefresh: document.getElementById('batch-refresh'),
    batchBroken: document.getElementById('batch-broken'),
    batchExclude: document.getElementById('batch-exclude'),
    batchModeBtn: document.getElementById('batch-mode-btn'),
    batchFirst: document.getElementById('batch-first'),
    batchLast: document.getElementById('batch-last'),
    batchNextTrial: document.getElementById('batch-next-trial')
};

/**
 * Toggle batch annotation mode.
 */
async function toggleBatchMode() {
    if (batchState.active) {
        closeBatchMode();
    } else {
        await openBatchMode();
    }
}

/**
 * Open batch annotation mode.
 */
async function openBatchMode() {
    if (!state.trialId) {
        showToast('Load a trial first', true);
        return;
    }

    try {
        batchState.active = true;
        batchState.currentPage = 0;
        batchState.selectedFrames.clear();
        batchState.lastClickedFrame = null;

        // Determine frame list based on toggle
        await updateBatchFrameList();

        // Load visibility data
        await loadBatchVisibility();

        // Show batch view, then defer rendering until after paint
        batchElements.batchView.style.display = 'flex';
        requestAnimationFrame(() => {
            renderBatchPage();
            updateBatchSelectionUI();
        });
    } catch (e) {
        console.error('Failed to open batch mode:', e);
        showToast('Failed to open batch mode', true);
        closeBatchMode();
    }
}

/**
 * Close batch annotation mode.
 */
function closeBatchMode() {
    batchState.active = false;
    batchElements.batchView.style.display = 'none';

    // Clean up prefetch cache
    batchState.prefetchCache.clear();
    batchState.prefetchPage = -1;

    // Clean up drag rect if exists
    if (batchState.dragRect) {
        batchState.dragRect.remove();
        batchState.dragRect = null;
    }

    // Refresh main view progress
    loadTrialFrames(state.trialId);
}

/**
 * Update the batch frame list based on showAllFrames toggle.
 */
async function updateBatchFrameList() {
    if (batchState.showAllFrames) {
        // Use all frames
        if (state.allFrames.length === 0) {
            await loadAllFrames(state.trialId);
        }
        batchState.allFrameIndices = [...state.allFrames];
    } else {
        // Use sampled frames
        batchState.allFrameIndices = [...state.sampledFrames];
    }
    batchState.currentPage = 0;
}

/**
 * Load visibility status for all frames.
 */
async function loadBatchVisibility() {
    try {
        const visibility = await api(`/frames/${state.trialId}/visibility`);
        batchState.frameVisibility = visibility;
        // Extract exclude flags from visibility data
        const excludeMap = {};
        for (const [frameIdx, data] of Object.entries(visibility)) {
            if (data.exclude) {
                excludeMap[frameIdx] = true;
            }
        }
        batchState.frameExclude = excludeMap;
    } catch (e) {
        console.error('Failed to load visibility:', e);
        batchState.frameVisibility = {};
        batchState.frameExclude = {};
    }
}

/**
 * Render the current page of thumbnails.
 */
function renderBatchPage() {
    const grid = batchElements.batchGrid;

    // Cancel in-flight thumbnail requests before removing elements
    grid.querySelectorAll('img').forEach(img => { img.src = ''; });
    grid.innerHTML = '';

    const startIdx = batchState.currentPage * batchState.pageSize;
    const endIdx = Math.min(startIdx + batchState.pageSize, batchState.allFrameIndices.length);
    const pageFrames = batchState.allFrameIndices.slice(startIdx, endIdx);

    // Update page info
    const total = batchState.allFrameIndices.length;
    const totalPages = Math.ceil(total / batchState.pageSize);
    batchElements.batchPageInfo.textContent = `${startIdx + 1}-${endIdx} of ${total}`;

    // Rebuild page dropdown
    const select = batchElements.batchPageSelect;
    const prevLen = select.options.length;
    if (prevLen !== totalPages) {
        select.innerHTML = '';
        for (let p = 0; p < totalPages; p++) {
            const s = p * batchState.pageSize;
            const e = Math.min(s + batchState.pageSize, total);
            const firstFrame = batchState.allFrameIndices[s];
            const lastFrame = batchState.allFrameIndices[e - 1];
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = `Page ${p + 1}  (#${firstFrame}–#${lastFrame})`;
            select.appendChild(opt);
        }
    }
    select.value = batchState.currentPage;

    // Update nav button states
    batchElements.batchPrev.disabled = batchState.currentPage === 0;
    batchElements.batchNext.disabled = endIdx >= total;

    // Bookend detection: first/last page only load one thumbnail
    const isFirstPage = batchState.currentPage === 0;
    const isLastPage = batchState.currentPage === totalPages - 1;
    const isBookend = isFirstPage || isLastPage;
    let bookendFrameIdx = null;
    if (isBookend && pageFrames.length > 0) {
        bookendFrameIdx = isFirstPage ? pageFrames[0] : pageFrames[pageFrames.length - 1];
    }
    batchState.unloadedThumbnails.clear();
    batchState.currentPageLoaded = 0;
    batchState.currentPageTotal = isBookend ? 1 : pageFrames.length;

    // Auto-size grid to fit pageFrames.length thumbnails
    const gridRect = grid.getBoundingClientRect();
    const gridW = gridRect.width - 30; // subtract padding (15px each side)
    const gridH = gridRect.height - 30;
    const gap = 8;
    const count = pageFrames.length || 1;
    // Find fewest columns where all thumbnails fit in the grid height
    let bestCols = 12;
    let bestThumbW = 150;
    for (let cols = 4; cols <= 20; cols++) {
        const rows = Math.ceil(count / cols);
        const tw = (gridW - gap * (cols - 1)) / cols;
        const th = tw * 0.75; // 4:3 aspect
        const totalH = rows * th + (rows - 1) * gap;
        if (totalH <= gridH && tw >= 60) {
            bestCols = cols;
            bestThumbW = Math.floor(tw);
            break; // take the FEWEST columns that fit (largest thumbnails)
        }
    }
    grid.style.gridTemplateColumns = `repeat(${bestCols}, 1fr)`;
    // Scale thumbnail request width: larger thumbnails get higher resolution
    const thumbRequestWidth = bestThumbW <= 150 ? 150 : Math.min(bestThumbW * 2, 600);
    const thumbQuality = bestThumbW <= 150 ? 60 : 75;
    batchState.thumbRequestWidth = thumbRequestWidth;
    batchState.thumbQuality = thumbQuality;

    // Create thumbnails
    pageFrames.forEach(frameIdx => {
        const thumb = document.createElement('div');
        thumb.className = 'batch-thumbnail';
        thumb.dataset.frameIdx = frameIdx;

        // Add image — use prefetch cache if available
        const img = document.createElement('img');
        const shouldLoad = !isBookend || frameIdx === bookendFrameIdx;
        const cachedImg = batchState.prefetchCache.get(frameIdx);
        if (cachedImg && cachedImg.complete && cachedImg.naturalWidth > 0) {
            img.src = cachedImg.src;
            batchState.currentPageLoaded++;
        } else if (shouldLoad) {
            img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${thumbRequestWidth}&quality=${thumbQuality}`;
        } else {
            img.style.display = 'none';
            thumb.style.background = '#2a2a2a';
            batchState.unloadedThumbnails.add(frameIdx);
        }
        img.alt = `Frame ${frameIdx}`;
        img.onload = () => {
            batchState.currentPageLoaded++;
            if (!batchState.bookendPending && batchState.currentPageLoaded >= batchState.currentPageTotal) {
                prefetchNextPage();
            }
        };
        img.onerror = () => {
            img.style.display = 'none';
            thumb.style.background = '#333';
            label.textContent = `#${frameIdx} (err)`;
            batchState.currentPageLoaded++;
            if (!batchState.bookendPending && batchState.currentPageLoaded >= batchState.currentPageTotal) {
                prefetchNextPage();
            }
        };
        thumb.appendChild(img);

        // Add frame label
        const label = document.createElement('div');
        label.className = 'frame-label';
        label.textContent = `#${frameIdx}`;
        thumb.appendChild(label);

        // Add phase color bar if labeled
        const phaseData = state.phaseLabels[frameIdx];
        if (phaseData) {
            const coarse = phaseData.coarse || phaseData;
            if (coarse && PHASE_COLORS[coarse]) {
                const bar = document.createElement('div');
                bar.className = 'phase-bar';
                bar.style.background = PHASE_COLORS[coarse].bg;
                thumb.appendChild(bar);
            }
        }

        // Apply visibility classes
        applyThumbnailClasses(thumb, frameIdx);

        // Add click handler
        thumb.addEventListener('click', (e) => handleThumbnailClick(e, frameIdx));
        thumb.addEventListener('dblclick', (e) => handleThumbnailDblClick(e, frameIdx));

        grid.appendChild(thumb);
    });

    batchState.bookendPending = isBookend && batchState.unloadedThumbnails.size > 0;

    // If all images were from cache, trigger prefetch immediately (but not on bookend pages)
    if (!isBookend && batchState.currentPageLoaded >= batchState.currentPageTotal) {
        prefetchNextPage();
    }
}

/**
 * Prefetch next page's thumbnails in the background.
 * Creates hidden Image objects that browser caches, so next page loads instantly.
 * Only prefetches one page ahead to limit memory usage.
 */
function prefetchNextPage() {
    const total = batchState.allFrameIndices.length;
    const totalPages = Math.ceil(total / batchState.pageSize);
    const nextPage = batchState.currentPage + 1;

    // Nothing to prefetch if we're on the last page or already prefetched this page
    if (nextPage >= totalPages || nextPage === batchState.prefetchPage) return;

    // Don't prefetch while on bookend pages (they have unloaded thumbnails competing for bandwidth)
    if (batchState.bookendPending) return;

    // Clear old prefetch cache to limit memory (only keep one page ahead)
    batchState.prefetchCache.clear();
    batchState.prefetchPage = nextPage;
    batchState.prefetchInFlight = 0;

    const startIdx = nextPage * batchState.pageSize;
    const endIdx = Math.min(startIdx + batchState.pageSize, total);
    const nextFrames = batchState.allFrameIndices.slice(startIdx, endIdx);
    const thumbW = batchState.thumbRequestWidth || 150;
    const thumbQ = batchState.thumbQuality || 60;

    // Stagger requests slightly to avoid overwhelming the server
    nextFrames.forEach((frameIdx, i) => {
        setTimeout(() => {
            // Abort if page changed while we were queuing
            if (batchState.prefetchPage !== nextPage) return;

            const img = new Image();
            img.onload = () => {
                batchState.prefetchCache.set(frameIdx, img);
            };
            img.onerror = () => {
                // Don't cache failed loads
            };
            img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${thumbW}&quality=${thumbQ}`;
        }, i * 20); // 20ms stagger between requests
    });
}

/**
 * Apply CSS classes to thumbnail based on visibility status.
 */
function applyThumbnailClasses(thumb, frameIdx) {
    const vis = batchState.frameVisibility[String(frameIdx)];
    const status = state.frameStatus[String(frameIdx)];

    // Clear existing classes
    thumb.classList.remove('selected', 't1-out', 't2-out', 'both-out', 'complete', 'broken', 'skipped', 'exclude');

    // Selected
    if (batchState.selectedFrames.has(frameIdx)) {
        thumb.classList.add('selected');
    }

    // Broken takes priority over everything
    if (status === 'broken') {
        thumb.classList.add('broken');
        return;
    }

    // Skipped frames
    if (status === 'skipped') {
        thumb.classList.add('skipped');
        return;
    }

    // Visibility states
    if (vis) {
        const t1AllOut = vis.t1 && Object.values(vis.t1).every(v => v === -1);
        const t2AllOut = vis.t2 && Object.values(vis.t2).every(v => v === -1);

        if (t1AllOut && t2AllOut) {
            thumb.classList.add('both-out');
        } else if (t1AllOut) {
            thumb.classList.add('t1-out');
        } else if (t2AllOut) {
            thumb.classList.add('t2-out');
        }
    }

    // Negative fallback: show "OUT" badge even if visibility data isn't loaded
    if (!thumb.classList.contains('both-out') && status === 'negative') {
        thumb.classList.add('both-out');
    }

    // Completed status
    if (status === 'completed' && !thumb.classList.contains('both-out')) {
        thumb.classList.add('complete');
    }

    // Exclude badge (can co-exist with other states)
    if (batchState.frameExclude && batchState.frameExclude[String(frameIdx)]) {
        thumb.classList.add('exclude');
    }

    // Peg copy mode highlighting
    thumb.classList.remove('peg-copy-source', 'peg-copy-target');
    if (batchState.pegCopyMode) {
        if (frameIdx === batchState.pegCopySource) {
            thumb.classList.add('peg-copy-source');
        }
        if (batchState.pegCopyTargets.has(frameIdx)) {
            thumb.classList.add('peg-copy-target');
        }
    }
}

/**
 * Handle thumbnail click with modifier keys.
 */
function handleThumbnailClick(e, frameIdx) {
    e.preventDefault();
    e.stopPropagation();

    // Peg copy mode: select source or targets
    if (batchState.pegCopyMode === 'select_source') {
        batchState.pegCopySource = frameIdx;
        batchState.pegCopyMode = 'select_targets';
        batchState.pegCopyTargets.clear();
        updateThumbnailClasses();
        showToast(`Source: frame ${frameIdx}. Click/Shift+Click targets, Enter to confirm, Esc to cancel`);
        return;
    }
    if (batchState.pegCopyMode === 'select_targets') {
        if (frameIdx === batchState.pegCopySource) return; // Can't target source
        if (e.shiftKey && batchState.lastClickedFrame !== null) {
            // Range select targets
            const start = Math.min(batchState.lastClickedFrame, frameIdx);
            const end = Math.max(batchState.lastClickedFrame, frameIdx);
            const pageFrames = batchState.allFrameIndices;
            const startPage = pageFrames.indexOf(start >= 0 ? start : pageFrames[0]);
            for (const fi of pageFrames) {
                if (fi >= start && fi <= end && fi !== batchState.pegCopySource) {
                    batchState.pegCopyTargets.add(fi);
                }
            }
        } else if (e.ctrlKey || e.metaKey) {
            // Toggle target
            if (batchState.pegCopyTargets.has(frameIdx)) {
                batchState.pegCopyTargets.delete(frameIdx);
            } else {
                batchState.pegCopyTargets.add(frameIdx);
            }
        } else {
            // Add to targets (don't clear existing)
            batchState.pegCopyTargets.add(frameIdx);
        }
        batchState.lastClickedFrame = frameIdx;
        updateThumbnailClasses();
        showToast(`${batchState.pegCopyTargets.size} target frames selected. Enter to confirm.`);
        return;
    }

    if (e.ctrlKey || e.metaKey) {
        // Toggle selection
        if (batchState.selectedFrames.has(frameIdx)) {
            batchState.selectedFrames.delete(frameIdx);
        } else {
            batchState.selectedFrames.add(frameIdx);
        }
        batchState.lastClickedFrame = frameIdx;
    } else if (e.shiftKey && batchState.lastClickedFrame !== null) {
        // Range selection
        selectFrameRange(batchState.lastClickedFrame, frameIdx);
    } else {
        // Single selection (clear others)
        batchState.selectedFrames.clear();
        batchState.selectedFrames.add(frameIdx);
        batchState.lastClickedFrame = frameIdx;
    }

    updateBatchSelectionUI();
    updateThumbnailClasses();
}

/**
 * Handle double-click to navigate to frame.
 */
function handleThumbnailDblClick(e, frameIdx) {
    e.preventDefault();
    e.stopPropagation();

    // Close batch mode and navigate to the frame
    closeBatchMode();
    navigateToFrame(frameIdx).catch(err => console.error('Navigate failed:', err));
}

/**
 * Select a range of frames from start to end (inclusive).
 */
function selectFrameRange(startFrame, endFrame) {
    const startIdx = batchState.allFrameIndices.indexOf(startFrame);
    const endIdx = batchState.allFrameIndices.indexOf(endFrame);

    if (startIdx === -1 || endIdx === -1) return;

    const minIdx = Math.min(startIdx, endIdx);
    const maxIdx = Math.max(startIdx, endIdx);

    for (let i = minIdx; i <= maxIdx; i++) {
        batchState.selectedFrames.add(batchState.allFrameIndices[i]);
    }
}

/**
 * Update thumbnail classes after selection change.
 */
function updateThumbnailClasses() {
    const thumbnails = batchElements.batchGrid.querySelectorAll('.batch-thumbnail');
    thumbnails.forEach(thumb => {
        const frameIdx = parseInt(thumb.dataset.frameIdx);
        applyThumbnailClasses(thumb, frameIdx);
    });
}

/**
 * Update batch selection count UI.
 */
function updateBatchSelectionUI() {
    const count = batchState.selectedFrames.size;
    batchElements.batchCount.textContent = `${count} selected`;

    // Enable/disable action buttons
    const hasSelection = count > 0;
    batchElements.batchT1Out.disabled = !hasSelection;
    batchElements.batchT2Out.disabled = !hasSelection;
    batchElements.batchBothOut.disabled = !hasSelection;
    if (batchElements.batchBroken) batchElements.batchBroken.disabled = !hasSelection;
    if (batchElements.batchExclude) batchElements.batchExclude.disabled = !hasSelection;
}

/**
 * Refresh batch thumbnails by cache-busting image URLs.
 * If frames are selected, refresh only those; otherwise refresh all on the page.
 */
function refreshBatchThumbnails() {
    const thumbnails = batchElements.batchGrid.querySelectorAll('.batch-thumbnail');
    const hasSelection = batchState.selectedFrames.size > 0;

    // Bookend mode: first press loads unloaded thumbnails
    if (batchState.bookendPending && !hasSelection) {
        let count = 0;
        thumbnails.forEach(thumb => {
            const frameIdx = parseInt(thumb.dataset.frameIdx);
            if (batchState.unloadedThumbnails.has(frameIdx)) {
                const img = thumb.querySelector('img');
                if (img) {
                    img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${batchState.thumbRequestWidth || 150}&quality=${batchState.thumbQuality || 60}`;
                    img.style.display = '';
                    thumb.style.background = '';
                }
                batchState.unloadedThumbnails.delete(frameIdx);
                count++;
            }
        });
        batchState.bookendPending = false;
        showToast(`Loaded ${count} thumbnail${count !== 1 ? 's' : ''}`);
        return;
    }

    // Normal cache-bust refresh
    const timestamp = Date.now();
    let count = 0;

    thumbnails.forEach(thumb => {
        const frameIdx = parseInt(thumb.dataset.frameIdx);
        if (!hasSelection || batchState.selectedFrames.has(frameIdx)) {
            const img = thumb.querySelector('img');
            if (img) {
                // Strip old cache-bust param, add new one
                const base = img.src.replace(/&t=\d+/, '');
                img.src = base + '&t=' + timestamp;
                img.style.display = '';  // un-hide if previously errored
                // Reset error state
                const label = thumb.querySelector('.frame-label');
                if (label) label.textContent = `#${frameIdx}`;
                thumb.style.background = '';
            }
            count++;
        }
    });

    showToast(`Refreshed ${count} thumbnail${count !== 1 ? 's' : ''}`);
}

/**
 * Jump to the page containing a specific frame index.
 * @param {number} frameIdx - The frame index to navigate to.
 */
function goToBatchFrame(frameIdx) {
    const idx = batchState.allFrameIndices.indexOf(frameIdx);
    if (idx === -1) {
        // Try closest frame
        let closest = batchState.allFrameIndices[0];
        for (const f of batchState.allFrameIndices) {
            if (Math.abs(f - frameIdx) < Math.abs(closest - frameIdx)) closest = f;
        }
        const closestIdx = batchState.allFrameIndices.indexOf(closest);
        batchState.currentPage = Math.floor(closestIdx / batchState.pageSize);
        renderBatchPage();
        showToast(`Frame #${frameIdx} not found — jumped to nearest #${closest}`);
        return;
    }
    batchState.currentPage = Math.floor(idx / batchState.pageSize);
    renderBatchPage();
}

/**
 * Mark selected frames with tool out status.
 * @param {number} toolNum - Tool number (1, 2, or 0 for both)
 */
async function batchMarkToolOut(toolNum) {
    if (batchState.selectedFrames.size === 0) {
        showToast('No frames selected', true);
        return;
    }

    const frameIndices = Array.from(batchState.selectedFrames);

    // Build visibility update
    const updates = {};
    if (toolNum === 0 || toolNum === 1) {
        updates.tool1_visibility = {
            mask: -1, lines: -1, joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
        };
    }
    if (toolNum === 0 || toolNum === 2) {
        updates.tool2_visibility = {
            mask: -1, lines: -1, joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
        };
    }

    try {
        const result = await api(`/frames/${state.trialId}/batch`, {
            method: 'POST',
            body: JSON.stringify({
                frame_indices: frameIndices,
                updates: updates
            })
        });

        if (result.success) {
            const label = toolNum === 0 ? 'Both tools' : `Tool ${toolNum}`;
            showToast(`${label} marked Out on ${result.updated_count} frames`);

            // Update local visibility cache
            frameIndices.forEach(idx => {
                if (!batchState.frameVisibility[String(idx)]) {
                    batchState.frameVisibility[String(idx)] = { t1: {}, t2: {} };
                }
                if (toolNum === 0 || toolNum === 1) {
                    batchState.frameVisibility[String(idx)].t1 = {
                        mask: -1, lines: -1, joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
                    };
                }
                if (toolNum === 0 || toolNum === 2) {
                    batchState.frameVisibility[String(idx)].t2 = {
                        mask: -1, lines: -1, joint: -1, ee_tip: -1, ee_left: -1, ee_right: -1
                    };
                }
            });

            // Update main frameStatus so progress/dropdown reflect changes immediately
            frameIndices.forEach(idx => {
                if (toolNum === 0) {
                    // Both tools out = negative frame (no tools visible)
                    state.frameStatus[String(idx)] = 'negative';
                } else {
                    // Single tool out: negative if the other tool is also out
                    const vis = batchState.frameVisibility[String(idx)];
                    if (vis) {
                        const otherTool = toolNum === 1 ? vis.t2 : vis.t1;
                        const otherAllOut = otherTool && Object.values(otherTool).every(v => v === -1);
                        if (otherAllOut) {
                            state.frameStatus[String(idx)] = 'negative';
                        }
                    }
                }
            });
            recalculateProgress();
            updateFrameJumpDropdown();

            // Clear selection and refresh
            batchState.selectedFrames.clear();
            updateBatchSelectionUI();
            updateThumbnailClasses();

            // Also refresh from backend for consistency
            await refreshProgressData();
        } else {
            showToast('Batch update failed', true);
        }
    } catch (e) {
        console.error('Batch update error:', e);
        showToast('Batch update failed', true);
    }
}

/**
 * Mark selected frames as broken (move to broken/ folder).
 * This is destructive and cannot be undone.
 */
async function batchMarkBroken() {
    if (batchState.selectedFrames.size === 0) {
        showToast('No frames selected', true);
        return;
    }

    const count = batchState.selectedFrames.size;
    if (!confirm(`Move ${count} frame${count !== 1 ? 's' : ''} to broken/? This CANNOT be undone.`)) {
        return;
    }

    const frameIndices = Array.from(batchState.selectedFrames);

    try {
        const result = await api(`/frames/${state.trialId}/batch-broken`, {
            method: 'POST',
            body: JSON.stringify({ frame_indices: frameIndices })
        });

        if (result.success) {
            showToast(`Moved ${result.moved_count} frame${result.moved_count !== 1 ? 's' : ''} to broken/`);

            // Update local state
            frameIndices.forEach(idx => {
                state.frameStatus[String(idx)] = 'broken';
            });

            // Remove broken frames from ALL frame lists so they never reappear
            const brokenSet = new Set(frameIndices);
            batchState.allFrameIndices = batchState.allFrameIndices.filter(
                idx => !brokenSet.has(idx)
            );
            state.allFrames = state.allFrames.filter(
                idx => !brokenSet.has(idx)
            );
            state.sampledFrames = state.sampledFrames.filter(
                idx => !brokenSet.has(idx)
            );

            // Clear selection and refresh
            batchState.selectedFrames.clear();
            recalculateProgress();
            updateFrameJumpDropdown();
            updateBatchSelectionUI();
            renderBatchPage();

            // Also refresh from backend for consistency
            await refreshProgressData();

            if (result.failed.length > 0) {
                showToast(`${result.failed.length} frame(s) failed to move`, true);
            }
        } else {
            showToast('Failed to move frames to broken/', true);
        }
    } catch (e) {
        console.error('Batch broken error:', e);
        showToast('Failed to move frames to broken/', true);
    }
}

/**
 * Mark selected frames as excluded (training-only, not for validation).
 */
async function batchMarkExclude() {
    if (batchState.selectedFrames.size === 0) {
        showToast('No frames selected', true);
        return;
    }

    const frameIndices = Array.from(batchState.selectedFrames);

    try {
        const result = await api(`/frames/${state.trialId}/batch`, {
            method: 'POST',
            body: JSON.stringify({
                frame_indices: frameIndices,
                updates: { exclude: true }
            })
        });

        if (result.success) {
            showToast(`Marked ${result.updated_count} frame${result.updated_count !== 1 ? 's' : ''} as training-only`);

            // Update local exclude cache
            frameIndices.forEach(idx => {
                batchState.frameExclude[String(idx)] = true;
            });

            // Clear selection and refresh thumbnails
            batchState.selectedFrames.clear();
            updateBatchSelectionUI();
            updateThumbnailClasses();
        } else {
            showToast('Failed to mark frames as training-only', true);
        }
    } catch (e) {
        console.error('Batch exclude error:', e);
        showToast('Failed to mark frames as training-only', true);
    }
}

/**
 * Refresh progress data and update dropdowns.
 */
async function refreshProgressData() {
    showLoading('Refreshing progress...');
    try {
        // Reload datasets with progress to update percentages
        await loadDatasetsWithProgress();

        // Update trial dropdown if a dataset is selected
        if (state.dataset && state.trialId) {
            const datasetData = state.datasetsProgress.find(d => d.name === state.dataset);
            if (datasetData) {
                // Repopulate trial select while preserving current selection
                const currentTrialId = state.trialId;
                elements.trialSelect.innerHTML = '<option value="">Select Trial...</option>';
                datasetData.trials.forEach(trial => {
                    const opt = document.createElement('option');
                    opt.value = trial.trial_id;
                    applyTrialProgress(opt, trial.trial_name, trial);
                    if (trial.trial_id === currentTrialId) {
                        opt.selected = true;
                    }
                    elements.trialSelect.appendChild(opt);
                });
            }
        }

        // Reload frame status for the current trial
        if (state.trialId) {
            await fetchFrameStatus(state.trialId);
        }
    } catch (e) {
        console.error('Failed to refresh progress:', e);
    } finally {
        hideLoading();
    }
}

/**
 * Handle drag selection start on batch grid.
 */
function handleBatchDragStart(e) {
    if (e.button !== 0) return;

    // Only start drag if clicking on grid background, not thumbnail
    if (e.target.closest('.batch-thumbnail')) return;

    batchState.isDragging = true;
    batchState.dragStartX = e.clientX;
    batchState.dragStartY = e.clientY;

    // Create drag rectangle
    batchState.dragRect = document.createElement('div');
    batchState.dragRect.className = 'batch-drag-rect';
    document.body.appendChild(batchState.dragRect);
}

/**
 * Handle drag selection move.
 */
function handleBatchDragMove(e) {
    if (!batchState.isDragging || !batchState.dragRect) return;

    const x1 = Math.min(batchState.dragStartX, e.clientX);
    const y1 = Math.min(batchState.dragStartY, e.clientY);
    const x2 = Math.max(batchState.dragStartX, e.clientX);
    const y2 = Math.max(batchState.dragStartY, e.clientY);

    batchState.dragRect.style.left = x1 + 'px';
    batchState.dragRect.style.top = y1 + 'px';
    batchState.dragRect.style.width = (x2 - x1) + 'px';
    batchState.dragRect.style.height = (y2 - y1) + 'px';
}

/**
 * Handle drag selection end.
 */
function handleBatchDragEnd(e) {
    if (!batchState.isDragging) return;

    batchState.isDragging = false;

    if (batchState.dragRect) {
        const dragRect = batchState.dragRect.getBoundingClientRect();

        // Check if drag was significant
        if (dragRect.width > 5 && dragRect.height > 5) {
            // Clear selection unless holding Ctrl
            if (!e.ctrlKey && !e.metaKey) {
                batchState.selectedFrames.clear();
            }

            // Find all thumbnails intersecting the drag rectangle
            const thumbnails = batchElements.batchGrid.querySelectorAll('.batch-thumbnail');
            thumbnails.forEach(thumb => {
                const thumbRect = thumb.getBoundingClientRect();

                // Check intersection
                if (!(thumbRect.right < dragRect.left ||
                      thumbRect.left > dragRect.right ||
                      thumbRect.bottom < dragRect.top ||
                      thumbRect.top > dragRect.bottom)) {
                    const frameIdx = parseInt(thumb.dataset.frameIdx);
                    batchState.selectedFrames.add(frameIdx);
                }
            });

            updateBatchSelectionUI();
            updateThumbnailClasses();
        }

        batchState.dragRect.remove();
        batchState.dragRect = null;
    }
}

/**
 * Handle keyboard shortcuts in batch mode.
 */
function handleBatchKeydown(e) {
    if (!batchState.active) return false;

    // Don't intercept keys when typing in an input or select
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return false;

    const key = e.key.toLowerCase();

    // Peg copy mode: Enter to confirm, Escape to cancel
    if (batchState.pegCopyMode) {
        if (key === 'enter' && batchState.pegCopyMode === 'select_targets' && batchState.pegCopyTargets.size > 0) {
            executeBatchCopyPegs();
            return true;
        }
        if (key === 'escape') {
            batchState.pegCopyMode = null;
            batchState.pegCopySource = null;
            batchState.pegCopyTargets.clear();
            document.getElementById('batch-copy-pegs')?.classList.remove('active');
            updateThumbnailClasses();
            showToast('Copy Pegs cancelled');
            return true;
        }
    }

    switch (key) {
        case 'escape':
            closeBatchMode();
            return true;
        case '1':
            batchMarkToolOut(1);
            return true;
        case '2':
            batchMarkToolOut(2);
            return true;
        case 'a':
            if (!e.ctrlKey && !e.metaKey) {
                // Select all on current page
                const startIdx = batchState.currentPage * batchState.pageSize;
                const endIdx = Math.min(startIdx + batchState.pageSize, batchState.allFrameIndices.length);
                for (let i = startIdx; i < endIdx; i++) {
                    batchState.selectedFrames.add(batchState.allFrameIndices[i]);
                }
                updateBatchSelectionUI();
                updateThumbnailClasses();
                return true;
            }
            break;
        case 'd':
            // Deselect all
            batchState.selectedFrames.clear();
            batchState.lastClickedFrame = null;
            updateBatchSelectionUI();
            updateThumbnailClasses();
            return true;
        case 'b':
            batchMarkBroken();
            return true;
        case 'x':
            batchMarkExclude();
            return true;
        case 'r':
            refreshBatchThumbnails();
            return true;
        case 'g':
            batchElements.batchGotoFrame.focus();
            return true;
        case 'arrowleft':
            if (batchState.currentPage > 0) {
                batchState.currentPage--;
                renderBatchPage();
            }
            return true;
        case 'arrowright':
            const maxPage = Math.ceil(batchState.allFrameIndices.length / batchState.pageSize) - 1;
            if (batchState.currentPage < maxPage) {
                batchState.currentPage++;
                renderBatchPage();
            }
            return true;
        case 'home':
            batchState.currentPage = 0;
            renderBatchPage();
            return true;
        case 'end':
            batchState.currentPage = Math.ceil(batchState.allFrameIndices.length / batchState.pageSize) - 1;
            renderBatchPage();
            return true;
        case 'n':
            batchElements.batchNextTrial.click();
            return true;
    }

    // Phase keybinds: Z=idle, T=place are unambiguous; others conflict with batch actions
    // Users can click the inline phase buttons for conflicting keys (D=grasp, X=reach, R=transfer, G=return)
    if (key === 'z') {
        batchAssignPhase('idle');
        return true;
    }
    if (key === 't') {
        batchAssignPhase('place');
        return true;
    }

    return false;
}

// Batch mode event listeners
if (batchElements.batchModeBtn) {
    batchElements.batchModeBtn.addEventListener('click', toggleBatchMode);
}
if (batchElements.batchClose) {
    batchElements.batchClose.addEventListener('click', closeBatchMode);
}
if (batchElements.batchPrev) {
    batchElements.batchPrev.addEventListener('click', () => {
        if (batchState.currentPage > 0) {
            batchState.currentPage--;
            renderBatchPage();
        }
    });
}
if (batchElements.batchNext) {
    batchElements.batchNext.addEventListener('click', () => {
        const maxPage = Math.ceil(batchState.allFrameIndices.length / batchState.pageSize) - 1;
        if (batchState.currentPage < maxPage) {
            batchState.currentPage++;
            renderBatchPage();
        }
    });
}
if (batchElements.batchFirst) {
    batchElements.batchFirst.addEventListener('click', () => {
        batchState.currentPage = 0;
        renderBatchPage();
    });
}
if (batchElements.batchLast) {
    batchElements.batchLast.addEventListener('click', () => {
        batchState.currentPage = Math.ceil(batchState.allFrameIndices.length / batchState.pageSize) - 1;
        renderBatchPage();
    });
}
if (batchElements.batchNextTrial) {
    batchElements.batchNextTrial.addEventListener('click', async () => {
        await switchTrial(+1);
        if (state.trialId) {
            await openBatchMode();
        }
    });
}
if (batchElements.batchPageSelect) {
    batchElements.batchPageSelect.addEventListener('change', (e) => {
        batchState.currentPage = parseInt(e.target.value);
        renderBatchPage();
    });
}
if (batchElements.batchGotoFrame) {
    batchElements.batchGotoFrame.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            const val = parseInt(batchElements.batchGotoFrame.value);
            if (!isNaN(val)) {
                goToBatchFrame(val);
                batchElements.batchGotoFrame.value = '';
                batchElements.batchGotoFrame.blur();
            }
        }
    });
}
if (batchElements.batchT1Out) {
    batchElements.batchT1Out.addEventListener('click', () => batchMarkToolOut(1));
}
if (batchElements.batchT2Out) {
    batchElements.batchT2Out.addEventListener('click', () => batchMarkToolOut(2));
}
if (batchElements.batchBothOut) {
    batchElements.batchBothOut.addEventListener('click', () => batchMarkToolOut(0));
}
if (batchElements.batchClearSelection) {
    batchElements.batchClearSelection.addEventListener('click', () => {
        batchState.selectedFrames.clear();
        batchState.lastClickedFrame = null;
        updateBatchSelectionUI();
        updateThumbnailClasses();
    });
}
if (batchElements.batchRefresh) {
    batchElements.batchRefresh.addEventListener('click', refreshBatchThumbnails);
}
if (batchElements.batchBroken) {
    batchElements.batchBroken.addEventListener('click', batchMarkBroken);
}
if (batchElements.batchExclude) {
    batchElements.batchExclude.addEventListener('click', batchMarkExclude);
}
if (batchElements.batchShowAll) {
    batchElements.batchShowAll.addEventListener('change', async (e) => {
        batchState.showAllFrames = e.target.checked;
        await updateBatchFrameList();
        renderBatchPage();
    });
}

// Batch Copy Pegs button
const batchCopyPegsBtn = document.getElementById('batch-copy-pegs');
if (batchCopyPegsBtn) {
    batchCopyPegsBtn.addEventListener('click', () => {
        if (batchState.pegCopyMode) {
            // Cancel
            batchState.pegCopyMode = null;
            batchState.pegCopySource = null;
            batchState.pegCopyTargets.clear();
            batchCopyPegsBtn.classList.remove('active');
            updateThumbnailClasses();
            showToast('Copy Pegs cancelled');
        } else {
            batchState.pegCopyMode = 'select_source';
            batchState.pegCopySource = null;
            batchState.pegCopyTargets.clear();
            batchCopyPegsBtn.classList.add('active');
            showToast('Click source frame to copy pegs from');
        }
    });
}

// Drag selection events on batch grid
if (batchElements.batchGrid) {
    batchElements.batchGrid.addEventListener('mousedown', handleBatchDragStart);
}
document.addEventListener('mousemove', handleBatchDragMove);
document.addEventListener('mouseup', handleBatchDragEnd);

// ============================================================================
// Batch Phase Assignment
// ============================================================================

// Phase buttons are now inline in batch actions bar (no toggle needed)

/**
 * Assign phase to all selected frames in batch mode via bulk API.
 */
async function executeBatchCopyPegs() {
    if (!batchState.pegCopySource === null || batchState.pegCopyTargets.size === 0) return;
    const targetArray = [...batchState.pegCopyTargets].sort((a, b) => a - b);
    try {
        const resp = await fetch(`/api/frames/${state.trialId}/batch-copy-pegs`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                source_frame: batchState.pegCopySource,
                target_frames: targetArray,
            }),
        });
        const result = await resp.json();
        if (result.success) {
            showToast(`Copied pegs to ${result.count} frames`);
        } else {
            showToast('Copy failed', true);
        }
    } catch (err) {
        console.error('Batch copy pegs error:', err);
        showToast('Copy failed: ' + err.message, true);
    }
    // Reset state
    batchState.pegCopyMode = null;
    batchState.pegCopySource = null;
    batchState.pegCopyTargets.clear();
    document.getElementById('batch-copy-pegs')?.classList.remove('active');
    updateThumbnailClasses();
}

async function batchAssignPhase(coarsePhase) {
    if (batchState.selectedFrames.size === 0) {
        showToast('No frames selected', true);
        return;
    }

    const cycleInput = document.getElementById('batch-phase-cycle');
    const cycleIndex = cycleInput ? parseInt(cycleInput.value) || 0 : 0;

    const selectedArray = [...batchState.selectedFrames].sort((a, b) => a - b);
    // Mirror phase to both tools so the dual-tool data model stays consistent.
    const phaseData = {
        coarse: coarsePhase,
        tool1: coarsePhase,
        tool2: coarsePhase,
        fine: '',
        cycle_index: cycleIndex,
        active_tool: 0,
        events: [],
    };

    try {
        // Use bulk API for contiguous ranges, or individual saves
        await api(`/trials/${state.trialId}/phase_bulk`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                frames: selectedArray,
                phase_data: phaseData
            })
        });

        // Update local cache
        for (const frameIdx of selectedArray) {
            state.phaseLabels[frameIdx] = {
                coarse: coarsePhase,
                tool1: coarsePhase,
                tool2: coarsePhase,
                cycle_index: cycleIndex,
            };
        }

        showToast(`Phase "${coarsePhase}" set for ${selectedArray.length} frames`);
        renderBatchPage();  // Re-render to show phase bars on thumbnails
        drawPhaseStrip();
    } catch (e) {
        showToast('Failed to set phases: ' + e.message, true);
    }
}

// ============================================================================
// Phase & Peg Event Listeners
// ============================================================================

// Mode switcher buttons
document.querySelectorAll('#mode-switcher .mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        state.annotationMode = btn.dataset.mode;
        updateAnnotationModeUI();
        rerenderTrialOptionsForMode();
        updateDatasetDropdownPercentage();
        drawAllAnnotations();
    });
});

// Phase control buttons (main view)
document.querySelectorAll('#phase-controls .phase-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        setPhaseForCurrentFrame(btn.dataset.phase);
    });
});


// Inline batch phase buttons
document.querySelectorAll('.batch-phase-assign').forEach(btn => {
    btn.addEventListener('click', () => {
        batchAssignPhase(btn.dataset.phase);
    });
});

// Phase info guide button
const phaseInfoBtn = document.getElementById('phase-info-btn');
const phaseGuideOverlay = document.getElementById('phase-guide-overlay');
const phaseGuideCloseBtn = document.getElementById('phase-guide-close');
if (phaseInfoBtn && phaseGuideOverlay) {
    phaseInfoBtn.addEventListener('click', () => { phaseGuideOverlay.style.display = 'flex'; });
    phaseGuideCloseBtn?.addEventListener('click', () => { phaseGuideOverlay.style.display = 'none'; });
    phaseGuideOverlay.addEventListener('click', (e) => {
        if (e.target === phaseGuideOverlay) phaseGuideOverlay.style.display = 'none';
    });
}

// Phase strip click navigation
const phaseStripCanvas = document.getElementById('phase-strip-canvas');
if (phaseStripCanvas) {
    phaseStripCanvas.addEventListener('click', (e) => {
        const totalFrames = state.allFrames.length || state.sampledFrames.length;
        if (!totalFrames) return;
        const rect = phaseStripCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const targetFrame = Math.round((x / rect.width) * totalFrames);
        // Navigate to nearest sampled frame
        const nearest = state.sampledFrames.reduce((prev, curr) =>
            Math.abs(curr - targetFrame) < Math.abs(prev - targetFrame) ? curr : prev
        , state.sampledFrames[0]);
        if (nearest !== undefined) {
            const idx = state.sampledFrames.indexOf(nearest);
            if (idx >= 0) navigateToFrame(idx);
        }
    });
}

// Peg grid buttons
document.querySelectorAll('#peg-grid .peg-btn').forEach(btn => {
    btn.addEventListener('click', () => selectPeg(parseInt(btn.dataset.peg)));
});

// Peg state button clicks
document.querySelectorAll('.peg-state-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.peg-state-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        savePegDataFromUI();
    });
});

// Peg post button clicks
document.querySelectorAll('.peg-post-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.peg-post-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        savePegDataFromUI();
    });
});

// Peg drawing tool buttons
const pegBboxBtn = document.getElementById('peg-bbox-btn');
const pegMaskBtn = document.getElementById('peg-mask-btn');
const pegClearBtn = document.getElementById('peg-clear-btn');
const pegCopyPriorBtn = document.getElementById('peg-copy-prior-btn');

if (pegBboxBtn) {
    pegBboxBtn.addEventListener('click', () => {
        state.pegDrawingTool = state.pegDrawingTool === 'bbox' ? null : 'bbox';
        state.pegboardTool = null;
        state.pegBboxStart = null;
        state.tempPoints = [];
        pegBboxBtn.classList.toggle('active', state.pegDrawingTool === 'bbox');
        pegMaskBtn?.classList.remove('active');
        clearPegboardBtnActive();
        if (state.pegDrawingTool === 'bbox') showToast('Click two corners to draw bbox');
    });
}
if (pegMaskBtn) {
    pegMaskBtn.addEventListener('click', () => {
        state.pegDrawingTool = state.pegDrawingTool === 'mask' ? null : 'mask';
        state.pegboardTool = null;
        state.isDrawing = false;
        state.tempPoints = [];
        pegMaskBtn.classList.toggle('active', state.pegDrawingTool === 'mask');
        pegBboxBtn?.classList.remove('active');
        clearPegboardBtnActive();
        if (state.pegDrawingTool === 'mask') showToast('Click to add mask points, snap to first or Enter to finish');
    });
}
if (pegClearBtn) {
    pegClearBtn.addEventListener('click', () => {
        if (!state.currentAnnotation || !state.selectedPegId) return;
        const pegs = state.currentAnnotation.pegs || [];
        state.currentAnnotation.pegs = pegs.filter(p => p.id !== state.selectedPegId);
        state.annotationDirty = true;
        saveAnnotations();
        drawAllAnnotations();
        showToast(`Cleared Peg ${state.selectedPegId}`);
    });
}
if (pegCopyPriorBtn) {
    pegCopyPriorBtn.addEventListener('click', async () => {
        if (!state.selectedPegId || !state.currentAnnotation) return;
        try {
            const result = await api(`/frames/${state.trialId}/latest-peg-frame?before=${state.frameIdx}&data_type=pegs`);
            if (!result.annotation?.pegs?.length) {
                showToast('No prior peg data found', true);
                return;
            }
            const priorPegs = result.annotation.pegs;
            const priorPeg = priorPegs.find(p => p.id === state.selectedPegId);
            if (!priorPeg) { showToast(`No prior data for Peg ${state.selectedPegId}`, true); return; }
            if (!state.currentAnnotation.pegs) state.currentAnnotation.pegs = [];
            state.currentAnnotation.pegs = state.currentAnnotation.pegs.filter(p => p.id !== state.selectedPegId);
            state.currentAnnotation.pegs.push(JSON.parse(JSON.stringify(priorPeg)));
            state.annotationDirty = true;
            saveAnnotations();
            loadPegDataToUI(state.selectedPegId);
            drawAllAnnotations();
            showToast(`Copied Peg ${state.selectedPegId} from frame ${result.frame_idx}`);
        } catch (err) {
            showToast('Error finding prior peg data', true);
            console.error(err);
        }
    });
}

// Individual post button listeners (S1-S6, T1-T6)
document.querySelectorAll('.post-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const postType = btn.dataset.postType; // 'source' or 'target'
        const postIdx = parseInt(btn.dataset.postIdx);
        const label = postType === 'source' ? `S${postIdx + 1}` : `T${postIdx + 1}`;

        // Toggle off if same button clicked again
        if (state.pegboardTool === 'post' && state.activePostTarget?.type === postType && state.activePostTarget?.index === postIdx) {
            state.pegboardTool = null;
            state.activePostTarget = null;
            state.isDrawing = false;
            state.tempPoints = [];
            clearPegboardBtnActive();
            drawAllAnnotations();
            return;
        }

        // Activate post mask drawing for this specific post
        state.pegDrawingTool = null;
        state.pegKeypointIdx = null;
        state.pegEditMode = false;
        state.postMaskMode = null;
        state.isDrawing = false;
        state.tempPoints = [];
        clearPegboardBtnActive();
        state.pegboardTool = 'post';
        state.activePostTarget = {type: postType, index: postIdx};
        btn.classList.add('active');
        document.getElementById('peg-bbox-btn')?.classList.remove('active');
        document.getElementById('peg-mask-btn')?.classList.remove('active');
        document.querySelectorAll('.peg-kp-btn').forEach(b => b.classList.remove('active'));
        showToast(`Draw mask for ${label} — snap to first or Enter to finish`);
    });
});

// Pegboard button listeners
const pegboardOutlineBtn = document.getElementById('pegboard-outline-btn');
const pegboardCopyBtn = document.getElementById('pegboard-copy-btn');

if (pegboardOutlineBtn) {
    pegboardOutlineBtn.addEventListener('click', () => {
        if (state.pegboardTool === 'outline') {
            state.pegboardTool = null;
            state.isDrawing = false;
            state.tempPoints = [];
            clearPegboardBtnActive();
        } else {
            state.pegboardTool = 'outline';
            state.pegDrawingTool = null;
            state.isDrawing = false;
            state.tempPoints = [];
            clearPegboardBtnActive();
            pegboardOutlineBtn.classList.add('active');
            document.getElementById('peg-bbox-btn')?.classList.remove('active');
            document.getElementById('peg-mask-btn')?.classList.remove('active');
            showToast('Click to draw board outline polygon, snap to first or Enter to finish');
        }
    });
}
if (pegboardCopyBtn) {
    pegboardCopyBtn.addEventListener('click', async () => {
        if (!state.currentAnnotation) return;
        try {
            const result = await api(`/frames/${state.trialId}/latest-peg-frame?before=${state.frameIdx}&data_type=any`);
            if (!result.annotation) {
                showToast('No prior peg/pegboard data found', true);
                return;
            }
            const prior = result.annotation;
            // Copy pegs if present
            if (prior.pegs?.length > 0) {
                state.currentAnnotation.pegs = JSON.parse(JSON.stringify(prior.pegs));
            }
            // Copy pegboard if present
            if (prior.pegboard) {
                state.currentAnnotation.pegboard = JSON.parse(JSON.stringify(prior.pegboard));
            }
            state.annotationDirty = true;
            saveAnnotations();
            updatePegboardCounters();
            updatePostButtonStatus();
            updatePegStatus();
            drawAllAnnotations();
            showToast(`Copied all peg data from frame ${result.frame_idx}`);
        } catch (err) {
            showToast('Error finding prior peg data', true);
            console.error(err);
        }
    });
}

// Pegboard Clear All button
const pegboardClearBtn = document.getElementById('pegboard-clear-btn');
if (pegboardClearBtn) {
    pegboardClearBtn.addEventListener('click', () => {
        if (!state.currentAnnotation) return;
        state.currentAnnotation.pegboard = {
            source_posts: [], target_posts: [], board_mask: [],
            source_post_masks: [[], [], [], [], [], []], target_post_masks: [[], [], [], [], [], []],
            source_post_keypoints: [null, null, null, null, null, null],
            target_post_keypoints: [null, null, null, null, null, null],
        };
        state.currentAnnotation.pegs = [];
        state.annotationDirty = true;
        saveAnnotations();
        updatePegboardCounters();
        updatePostButtonStatus();
        updatePegStatus();
        drawAllAnnotations();
        showToast('Cleared all peg & pegboard annotations');
    });
}

// Pegboard Edit All button
const pegboardEditBtn = document.getElementById('pegboard-edit-btn');
if (pegboardEditBtn) {
    pegboardEditBtn.addEventListener('click', () => {
        if (state.pegEditMode) {
            state.pegEditMode = false;
            state.pegEditPhase = null;
            state.pegEditRect = null;
            state.pegEditSelectedItems = [];
            state.pegEditDragStart = null;
            pegboardEditBtn.classList.remove('active');
            drawAllAnnotations();
            showToast('Edit mode off');
        } else {
            state.pegDrawingTool = null;
            state.pegboardTool = null;
            state.pegKeypointIdx = null;
            clearPegboardBtnActive();
            // Set AFTER clearPegboardBtnActive (which resets pegEditMode)
            state.pegEditMode = true;
            state.pegEditPhase = 'selecting';
            state.pegEditRect = null;
            state.pegEditSelectedItems = [];
            pegboardEditBtn.classList.add('active');
            document.getElementById('peg-bbox-btn')?.classList.remove('active');
            document.getElementById('peg-mask-btn')?.classList.remove('active');
            document.querySelectorAll('.peg-kp-btn').forEach(b => b.classList.remove('active'));
            showToast('Edit All: draw rectangle to select annotations');
        }
    });
}

// Peg keypoint button listeners (KP1, KP2, KP3)
['peg-kp1-btn', 'peg-kp2-btn', 'peg-kp3-btn'].forEach((id, idx) => {
    const btn = document.getElementById(id);
    if (btn) {
        btn.addEventListener('click', () => {
            if (!state.selectedPegId) { showToast('Select a peg first', true); return; }
            if (state.pegKeypointIdx === idx) {
                selectPegKeypoint(null);
            } else {
                selectPegKeypoint(idx);
                showToast(`Click to place KP${idx + 1} for Peg ${state.selectedPegId}`);
            }
        });
    }
});

// Right-click on canvas: no longer finalizes peg mask / board outline
// (use snap-to-first click or Enter key instead — right-click is for panning)

// Batch preview button — open selected frame full-res in new tab
document.getElementById('batch-preview')?.addEventListener('click', () => {
    if (!state.trialId) { showToast('No trial loaded', true); return; }
    const selected = [...(batchState.selectedFrames || [])];
    if (selected.length === 1) {
        window.open(`/api/frames/${state.trialId}/${selected[0]}/image`, '_blank');
    } else if (selected.length > 1) {
        selected.forEach(idx => window.open(`/api/frames/${state.trialId}/${idx}/image`, '_blank'));
    } else {
        showToast('Select a frame to preview', true);
    }
});

// Redraw phase strip on window resize
window.addEventListener('resize', () => drawPhaseStrip());

// Configurable batch page size
const batchPageSizeInput = document.getElementById('batch-page-size');
if (batchPageSizeInput) {
    batchPageSizeInput.value = batchState.pageSize;
    batchPageSizeInput.addEventListener('change', () => {
        const val = parseInt(batchPageSizeInput.value);
        if (val >= 6 && val <= 500) {
            batchState.pageSize = val;
            batchState.currentPage = 0;
            batchState.prefetchCache.clear();
            batchState.prefetchPage = -1;
            renderBatchPage();
        }
    });
}

// ============================================================================
// Video Annotation Mode
// ============================================================================

const videoState = {
    active: false,
    allFrameIndices: [],
    currentIdx: 0,
    isPlaying: false,
    playInterval: null,
    playSpeed: 1,
    clips: [],               // [{start, end, phase, cycle}] sorted by start
    selectedClipIdx: null,
    markStart: null,          // pending clip start frame index (into allFrameIndices)
    preloadCache: new Map(),
    preloadAhead: 30,
    preloadBehind: 5,
    fullCacheLoading: false,  // true while bulk preload is in progress
    fullCacheAbort: false,    // set true to cancel bulk preload
    fullCacheLoaded: 0,       // count of loaded frames
    fullCacheTotal: 0,        // total frames to load
    cachedTrialId: null,      // which trial is cached
    timelineZoom: 1,
    thumbWidth: 600,
    thumbQuality: 75,
    dragState: null,          // null | {type: 'left'|'right'|'move', clipIdx, startX, origStart, origEnd}
    autoStartNext: true,      // auto-place mark-start at next frame after mark-end
    autoSaveOnEdit: true,     // auto-save after phase assign / cycle change
    timelinePan: null,        // null | {startX, startScrollLeft} for right-click panning
};

const videoElements = {
    view: document.getElementById('video-view'),
    frameImg: document.getElementById('video-frame-img'),
    frameNum: document.getElementById('video-frame-num'),
    frameTotal: document.getElementById('video-frame-total'),
    currentPhase: document.getElementById('video-current-phase'),
    markIndicator: document.getElementById('video-mark-indicator'),
    markFrame: document.getElementById('video-mark-frame'),
    speedDisplay: document.getElementById('video-speed-display'),
    coverage: document.getElementById('video-coverage'),
    slider: document.getElementById('video-slider'),
    speedSelect: document.getElementById('video-speed-select'),
    playPauseBtn: document.getElementById('video-play-pause'),
    timelineContainer: document.getElementById('video-timeline-container'),
    timelineCanvas: document.getElementById('video-timeline-canvas'),
    trialName: document.getElementById('video-trial-name'),
    cycleInput: document.getElementById('video-cycle'),
    cacheBar: document.getElementById('video-cache-bar'),
    cacheProgress: document.getElementById('video-cache-progress'),
    cacheText: document.getElementById('video-cache-text'),
};

/**
 * Open video annotation mode.
 */
async function openVideoMode() {
    if (!state.trialId) { showToast('Load a trial first', true); return; }

    // Load all frames if not already loaded
    if (state.allFrames.length === 0) {
        await loadAllFrames(state.trialId);
    }
    videoState.allFrameIndices = [...state.allFrames];
    if (videoState.allFrameIndices.length === 0) {
        showToast('No frames found', true);
        return;
    }

    videoState.active = true;
    videoState.currentIdx = 0;
    videoState.isPlaying = false;
    videoState.selectedClipIdx = null;
    videoState.markStart = null;
    videoState.fullCacheAbort = false;
    if (videoState.activeTool !== 1 && videoState.activeTool !== 2) videoState.activeTool = 'both';
    // Reflect active-tool state on the buttons
    document.querySelectorAll('.video-tool-btn').forEach(b => {
        const t = b.dataset.tool === 'both' ? 'both' : parseInt(b.dataset.tool);
        b.classList.toggle('active', t === videoState.activeTool);
    });
    // Only clear cache if switching trials
    if (videoState.cachedTrialId !== state.trialId) {
        videoState.preloadCache.clear();
        videoState.cachedTrialId = state.trialId;
    }

    // Ensure phase labels are loaded before reconstructing clips
    await loadPhaseSummary(state.trialId);
    videoReconstructClips();

    // Setup UI
    videoElements.trialName.textContent = state.trialId;
    videoElements.slider.max = videoState.allFrameIndices.length - 1;
    videoElements.slider.value = 0;
    videoElements.frameTotal.textContent = videoState.allFrameIndices.length;
    videoElements.view.style.display = 'flex';
    videoElements.markIndicator.style.display = 'none';

    // Size timeline canvas
    videoResizeTimeline();
    videoLoadFrame(0);

    // Start loading ALL frames into cache
    videoPreloadAll();
}

/**
 * Close video mode.
 */
function closeVideoMode() {
    videoStopPlayback();
    videoState.active = false;
    videoState.fullCacheAbort = true;
    videoState.fullCacheLoading = false;
    // Keep cache — don't clear it so reopening is instant
    videoElements.view.style.display = 'none';
    // Refresh main view
    drawPhaseStrip();
}

/**
 * Reconstruct clips from state.phaseLabels.
 */
function videoReconstructClips() {
    // Clip schema: {start, end, tool1, tool2, cycle}
    // A new clip starts when (tool1, tool2, cycle) changes.
    videoState.clips = [];
    const frames = videoState.allFrameIndices;
    if (frames.length === 0) return;

    let clipStart = null;
    let clipT1 = null;
    let clipT2 = null;
    let clipCycle = 0;

    for (let i = 0; i < frames.length; i++) {
        const frameIdx = frames[i];
        const label = state.phaseLabels[frameIdx];
        // Pull tool1/tool2 (preferred) or fall back to legacy single-tool `coarse`.
        let t1 = null, t2 = null, cycle = 0;
        if (label) {
            if (typeof label === 'string') {
                t1 = t2 = label;
            } else {
                t1 = label.tool1 || label.coarse || null;
                t2 = label.tool2 || label.coarse || null;
                cycle = label.cycle_index || 0;
            }
        }
        const hasLabel = !!(t1 || t2);

        if (hasLabel) {
            if (t1 === clipT1 && t2 === clipT2 && cycle === clipCycle) {
                // continue
            } else {
                if (clipStart !== null) {
                    videoState.clips.push({start: clipStart, end: i - 1, tool1: clipT1, tool2: clipT2, cycle: clipCycle});
                }
                clipStart = i;
                clipT1 = t1 || 'idle';
                clipT2 = t2 || 'idle';
                clipCycle = cycle;
            }
        } else if (clipStart !== null) {
            videoState.clips.push({start: clipStart, end: i - 1, tool1: clipT1, tool2: clipT2, cycle: clipCycle});
            clipStart = null;
            clipT1 = null;
            clipT2 = null;
        }
    }
    if (clipStart !== null) {
        videoState.clips.push({start: clipStart, end: frames.length - 1, tool1: clipT1, tool2: clipT2, cycle: clipCycle});
    }
}

/**
 * Load and display a frame by index into allFrameIndices.
 */
function videoLoadFrame(idx) {
    if (idx < 0 || idx >= videoState.allFrameIndices.length) return;
    videoState.currentIdx = idx;
    const frameIdx = videoState.allFrameIndices[idx];

    // Check preload cache
    const cached = videoState.preloadCache.get(frameIdx);
    if (cached && cached.complete && cached.naturalWidth > 0) {
        videoElements.frameImg.src = cached.src;
    } else {
        videoElements.frameImg.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${videoState.thumbWidth}&quality=${videoState.thumbQuality}`;
    }

    // Update UI: show timeline position (1-indexed) and the PNG file index alongside.
    videoElements.frameNum.textContent = `${idx + 1} (#${frameIdx})`;
    videoElements.slider.value = idx;

    // Show current phase (per-tool)
    const clip = videoGetClipAt(idx);
    if (clip) {
        const t1 = clip.tool1 || 'idle';
        const t2 = clip.tool2 || 'idle';
        videoElements.currentPhase.textContent = (t1 === t2)
            ? `${t1} (Cycle ${clip.cycle})`
            : `T1:${t1} / T2:${t2} (Cycle ${clip.cycle})`;
        const colorPhase = (t1 !== 'idle') ? t1 : t2;
        videoElements.currentPhase.style.color = PHASE_COLORS[colorPhase]?.bg || '#fff';
    } else {
        videoElements.currentPhase.textContent = '—';
        videoElements.currentPhase.style.color = '#888';
    }

    // Update coverage
    videoUpdateCoverage();

    // Render timeline
    videoRenderTimeline();
    if (videoState.timelineZoom > 1) videoScrollTimelineToCursor();

    // Preload ahead
    videoPreloadFrames(idx);
}

/**
 * Get clip at a given array index.
 */
function videoGetClipAt(idx) {
    for (const clip of videoState.clips) {
        if (idx >= clip.start && idx <= clip.end) return clip;
    }
    return null;
}

/**
 * Update coverage display.
 */
function videoUpdateCoverage() {
    const total = videoState.allFrameIndices.length;
    let covered = 0;
    for (const clip of videoState.clips) {
        covered += clip.end - clip.start + 1;
    }
    const pct = total > 0 ? (covered / total * 100).toFixed(1) : 0;
    videoElements.coverage.innerHTML = `Coverage: <strong>${pct}%</strong>`;
}

/**
 * Preload frames around current position.
 */
/**
 * Preload nearby frames (fast, for immediate navigation).
 * No eviction since full cache handles everything.
 */
function videoPreloadFrames(centerIdx) {
    const start = Math.max(0, centerIdx - videoState.preloadBehind);
    const end = Math.min(videoState.allFrameIndices.length - 1, centerIdx + videoState.preloadAhead);

    for (let i = start; i <= end; i++) {
        const frameIdx = videoState.allFrameIndices[i];
        if (!videoState.preloadCache.has(frameIdx)) {
            const img = new Image();
            img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${videoState.thumbWidth}&quality=${videoState.thumbQuality}`;
            videoState.preloadCache.set(frameIdx, img);
        }
    }
}

/**
 * Preload ALL frames for the trial into cache.
 * Uses batched concurrent requests (6 at a time) to balance speed and server load.
 * Shows progress bar. User can annotate while loading continues.
 */
function videoPreloadAll() {
    const frames = videoState.allFrameIndices;
    const total = frames.length;

    // Check if already fully cached
    let alreadyCached = 0;
    for (const frameIdx of frames) {
        const img = videoState.preloadCache.get(frameIdx);
        if (img && img.complete && img.naturalWidth > 0) alreadyCached++;
    }
    if (alreadyCached >= total) {
        // Already fully loaded (e.g. reopening same trial)
        videoElements.cacheBar.style.display = 'none';
        return;
    }

    videoState.fullCacheLoading = true;
    videoState.fullCacheAbort = false;
    videoState.fullCacheLoaded = alreadyCached;
    videoState.fullCacheTotal = total;

    // Show progress bar
    videoElements.cacheBar.style.display = '';
    videoUpdateCacheProgress();

    const CONCURRENCY = 6; // parallel requests
    let nextToLoad = 0;

    function loadNext() {
        if (videoState.fullCacheAbort) return;

        // Find next frame not yet cached
        while (nextToLoad < total) {
            const frameIdx = frames[nextToLoad];
            const existing = videoState.preloadCache.get(frameIdx);
            if (existing && existing.complete && existing.naturalWidth > 0) {
                nextToLoad++;
                continue;
            }
            break;
        }

        if (nextToLoad >= total) {
            // Check if all done
            checkAllDone();
            return;
        }

        const idx = nextToLoad;
        nextToLoad++;
        const frameIdx = frames[idx];

        const img = new Image();
        img.onload = () => {
            videoState.preloadCache.set(frameIdx, img);
            videoState.fullCacheLoaded++;
            videoUpdateCacheProgress();
            loadNext();
        };
        img.onerror = () => {
            videoState.fullCacheLoaded++;
            videoUpdateCacheProgress();
            loadNext();
        };
        img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${videoState.thumbWidth}&quality=${videoState.thumbQuality}`;
    }

    function checkAllDone() {
        if (videoState.fullCacheLoaded >= total) {
            videoState.fullCacheLoading = false;
            videoElements.cacheBar.style.display = 'none';
            showToast(`All ${total} frames cached — playback ready`);
        }
    }

    // Launch concurrent loaders
    for (let i = 0; i < CONCURRENCY; i++) {
        loadNext();
    }
}

function videoUpdateCacheProgress() {
    const pct = videoState.fullCacheTotal > 0
        ? (videoState.fullCacheLoaded / videoState.fullCacheTotal * 100).toFixed(1)
        : 0;
    videoElements.cacheProgress.style.width = pct + '%';
    videoElements.cacheText.textContent = `Loading frames: ${videoState.fullCacheLoaded} / ${videoState.fullCacheTotal} (${pct}%)`;
}

/**
 * Play/pause toggle.
 */
function videoTogglePlayback() {
    if (videoState.isPlaying) {
        videoStopPlayback();
    } else {
        videoStartPlayback();
    }
}

function videoStartPlayback() {
    videoState.isPlaying = true;
    videoElements.playPauseBtn.innerHTML = '&#x23F8;'; // pause icon
    videoElements.playPauseBtn.classList.add('playing');

    const fps = state.trialId.includes('6DOF') ? 26 : 13;
    const targetInterval = Math.max(16, Math.round(1000 / (fps * videoState.playSpeed)));

    function playNextFrame() {
        if (!videoState.isPlaying) return;
        if (videoState.currentIdx >= videoState.allFrameIndices.length - 1) {
            videoStopPlayback();
            return;
        }

        const nextIdx = videoState.currentIdx + 1;
        const frameIdx = videoState.allFrameIndices[nextIdx];
        const startTime = performance.now();

        // Check if next frame is already cached and loaded
        const cached = videoState.preloadCache.get(frameIdx);
        if (cached && cached.complete && cached.naturalWidth > 0) {
            // Cached — display immediately and schedule next
            videoLoadFrame(nextIdx);
            const elapsed = performance.now() - startTime;
            const delay = Math.max(1, targetInterval - elapsed);
            videoState.playInterval = setTimeout(playNextFrame, delay);
        } else {
            // Not cached — load and wait for it
            const img = new Image();
            img.onload = () => {
                if (!videoState.isPlaying) return;
                videoState.preloadCache.set(frameIdx, img);
                videoLoadFrame(nextIdx);
                const elapsed = performance.now() - startTime;
                const delay = Math.max(1, targetInterval - elapsed);
                videoState.playInterval = setTimeout(playNextFrame, delay);
            };
            img.onerror = () => {
                if (!videoState.isPlaying) return;
                // Skip errored frame
                videoState.currentIdx = nextIdx;
                videoState.playInterval = setTimeout(playNextFrame, targetInterval);
            };
            img.src = `/api/frames/${state.trialId}/${frameIdx}/thumbnail?width=${videoState.thumbWidth}&quality=${videoState.thumbQuality}`;
        }
    }

    playNextFrame();
}

function videoStopPlayback() {
    videoState.isPlaying = false;
    if (videoState.playInterval) {
        clearTimeout(videoState.playInterval);
        videoState.playInterval = null;
    }
    videoElements.playPauseBtn.innerHTML = '&#x25B6;'; // play icon
    videoElements.playPauseBtn.classList.remove('playing');
}

function videoSetSpeed(speed) {
    videoState.playSpeed = speed;
    videoElements.speedDisplay.textContent = speed + 'x';
    videoElements.speedSelect.value = speed;
    // Restart playback at new speed if playing
    if (videoState.isPlaying) {
        videoStopPlayback();
        videoStartPlayback();
    }
}

// ---- Timeline rendering ----

function videoResizeTimeline() {
    const container = videoElements.timelineContainer;
    const canvas = videoElements.timelineCanvas;
    const w = container.clientWidth * videoState.timelineZoom;
    const h = 60;
    canvas.width = Math.max(w, container.clientWidth);
    canvas.height = h;
    canvas.style.width = canvas.width + 'px';
}

function videoRenderTimeline() {
    const canvas = videoElements.timelineCanvas;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const total = videoState.allFrameIndices.length;
    if (total === 0) return;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, w, h);

    // Draw clips — two stacked rows: tool1 top, tool2 bottom.
    const top = 4;
    const bottom = h - 4;
    const rowH = (bottom - top) / 2;
    const row1Y = top;
    const row2Y = top + rowH;

    for (let i = 0; i < videoState.clips.length; i++) {
        const clip = videoState.clips[i];
        const x1 = (clip.start / total) * w;
        const x2 = ((clip.end + 1) / total) * w;
        const t1 = clip.tool1 || 'idle';
        const t2 = clip.tool2 || 'idle';
        const isSel = (i === videoState.selectedClipIdx);

        ctx.globalAlpha = isSel ? 1.0 : 0.75;
        ctx.fillStyle = PHASE_COLORS[t1]?.bg || '#555';
        ctx.fillRect(x1, row1Y, x2 - x1, rowH);
        ctx.fillStyle = PHASE_COLORS[t2]?.bg || '#555';
        ctx.fillRect(x1, row2Y, x2 - x1, rowH);

        if (isSel) {
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, top, x2 - x1, bottom - top);
            ctx.fillStyle = '#fff';
            ctx.fillRect(x1 - 2, top, 5, bottom - top);
            ctx.fillRect(x2 - 3, top, 5, bottom - top);
        }

        // Cycle label centred across both rows
        const clipW = x2 - x1;
        if (clipW > 20) {
            ctx.globalAlpha = 1.0;
            // Pick text colour from whichever tool is non-idle (else fall back to row1's text colour)
            const labelColour = PHASE_COLORS[(t1 !== 'idle') ? t1 : t2]?.text || '#fff';
            ctx.fillStyle = labelColour;
            ctx.font = `${Math.min(11, rowH - 2)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const label = clip.cycle > 0 ? `C${clip.cycle}` : '';
            if (label) ctx.fillText(label, (x1 + x2) / 2, top + (bottom - top) / 2);
        }

        ctx.globalAlpha = 1.0;
    }

    // Divider line + T1/T2 row labels
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.fillRect(0, row2Y, w, 1);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = `bold ${Math.max(9, Math.floor(rowH * 0.55))}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('T1', 4, row1Y + rowH / 2);
    ctx.fillText('T2', 4, row2Y + rowH / 2);

    // Mark start indicator
    if (videoState.markStart !== null) {
        const mx = (videoState.markStart / total) * w;
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(mx, 0);
        ctx.lineTo(mx, h);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Cursor (current frame position)
    const cx = (videoState.currentIdx / total) * w;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    // Cursor triangle at top
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(cx - 5, 0);
    ctx.lineTo(cx + 5, 0);
    ctx.lineTo(cx, 6);
    ctx.closePath();
    ctx.fill();
}

// ---- Timeline interaction ----

function videoTimelineGetIdx(e) {
    const canvas = videoElements.timelineCanvas;
    const rect = canvas.getBoundingClientRect();
    // getBoundingClientRect() already accounts for container scroll
    const x = e.clientX - rect.left;
    const ratio = x / canvas.width;
    return Math.round(ratio * (videoState.allFrameIndices.length - 1));
}

videoElements.timelineCanvas?.addEventListener('mousedown', (e) => {
    // Right-click is handled by the pan listener
    if (e.button !== 0) return;
    const idx = videoTimelineGetIdx(e);
    const total = videoState.allFrameIndices.length;
    const canvas = videoElements.timelineCanvas;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;

    // Check if clicking on selected clip handles
    if (videoState.selectedClipIdx !== null) {
        const clip = videoState.clips[videoState.selectedClipIdx];
        const x1 = (clip.start / total) * canvas.width;
        const x2 = ((clip.end + 1) / total) * canvas.width;

        if (Math.abs(mx - x1) < 8) {
            videoState.dragState = {type: 'left', clipIdx: videoState.selectedClipIdx, startX: mx, origStart: clip.start, origEnd: clip.end};
            return;
        }
        if (Math.abs(mx - x2) < 8) {
            videoState.dragState = {type: 'right', clipIdx: videoState.selectedClipIdx, startX: mx, origStart: clip.start, origEnd: clip.end};
            return;
        }
        if (mx > x1 && mx < x2) {
            videoState.dragState = {type: 'move', clipIdx: videoState.selectedClipIdx, startX: mx, origStart: clip.start, origEnd: clip.end};
            return;
        }
    }

    // Check if clicking on any clip to select it
    for (let i = 0; i < videoState.clips.length; i++) {
        const clip = videoState.clips[i];
        const x1 = (clip.start / total) * canvas.width;
        const x2 = ((clip.end + 1) / total) * canvas.width;
        if (mx >= x1 && mx <= x2) {
            videoState.selectedClipIdx = i;
            if (videoElements.cycleInput) videoElements.cycleInput.value = clip.cycle;
            videoLoadFrame(clip.start);
            return;
        }
    }

    // Click on empty space — seek
    videoState.selectedClipIdx = null;
    if (idx >= 0 && idx < total) videoLoadFrame(idx);
});

videoElements.timelineCanvas?.addEventListener('mousemove', (e) => {
    if (!videoState.dragState) return;
    const ds = videoState.dragState;
    const canvas = videoElements.timelineCanvas;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const total = videoState.allFrameIndices.length;
    const dx = mx - ds.startX;
    const dFrames = Math.round((dx / canvas.width) * total);
    const clip = videoState.clips[ds.clipIdx];

    // Find neighboring clip boundaries
    const prevEnd = ds.clipIdx > 0 ? videoState.clips[ds.clipIdx - 1].end : -1;
    const nextStart = ds.clipIdx < videoState.clips.length - 1 ? videoState.clips[ds.clipIdx + 1].start : total;

    if (ds.type === 'left') {
        clip.start = Math.max(prevEnd + 1, Math.min(clip.end - 1, ds.origStart + dFrames));
    } else if (ds.type === 'right') {
        clip.end = Math.min(nextStart - 1, Math.max(clip.start + 1, ds.origEnd + dFrames));
    } else if (ds.type === 'move') {
        const len = ds.origEnd - ds.origStart;
        let newStart = ds.origStart + dFrames;
        newStart = Math.max(prevEnd + 1, Math.min(nextStart - 1 - len, newStart));
        clip.start = newStart;
        clip.end = newStart + len;
    }
    videoRenderTimeline();
});

document.addEventListener('mouseup', () => {
    if (videoState.dragState) {
        videoState.dragState = null;
        videoRenderTimeline();
    }
});

// ---- Timeline zoom (scroll wheel) ----

videoElements.timelineContainer?.addEventListener('wheel', (e) => {
    e.preventDefault();
    const container = videoElements.timelineContainer;
    const oldZoom = videoState.timelineZoom;

    // Zoom in/out
    if (e.deltaY < 0) {
        videoState.timelineZoom = Math.min(50, oldZoom * 1.25);
    } else {
        videoState.timelineZoom = Math.max(1, oldZoom / 1.25);
    }

    if (videoState.timelineZoom === oldZoom) return;

    // Zoom centered on mouse position
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const scrollBefore = container.scrollLeft;
    const ratioAtMouse = (scrollBefore + mouseX) / (container.clientWidth * oldZoom);

    videoResizeTimeline();
    videoRenderTimeline();

    // Adjust scroll so the frame under the mouse stays put
    const newCanvasW = container.clientWidth * videoState.timelineZoom;
    container.scrollLeft = ratioAtMouse * newCanvasW - mouseX;
}, {passive: false});

/**
 * Auto-scroll timeline to keep cursor visible.
 */
function videoScrollTimelineToCursor() {
    const container = videoElements.timelineContainer;
    const canvas = videoElements.timelineCanvas;
    const total = videoState.allFrameIndices.length;
    if (total === 0) return;

    const cursorX = (videoState.currentIdx / total) * canvas.width;
    const scrollLeft = container.scrollLeft;
    const viewWidth = container.clientWidth;
    const margin = viewWidth * 0.15;

    if (cursorX < scrollLeft + margin) {
        container.scrollLeft = cursorX - margin;
    } else if (cursorX > scrollLeft + viewWidth - margin) {
        container.scrollLeft = cursorX - viewWidth + margin;
    }
}

// ---- Clip operations ----

function videoMarkStart() {
    videoState.markStart = videoState.currentIdx;
    videoElements.markIndicator.style.display = '';
    videoElements.markFrame.textContent = videoState.allFrameIndices[videoState.currentIdx];
    showToast(`Mark start: frame ${videoState.allFrameIndices[videoState.currentIdx]}`);
    videoRenderTimeline();
}

function videoMarkEnd() {
    if (videoState.markStart === null) {
        showToast('Press M first to mark start', true);
        return;
    }

    let start = Math.min(videoState.markStart, videoState.currentIdx);
    let end = Math.max(videoState.markStart, videoState.currentIdx);

    // Check for overlaps
    for (const clip of videoState.clips) {
        if (start <= clip.end && end >= clip.start) {
            showToast('Clip would overlap existing clip', true);
            return;
        }
    }

    const cycle = videoElements.cycleInput ? parseInt(videoElements.cycleInput.value) || 0 : 0;
    videoState.clips.push({start, end, tool1: 'idle', tool2: 'idle', cycle});
    videoState.clips.sort((a, b) => a.start - b.start);

    // Select the new clip
    videoState.selectedClipIdx = videoState.clips.findIndex(c => c.start === start && c.end === end);

    // Clear mark
    videoState.markStart = null;
    videoElements.markIndicator.style.display = 'none';

    showToast(`Clip created (${end - start + 1} frames) — assign a phase`);
    videoRenderTimeline();
    videoUpdateCoverage();

    // Auto-start: place mark at next frame after end
    if (videoState.autoStartNext && end + 1 < videoState.allFrameIndices.length) {
        videoState.markStart = end + 1;
        videoElements.markIndicator.style.display = '';
        videoElements.markFrame.textContent = videoState.allFrameIndices[end + 1];
        videoLoadFrame(end + 1);
    }
}

function videoDeleteClip() {
    if (videoState.selectedClipIdx === null) {
        showToast('No clip selected', true);
        return;
    }
    videoState.clips.splice(videoState.selectedClipIdx, 1);
    videoState.selectedClipIdx = null;
    showToast('Clip deleted');
    videoRenderTimeline();
    videoUpdateCoverage();
}

function videoAssignPhase(phase) {
    if (videoState.selectedClipIdx === null) {
        showToast('Select a clip first (click timeline or use M/N to create)', true);
        return;
    }
    const clip = videoState.clips[videoState.selectedClipIdx];
    const target = videoGetActiveTool();
    if (target === 'both') {
        clip.tool1 = phase;
        clip.tool2 = phase;
    } else if (target === 1) {
        clip.tool1 = phase;
    } else if (target === 2) {
        clip.tool2 = phase;
    }
    const cycle = videoElements.cycleInput ? parseInt(videoElements.cycleInput.value) || 0 : 0;
    clip.cycle = cycle;
    const label = target === 'both' ? 'T1+T2' : `T${target}`;
    showToast(`${label}: ${phase} (cycle ${cycle})`);
    videoRenderTimeline();
    videoUpdateCoverage();
    videoLoadFrame(videoState.currentIdx);  // refresh phase readout

    // Auto-increment cycle after completing a "place" phase (new peg transfer cycle)
    if (phase === 'place' && videoElements.cycleInput) {
        videoElements.cycleInput.value = cycle + 1;
    }

    if (videoState.autoSaveOnEdit) {
        videoSaveClips();
    } else {
        videoMarkUnsaved();
    }
}

function videoGetActiveTool() {
    if (videoState.activeTool === 1 || videoState.activeTool === 2) return videoState.activeTool;
    return 'both';
}

function videoSetActiveTool(target) {
    if (target !== 'both' && target !== 1 && target !== 2) target = 'both';
    videoState.activeTool = target;
    document.querySelectorAll('.video-tool-btn').forEach(b => {
        const t = b.dataset.tool === 'both' ? 'both' : parseInt(b.dataset.tool);
        b.classList.toggle('active', t === target);
    });
    const label = target === 'both' ? 'Both tools' : `Tool ${target} only`;
    showToast(`Apply to: ${label}`);
}

function videoToggleActiveToolKey(toolNum) {
    const cur = videoGetActiveTool();
    videoSetActiveTool(cur === toolNum ? 'both' : toolNum);
}

function videoJumpPrevClip() {
    if (videoState.clips.length === 0) return;
    // Find clip before current position
    for (let i = videoState.clips.length - 1; i >= 0; i--) {
        if (videoState.clips[i].end < videoState.currentIdx) {
            videoState.selectedClipIdx = i;
            videoLoadFrame(videoState.clips[i].start);
            return;
        }
    }
    // Wrap to last
    videoState.selectedClipIdx = videoState.clips.length - 1;
    videoLoadFrame(videoState.clips[videoState.clips.length - 1].start);
}

function videoJumpNextClip() {
    if (videoState.clips.length === 0) return;
    for (let i = 0; i < videoState.clips.length; i++) {
        if (videoState.clips[i].start > videoState.currentIdx) {
            videoState.selectedClipIdx = i;
            videoLoadFrame(videoState.clips[i].start);
            return;
        }
    }
    // Wrap to first
    videoState.selectedClipIdx = 0;
    videoLoadFrame(videoState.clips[0].start);
}

function videoJumpUnmarked() {
    const total = videoState.allFrameIndices.length;
    // Find first index after current that has no clip
    for (let i = videoState.currentIdx + 1; i < total; i++) {
        if (!videoGetClipAt(i)) {
            videoLoadFrame(i);
            showToast(`Jumped to unmarked frame ${videoState.allFrameIndices[i]}`);
            return;
        }
    }
    // Wrap from beginning
    for (let i = 0; i <= videoState.currentIdx; i++) {
        if (!videoGetClipAt(i)) {
            videoLoadFrame(i);
            showToast(`Jumped to unmarked frame ${videoState.allFrameIndices[i]}`);
            return;
        }
    }
    showToast('All frames are covered!');
}

/**
 * Update the cycle of the currently selected clip from the cycle input.
 */
function videoUpdateSelectedClipCycle() {
    if (videoState.selectedClipIdx === null) return;
    const cycle = videoElements.cycleInput ? parseInt(videoElements.cycleInput.value) || 0 : 0;
    videoState.clips[videoState.selectedClipIdx].cycle = cycle;
    videoRenderTimeline();
    if (videoState.autoSaveOnEdit) {
        videoSaveClips();
    } else {
        videoMarkUnsaved();
    }
}

/**
 * Toggle the raw annotation panel and populate with current annotations.
 */
function videoToggleAnnotationPanel() {
    const panel = document.getElementById('video-annotation-panel');
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
        document.getElementById('video-annotation-text').value = videoGetRawAnnotationJSON();
    } else {
        panel.style.display = 'none';
    }
}

/**
 * Build raw annotation JSON from current clips for copy-paste.
 */
function videoGetRawAnnotationJSON() {
    const annotations = {};
    for (const clip of videoState.clips) {
        const t1 = clip.tool1 || 'idle';
        const t2 = clip.tool2 || 'idle';
        // Explicit clips persist as coarse='idle' (not '') so phase_summary keeps them.
        const coarse = (t1 !== 'idle') ? t1 : (t2 !== 'idle' ? t2 : 'idle');
        for (let i = clip.start; i <= clip.end; i++) {
            const frameIdx = videoState.allFrameIndices[i];
            annotations[frameIdx] = {
                tool1: t1,
                tool2: t2,
                coarse,
                fine: '',
                cycle_index: clip.cycle,
                active_tool: 0,
                events: []
            };
        }
    }
    return JSON.stringify(annotations, null, 2);
}

/**
 * Backup current annotations to localStorage.
 */
function videoBackupToLocalStorage() {
    try {
        if (!state.trialId) return;
        const key = `video_phase_backup_${state.trialId}`;
        const backup = {
            timestamp: new Date().toISOString(),
            annotations: JSON.parse(videoGetRawAnnotationJSON()),
        };
        localStorage.setItem(key, JSON.stringify(backup));
        console.log(`[VideoBackup] Saved to localStorage for ${state.trialId}`);
    } catch (e) {
        console.warn('[VideoBackup] localStorage backup failed:', e);
    }
}

/**
 * Restore annotations from localStorage backup.
 */
async function videoRestoreFromLocalStorage() {
    if (!state.trialId) { showToast('No trial loaded', true); return; }
    const key = `video_phase_backup_${state.trialId}`;
    const raw = localStorage.getItem(key);
    if (!raw) { showToast('No backup found for this trial', true); return; }
    let backup;
    try {
        backup = JSON.parse(raw);
    } catch (e) {
        showToast('Corrupt backup data', true);
        return;
    }
    if (!backup.annotations || typeof backup.annotations !== 'object') {
        showToast('Backup has no annotation data', true);
        return;
    }
    const when = new Date(backup.timestamp).toLocaleString();
    if (!confirm(`Restore backup from ${when}? This will overwrite current annotations and save.`)) return;

    try {
        await videoApplyAnnotationData(backup.annotations);
        showToast(`Restored from backup (${when})`);
    } catch (e) {
        console.error('[VideoRestore] Failed:', e);
        showToast('Restore failed: ' + e.message, true);
    }
}

/**
 * Apply annotation JSON from the textarea: validate, save to server, rebuild timeline.
 */
async function videoApplyFromTextarea() {
    if (!state.trialId) { showToast('No trial loaded', true); return; }
    const textarea = document.getElementById('video-annotation-text');
    const text = textarea?.value?.trim();
    if (!text) { showToast('Textarea is empty — paste JSON first', true); return; }

    let data;
    try {
        data = JSON.parse(text);
    } catch (e) {
        showToast('Invalid JSON: ' + e.message, true);
        return;
    }
    if (typeof data !== 'object' || data === null || Array.isArray(data)) {
        showToast('Expected JSON object mapping frame indices to phase data', true);
        return;
    }
    const frameCount = Object.keys(data).length;
    if (frameCount === 0) { showToast('No annotations in JSON', true); return; }
    if (!confirm(`Apply ${frameCount} frame annotations? This will overwrite current labels and save.`)) return;

    try {
        videoBackupToLocalStorage();
        await videoApplyAnnotationData(data);
        showToast(`Applied and saved ${frameCount} frame annotations`);
    } catch (e) {
        console.error('[VideoApply] Failed:', e);
        showToast('Apply failed: ' + e.message, true);
    }
}

/**
 * Apply a frame→phase annotation object: save via bulk API, reload, rebuild clips.
 */
async function videoApplyAnnotationData(annotations) {
    const frames = videoState.allFrameIndices;

    // Snapshot which frames currently have labels (before we modify state)
    const previouslyLabeled = new Set();
    for (const frameIdx of frames) {
        if (state.phaseLabels[frameIdx]) previouslyLabeled.add(frameIdx);
    }

    // Build the set of frames covered by the new annotations
    const coveredFrames = new Set();
    for (const key of Object.keys(annotations)) {
        const frameIdx = Number(key);
        const label = annotations[key];
        const phase = label?.coarse || (typeof label === 'string' ? label : null);
        if (phase) coveredFrames.add(frameIdx);
    }

    // Group contiguous (tool1, tool2, cycle) ranges for bulk save
    let clipStart = null;
    let clipT1 = null;
    let clipT2 = null;
    let clipCycle = 0;
    const bulkOps = [];

    for (let i = 0; i < frames.length; i++) {
        const frameIdx = frames[i];
        const label = annotations[String(frameIdx)] || annotations[frameIdx];
        let t1 = null, t2 = null, cycle = 0;
        if (label) {
            if (typeof label === 'string') { t1 = t2 = label; }
            else {
                t1 = label.tool1 || label.coarse || null;
                t2 = label.tool2 || label.coarse || null;
                cycle = label.cycle_index || 0;
            }
        }
        const has = !!(t1 || t2);

        if (has) {
            if (t1 === clipT1 && t2 === clipT2 && cycle === clipCycle) {
                // continue
            } else {
                if (clipStart !== null) {
                    bulkOps.push({start: clipStart, end: frames[i - 1], tool1: clipT1, tool2: clipT2, cycle: clipCycle});
                }
                clipStart = frameIdx;
                clipT1 = t1 || 'idle';
                clipT2 = t2 || 'idle';
                clipCycle = cycle;
            }
        } else if (clipStart !== null) {
            bulkOps.push({start: clipStart, end: frames[i - 1], tool1: clipT1, tool2: clipT2, cycle: clipCycle});
            clipStart = null;
            clipT1 = null;
            clipT2 = null;
        }
    }
    if (clipStart !== null) {
        bulkOps.push({start: clipStart, end: frames[frames.length - 1], tool1: clipT1, tool2: clipT2, cycle: clipCycle});
    }

    // Build all operations into a single batched request
    const operations = [];
    for (const op of bulkOps) {
        const coarse = (op.tool1 !== 'idle') ? op.tool1 : (op.tool2 !== 'idle' ? op.tool2 : 'idle');
        operations.push({
            start_frame: op.start,
            end_frame: op.end,
            phase_data: {
                tool1: op.tool1, tool2: op.tool2,
                coarse, fine: '', cycle_index: op.cycle,
                active_tool: 0, events: [],
            },
        });
    }

    // Clear frames that had labels before but are not in the new data
    const toClear = frames.filter(f => previouslyLabeled.has(f) && !coveredFrames.has(f));
    if (toClear.length > 0) {
        operations.push({
            frames: toClear,
            phase_data: {
                tool1: 'idle', tool2: 'idle',
                coarse: '', fine: '', cycle_index: 0,
                active_tool: 0, events: [],
            },
        });
    }

    console.log(`[VideoApply] Batch saving ${operations.length} operations covering ${coveredFrames.size} frames`);
    await api(`/trials/${state.trialId}/phase_bulk`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ operations })
    });

    // Reload from server and rebuild UI
    await loadPhaseSummary(state.trialId);
    videoReconstructClips();
    videoState.selectedClipIdx = null;
    videoRenderTimeline();
    videoUpdateCoverage();

    // Refresh textarea to match new state
    const annPanel = document.getElementById('video-annotation-panel');
    if (annPanel && annPanel.style.display !== 'none') {
        document.getElementById('video-annotation-text').value = videoGetRawAnnotationJSON();
    }
}

function videoMarkUnsaved() {
    const btn = document.getElementById('video-save');
    if (!btn) return;
    btn.classList.add('video-save-dirty');
    btn.textContent = 'S Save *';
}

function videoMarkSaved() {
    const btn = document.getElementById('video-save');
    if (!btn) return;
    btn.classList.remove('video-save-dirty');
    btn.textContent = 'S Save';
}

async function videoSaveClips() {
    if (videoState.clips.length === 0) {
        showToast('No clips to save', true);
        return;
    }

    try {
        console.log(`[VideoSave] Saving ${videoState.clips.length} clips for ${state.trialId}`);

        // Backup to localStorage before saving
        videoBackupToLocalStorage();

        // Build all operations into a single batched request
        const operations = [];
        const coveredFrames = new Set();

        for (const clip of videoState.clips) {
            const startFrame = videoState.allFrameIndices[clip.start];
            const endFrame = videoState.allFrameIndices[clip.end];
            const t1 = clip.tool1 || 'idle';
            const t2 = clip.tool2 || 'idle';
            // Explicit clip → coarse='idle' (not '') so phase_summary keeps the frame.
            const coarse = (t1 !== 'idle') ? t1 : (t2 !== 'idle' ? t2 : 'idle');
            operations.push({
                start_frame: startFrame,
                end_frame: endFrame,
                phase_data: {
                    tool1: t1, tool2: t2,
                    coarse, fine: '', cycle_index: clip.cycle,
                    active_tool: 0, events: [],
                },
            });
            for (let i = clip.start; i <= clip.end; i++) {
                coveredFrames.add(videoState.allFrameIndices[i]);
            }
        }

        // Find frames that had labels before but are now uncovered (need clearing)
        const framesToClear = [];
        for (const frameIdx of videoState.allFrameIndices) {
            if (!coveredFrames.has(frameIdx) && state.phaseLabels[frameIdx]) {
                framesToClear.push(frameIdx);
            }
        }
        if (framesToClear.length > 0) {
            operations.push({
                frames: framesToClear,
                phase_data: {
                    tool1: 'idle', tool2: 'idle',
                    coarse: '', fine: '', cycle_index: 0,
                    active_tool: 0, events: [],
                },
            });
        }

        // Single request for all operations
        const result = await api(`/trials/${state.trialId}/phase_bulk`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ operations })
        });
        console.log(`[VideoSave] Batch saved: ${result.updated_count} frames updated`);

        // Reload phase labels from server to verify save succeeded
        await loadPhaseSummary(state.trialId);

        // Reconstruct clips from verified server data, preserving selection
        const prevIdx = videoState.currentIdx;
        videoReconstructClips();
        videoState.selectedClipIdx = videoState.clips.findIndex(c => prevIdx >= c.start && prevIdx <= c.end);
        if (videoState.selectedClipIdx === -1) videoState.selectedClipIdx = null;
        videoRenderTimeline();
        videoUpdateCoverage();

        console.log(`[VideoSave] Verified: ${Object.keys(state.phaseLabels).length} labeled frames, ${videoState.clips.length} clips reconstructed`);
        showToast(`Saved ${videoState.clips.length} clips (${coveredFrames.size} frames)`);
        videoMarkSaved();

        // Refresh raw annotation panel if open
        const annPanel = document.getElementById('video-annotation-panel');
        if (annPanel && annPanel.style.display !== 'none') {
            document.getElementById('video-annotation-text').value = videoGetRawAnnotationJSON();
        }
    } catch (e) {
        console.error('[VideoSave] Failed:', e);
        showToast('Save failed: ' + e.message, true);
    }
}

// ---- Video mode keyboard handler ----

function videoHandleKeydown(e) {
    if (!videoState.active) return false;

    // Don't capture if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return false;

    const key = e.key.toLowerCase();

    switch (key) {
        case ' ':
            e.preventDefault();
            videoTogglePlayback();
            return true;
        case 'arrowleft':
            e.preventDefault();
            videoLoadFrame(videoState.currentIdx - (e.shiftKey ? 10 : 1));
            return true;
        case 'arrowright':
            e.preventDefault();
            videoLoadFrame(videoState.currentIdx + (e.shiftKey ? 10 : 1));
            return true;
        case 'home':
            e.preventDefault();
            videoLoadFrame(0);
            return true;
        case 'end':
            e.preventDefault();
            videoLoadFrame(videoState.allFrameIndices.length - 1);
            return true;
        case 'm':
            videoMarkStart();
            return true;
        case 'n':
            videoMarkEnd();
            return true;
        case 'backspace':
        case 'delete':
            e.preventDefault();
            videoDeleteClip();
            return true;
        case '[':
            videoJumpPrevClip();
            return true;
        case ']':
            videoJumpNextClip();
            return true;
        case 'j':
            videoJumpUnmarked();
            return true;
        case 's':
            videoSaveClips();
            return true;
        case 'escape':
            closeVideoMode();
            return true;
        case '1':
            e.preventDefault();
            videoToggleActiveToolKey(1);
            return true;
        case '2':
            e.preventDefault();
            videoToggleActiveToolKey(2);
            return true;
        case '3':
            e.preventDefault();
            videoSetActiveTool('both');
            return true;
    }

    // Phase keybinds
    if (PHASE_KEYBINDS[key]) {
        videoAssignPhase(PHASE_KEYBINDS[key]);
        return true;
    }

    return false;
}

// ---- Video mode event listeners ----

document.getElementById('video-mode-btn')?.addEventListener('click', () => {
    if (videoState.active) {
        closeVideoMode();
    } else {
        openVideoMode();
    }
});

document.getElementById('video-close')?.addEventListener('click', closeVideoMode);

document.getElementById('video-play-pause')?.addEventListener('click', videoTogglePlayback);
document.getElementById('video-step-back')?.addEventListener('click', () => videoLoadFrame(videoState.currentIdx - 1));
document.getElementById('video-step-fwd')?.addEventListener('click', () => videoLoadFrame(videoState.currentIdx + 1));
document.getElementById('video-step-back-10')?.addEventListener('click', () => videoLoadFrame(videoState.currentIdx - 10));
document.getElementById('video-step-fwd-10')?.addEventListener('click', () => videoLoadFrame(videoState.currentIdx + 10));

videoElements.slider?.addEventListener('input', (e) => {
    videoLoadFrame(parseInt(e.target.value));
});

videoElements.speedSelect?.addEventListener('change', (e) => {
    videoSetSpeed(parseFloat(e.target.value));
});

document.getElementById('video-mark-start')?.addEventListener('click', videoMarkStart);
document.getElementById('video-mark-end')?.addEventListener('click', videoMarkEnd);
document.getElementById('video-delete-clip')?.addEventListener('click', videoDeleteClip);
document.getElementById('video-prev-clip')?.addEventListener('click', videoJumpPrevClip);
document.getElementById('video-next-clip')?.addEventListener('click', videoJumpNextClip);
document.getElementById('video-jump-unmarked')?.addEventListener('click', videoJumpUnmarked);
document.getElementById('video-save')?.addEventListener('click', videoSaveClips);

// Auto-start toggle
document.getElementById('video-auto-start')?.addEventListener('change', (e) => {
    videoState.autoStartNext = e.target.checked;
});

// Auto-save toggle
document.getElementById('video-auto-save')?.addEventListener('change', (e) => {
    videoState.autoSaveOnEdit = e.target.checked;
    if (videoState.autoSaveOnEdit) {
        // If turning auto-save back on and there are pending edits, flush them
        const btn = document.getElementById('video-save');
        if (btn && btn.classList.contains('video-save-dirty')) {
            videoSaveClips();
        }
    }
});

// Cycle +/- buttons
document.getElementById('video-cycle-dec')?.addEventListener('click', () => {
    const inp = videoElements.cycleInput;
    if (!inp) return;
    inp.value = Math.max(0, (parseInt(inp.value) || 0) - 1);
    videoUpdateSelectedClipCycle();
});
document.getElementById('video-cycle-inc')?.addEventListener('click', () => {
    const inp = videoElements.cycleInput;
    if (!inp) return;
    inp.value = (parseInt(inp.value) || 0) + 1;
    videoUpdateSelectedClipCycle();
});
videoElements.cycleInput?.addEventListener('change', () => {
    videoUpdateSelectedClipCycle();
});

// Raw annotation panel
document.getElementById('video-show-annotations')?.addEventListener('click', videoToggleAnnotationPanel);
document.getElementById('video-annotation-panel-close')?.addEventListener('click', () => {
    document.getElementById('video-annotation-panel').style.display = 'none';
});
document.getElementById('video-annotation-copy')?.addEventListener('click', () => {
    const textarea = document.getElementById('video-annotation-text');
    textarea.select();
    navigator.clipboard.writeText(textarea.value).then(() => showToast('Copied to clipboard'));
});
document.getElementById('video-annotation-apply')?.addEventListener('click', videoApplyFromTextarea);
document.getElementById('video-annotation-restore')?.addEventListener('click', videoRestoreFromLocalStorage);

// Right-click pan on timeline
videoElements.timelineContainer?.addEventListener('contextmenu', (e) => {
    if (videoState.timelineZoom > 1) e.preventDefault();
});
videoElements.timelineCanvas?.addEventListener('mousedown', (e) => {
    if (e.button === 2 && videoState.timelineZoom > 1) {
        e.preventDefault();
        videoState.timelinePan = {startX: e.clientX, startScrollLeft: videoElements.timelineContainer.scrollLeft};
    }
});
document.addEventListener('mousemove', (e) => {
    if (!videoState.timelinePan) return;
    const dx = e.clientX - videoState.timelinePan.startX;
    videoElements.timelineContainer.scrollLeft = videoState.timelinePan.startScrollLeft - dx;
});
document.addEventListener('mouseup', (e) => {
    if (videoState.timelinePan) {
        videoState.timelinePan = null;
    }
});

document.querySelectorAll('.video-phase-btn').forEach(btn => {
    btn.addEventListener('click', () => videoAssignPhase(btn.dataset.phase));
});

document.querySelectorAll('.video-tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const v = btn.dataset.tool;
        const target = v === 'both' ? 'both' : (parseInt(v) || 1);
        videoSetActiveTool(target);
    });
});

document.getElementById('video-next-trial')?.addEventListener('click', async () => {
    closeVideoMode();
    await switchTrial(+1);
    if (state.trialId) await openVideoMode();
});

document.getElementById('video-prev-trial')?.addEventListener('click', async () => {
    closeVideoMode();
    await switchTrial(-1);
    if (state.trialId) await openVideoMode();
});

// Resize timeline on window resize
window.addEventListener('resize', () => {
    if (videoState.active) {
        videoResizeTimeline();
        videoRenderTimeline();
    }
});

// ============================================================================
// Initialization
// ============================================================================

async function init() {
    console.log('Initializing Surgical Image Annotation Tool...');

    // Load phase definitions
    loadPhaseDefinitions();

    // Phase 1: Show dataset list immediately (fast /datasets endpoint)
    await loadDatasets();
    console.log('Datasets loaded — UI ready');

    // Phase 2: Load progress in background, update UI when ready
    loadProgressAsync();

    // Check SAM availability in background
    checkSamStatus();
}

init();

// Guide modal
const guideBtn = document.getElementById('guide-btn');
const guideModal = document.getElementById('guide-modal');
const guideClose = document.getElementById('guide-close');
if (guideBtn && guideModal) {
    guideBtn.addEventListener('click', () => { guideModal.style.display = 'flex'; });
    guideClose?.addEventListener('click', () => { guideModal.style.display = 'none'; });
    guideModal.addEventListener('click', (e) => { if (e.target === guideModal) guideModal.style.display = 'none'; });
}
