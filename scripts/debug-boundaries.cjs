/**
 * debug-boundaries.cjs
 *
 * Reads boundary_types.i8, plate_map.i16, and height_map.f32 from a generation
 * run directory and produces diagnostic visualizations and statistics about
 * tectonic boundary patterns.
 *
 * Outputs (saved to the run directory):
 *   1) boundary_types_vis.jpg  — color-coded boundary type map
 *   2) boundary_width.jpg      — neighbor-difference heat map showing boundary zone width
 *   3) pre_erosion_relief.jpg  — interior vs boundary cell height comparison
 *
 * Usage:
 *   node scripts/debug-boundaries.cjs [path-to-run-directory]
 */

const fs = require('fs');
const path = require('path');
const jpeg = require('jpeg-js');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const DEFAULT_RUN_DIR = path.resolve(
  __dirname,
  '..',
  'generations',
  'runs',
  '2026-02-25T01-10-03-008Z_seed-42'
);

const WIDTH = 2048;
const HEIGHT = 1024;
const TOTAL = WIDTH * HEIGHT;

// Boundary type colors
const BOUNDARY_COLORS = {
  0: [0, 0, 0],       // Interior: black
  1: [255, 40, 40],   // Convergent: bright red
  2: [40, 100, 255],  // Divergent: bright blue
  3: [40, 255, 40],   // Transform: bright green
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function writeJpg(filePath, width, height, pixelAt, quality = 92) {
  const rgba = Buffer.alloc(width * height * 4);
  let off = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const [r, g, b] = pixelAt(i, x, y);
      rgba[off++] = clamp(Math.round(r), 0, 255);
      rgba[off++] = clamp(Math.round(g), 0, 255);
      rgba[off++] = clamp(Math.round(b), 0, 255);
      rgba[off++] = 255;
    }
  }
  const encoded = jpeg.encode({ data: rgba, width, height }, quality);
  fs.writeFileSync(filePath, encoded.data);
}

function readFloat32(filePath) {
  const buf = fs.readFileSync(filePath);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function readInt8(filePath) {
  const buf = fs.readFileSync(filePath);
  return new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

function readInt16LE(filePath) {
  const buf = fs.readFileSync(filePath);
  // Int16Array requires alignment; copy into an aligned buffer
  const aligned = Buffer.alloc(buf.length);
  buf.copy(aligned);
  return new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
}

/** Wrap x horizontally (equirectangular), clamp y to valid range. */
function wrapCoord(x, y) {
  let wy = y;
  let wx = x;
  // Vertical: clamp (poles)
  if (wy < 0) wy = 0;
  if (wy >= HEIGHT) wy = HEIGHT - 1;
  // Horizontal: wrap
  wx = ((wx % WIDTH) + WIDTH) % WIDTH;
  return wy * WIDTH + wx;
}

// 8-neighbor offsets
const NEIGHBORS = [
  [-1, -1], [0, -1], [1, -1],
  [-1,  0],          [1,  0],
  [-1,  1], [0,  1], [1,  1],
];

// ---------------------------------------------------------------------------
// 1. Boundary types visualization
// ---------------------------------------------------------------------------
function createBoundaryTypesVis(boundaryTypes, runDir) {
  const outPath = path.join(runDir, 'boundary_types_vis.jpg');
  console.log(`Writing boundary types visualization: ${outPath}`);
  writeJpg(outPath, WIDTH, HEIGHT, (i) => {
    const t = boundaryTypes[i];
    return BOUNDARY_COLORS[t] || [128, 0, 128]; // magenta for unknown
  });
  return outPath;
}

// ---------------------------------------------------------------------------
// 2. Boundary width visualization
// ---------------------------------------------------------------------------
function computeNeighborDiffMap(boundaryTypes, plateMap) {
  const diffMap = new Uint8Array(TOTAL);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = y * WIDTH + x;
      const myType = boundaryTypes[i];
      const myPlate = plateMap[i];
      let diffCount = 0;
      for (const [dx, dy] of NEIGHBORS) {
        const ni = wrapCoord(x + dx, y + dy);
        if (boundaryTypes[ni] !== myType || plateMap[ni] !== myPlate) {
          diffCount++;
        }
      }
      diffMap[i] = diffCount;
    }
  }
  return diffMap;
}

function createBoundaryWidthVis(diffMap, runDir) {
  const outPath = path.join(runDir, 'boundary_width.jpg');
  console.log(`Writing boundary width visualization: ${outPath}`);
  writeJpg(outPath, WIDTH, HEIGHT, (i) => {
    const d = diffMap[i];
    let v;
    if (d === 0) v = 0;
    else if (d <= 2) v = 80;
    else if (d <= 4) v = 160;
    else v = 255;
    return [v, v, v];
  });
  return outPath;
}

// ---------------------------------------------------------------------------
// 3. Statistics
// ---------------------------------------------------------------------------
function printBoundaryStats(boundaryTypes, diffMap) {
  // Count each type
  const typeCounts = [0, 0, 0, 0];
  const typeNames = ['interior', 'convergent', 'divergent', 'transform'];
  for (let i = 0; i < TOTAL; i++) {
    const t = boundaryTypes[i];
    if (t >= 0 && t <= 3) typeCounts[t]++;
  }

  console.log('\n=== BOUNDARY TYPE COUNTS ===');
  for (let t = 0; t <= 3; t++) {
    const pct = ((typeCounts[t] / TOTAL) * 100).toFixed(2);
    console.log(`  Type ${t} (${typeNames[t].padEnd(11)}): ${String(typeCounts[t]).padStart(9)}  (${pct.padStart(6)}%)`);
  }

  const nonInterior = TOTAL - typeCounts[0];
  console.log(`\n  Non-interior cells: ${nonInterior} (${((nonInterior / TOTAL) * 100).toFixed(2)}%)`);

  // Average boundary width
  // For each non-interior cell, we measure how many of its 8 neighbors are also
  // non-interior (same or different type). This approximates the local "width"
  // of the boundary zone.
  // A more meaningful metric: for non-interior cells, compute the distance to
  // the nearest interior cell via BFS. Then report statistics on that distance.
  // But a simpler proxy: measure contiguous runs of non-interior cells along
  // scan lines.
  //
  // Approach: for each row, find contiguous runs of non-interior cells.
  const runLengths = [];
  for (let y = 0; y < HEIGHT; y++) {
    let runLen = 0;
    for (let x = 0; x < WIDTH; x++) {
      const i = y * WIDTH + x;
      if (boundaryTypes[i] !== 0) {
        runLen++;
      } else {
        if (runLen > 0) {
          runLengths.push(runLen);
          runLen = 0;
        }
      }
    }
    // Handle wrap-around: if the row ends with a boundary run, check if it
    // connects to the beginning of the same row
    if (runLen > 0) {
      // Check if the start of the row is also non-interior (wrap)
      let startRun = 0;
      for (let x = 0; x < WIDTH; x++) {
        if (boundaryTypes[y * WIDTH + x] !== 0) startRun++;
        else break;
      }
      if (startRun > 0 && startRun < WIDTH) {
        // The last run wraps to the first run; combine them, but remove the
        // first run that was already pushed (if it was)
        // Actually, the first run hasn't been pushed yet because we iterate
        // left to right and only push when we hit a 0. If the entire row
        // starts with boundary cells, the startRun cells were part of the
        // first segment, which ended at some point. That segment was pushed.
        // The current runLen at end-of-row wraps to the start.
        // For simplicity, just push them separately.
        runLengths.push(runLen + startRun);
      } else {
        runLengths.push(runLen);
      }
    }
  }

  if (runLengths.length > 0) {
    runLengths.sort((a, b) => a - b);
    const sum = runLengths.reduce((s, v) => s + v, 0);
    const mean = sum / runLengths.length;
    const median = runLengths[Math.floor(runLengths.length / 2)];
    const p90 = runLengths[Math.floor(runLengths.length * 0.9)];
    const max = runLengths[runLengths.length - 1];
    console.log('\n=== BOUNDARY WIDTH (horizontal scan-line runs of non-interior cells) ===');
    console.log(`  Total runs:  ${runLengths.length}`);
    console.log(`  Mean width:  ${mean.toFixed(2)} cells`);
    console.log(`  Median:      ${median} cells`);
    console.log(`  P90:         ${p90} cells`);
    console.log(`  Max:         ${max} cells`);

    // Width histogram
    const widthBins = [
      { label: '1 cell',    lo: 1, hi: 1, count: 0 },
      { label: '2 cells',   lo: 2, hi: 2, count: 0 },
      { label: '3-4 cells', lo: 3, hi: 4, count: 0 },
      { label: '5-8 cells', lo: 5, hi: 8, count: 0 },
      { label: '9-16 cells',lo: 9, hi: 16, count: 0 },
      { label: '17-32 cells',lo:17, hi: 32, count: 0 },
      { label: '33+ cells', lo: 33, hi: Infinity, count: 0 },
    ];
    for (const len of runLengths) {
      for (const bin of widthBins) {
        if (len >= bin.lo && len <= bin.hi) {
          bin.count++;
          break;
        }
      }
    }
    console.log('\n  Width distribution:');
    const maxBar = 50;
    const maxCount = Math.max(...widthBins.map(b => b.count));
    for (const bin of widthBins) {
      const pct = ((bin.count / runLengths.length) * 100).toFixed(1);
      const barLen = maxCount > 0 ? Math.round((bin.count / maxCount) * maxBar) : 0;
      const bar = '#'.repeat(barLen);
      console.log(`    ${bin.label.padEnd(14)} ${String(bin.count).padStart(7)}  (${pct.padStart(5)}%)  ${bar}`);
    }
  }

  // Isolated boundary cells: non-zero type where ALL 8 neighbors are interior (type 0)
  let isolatedCount = 0;
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = y * WIDTH + x;
      if (boundaryTypes[i] === 0) continue;
      let allInterior = true;
      for (const [dx, dy] of NEIGHBORS) {
        const ni = wrapCoord(x + dx, y + dy);
        if (boundaryTypes[ni] !== 0) {
          allInterior = false;
          break;
        }
      }
      if (allInterior) isolatedCount++;
    }
  }
  console.log(`\n  Isolated boundary cells (non-zero surrounded by all interior): ${isolatedCount}`);
  if (nonInterior > 0) {
    console.log(`    (${((isolatedCount / nonInterior) * 100).toFixed(2)}% of all boundary cells)`);
  }
}

// ---------------------------------------------------------------------------
// 4. Pre-erosion relief analysis & visualization
// ---------------------------------------------------------------------------
function createPreErosionRelief(boundaryTypes, heightMap, runDir) {
  // Separate heights into interior vs boundary cells
  const interiorHeights = [];
  const boundaryHeights = [];
  for (let i = 0; i < TOTAL; i++) {
    const h = heightMap[i];
    if (boundaryTypes[i] === 0) {
      interiorHeights.push(h);
    } else {
      boundaryHeights.push(h);
    }
  }

  // Per-type height stats
  const typeHeights = [[], [], [], []];
  const typeNames = ['interior', 'convergent', 'divergent', 'transform'];
  for (let i = 0; i < TOTAL; i++) {
    const t = boundaryTypes[i];
    if (t >= 0 && t <= 3) typeHeights[t].push(heightMap[i]);
  }

  console.log('\n=== HEIGHT STATISTICS BY BOUNDARY TYPE ===');
  for (let t = 0; t <= 3; t++) {
    const arr = typeHeights[t];
    if (arr.length === 0) continue;
    arr.sort((a, b) => a - b);
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const min = arr[0];
    const max = arr[arr.length - 1];
    const p10 = arr[Math.floor(arr.length * 0.1)];
    const p50 = arr[Math.floor(arr.length * 0.5)];
    const p90 = arr[Math.floor(arr.length * 0.9)];
    console.log(`  ${typeNames[t].padEnd(11)}: n=${String(arr.length).padStart(8)}  min=${min.toFixed(0).padStart(7)}  p10=${p10.toFixed(0).padStart(7)}  median=${p50.toFixed(0).padStart(7)}  mean=${mean.toFixed(0).padStart(7)}  p90=${p90.toFixed(0).padStart(7)}  max=${max.toFixed(0).padStart(7)}`);
  }

  // Height histogram: interior vs boundary
  const histMin = -8000;
  const histMax = 10000;
  const binSize = 500;
  const numBins = Math.ceil((histMax - histMin) / binSize);
  const interiorHist = new Uint32Array(numBins);
  const boundaryHist = new Uint32Array(numBins);

  for (const h of interiorHeights) {
    const bin = clamp(Math.floor((h - histMin) / binSize), 0, numBins - 1);
    interiorHist[bin]++;
  }
  for (const h of boundaryHeights) {
    const bin = clamp(Math.floor((h - histMin) / binSize), 0, numBins - 1);
    boundaryHist[bin]++;
  }

  // Normalize histograms to fractions
  const interiorTotal = interiorHeights.length || 1;
  const boundaryTotal = boundaryHeights.length || 1;

  console.log('\n=== HEIGHT HISTOGRAM: INTERIOR vs BOUNDARY ===');
  console.log('  Elevation range      Interior%  Boundary%  Comparison');
  const maxBarLen = 40;
  for (let b = 0; b < numBins; b++) {
    const lo = histMin + b * binSize;
    const hi = lo + binSize;
    const iPct = (interiorHist[b] / interiorTotal) * 100;
    const bPct = (boundaryHist[b] / boundaryTotal) * 100;
    if (iPct < 0.05 && bPct < 0.05) continue; // skip empty bins
    const label = `${String(lo).padStart(6)}..${String(hi).padStart(6)}m`;
    const iBar = '#'.repeat(Math.round((iPct / 30) * maxBarLen));
    const bBar = '='.repeat(Math.round((bPct / 30) * maxBarLen));
    console.log(`  ${label}  ${iPct.toFixed(1).padStart(6)}%  ${bPct.toFixed(1).padStart(6)}%  I:${iBar}`);
    if (bBar.length > 0) {
      console.log(`  ${''.padEnd(22)}${''.padEnd(8)}${''.padEnd(8)}  B:${bBar}`);
    }
  }

  // Create pre-erosion relief visualization
  // Color scheme:
  //   - Interior cells: blue-to-white by height (proxy for "pre-erosion base level")
  //   - Boundary cells: red-to-yellow by height (showing tectonic uplift/subsidence)
  const outPath = path.join(runDir, 'pre_erosion_relief.jpg');
  console.log(`\nWriting pre-erosion relief map: ${outPath}`);

  // Find height range for normalization
  let minH = Infinity, maxH = -Infinity;
  for (let i = 0; i < TOTAL; i++) {
    if (heightMap[i] < minH) minH = heightMap[i];
    if (heightMap[i] > maxH) maxH = heightMap[i];
  }
  const range = maxH - minH || 1;

  writeJpg(outPath, WIDTH, HEIGHT, (i) => {
    const h = heightMap[i];
    const t = (h - minH) / range; // 0..1 normalized
    const btype = boundaryTypes[i];

    if (btype === 0) {
      // Interior: dark blue (low) -> light cyan (high)
      return [
        Math.round(20 + 60 * t),
        Math.round(20 + 100 * t),
        Math.round(80 + 175 * t),
      ];
    } else if (btype === 1) {
      // Convergent: dark red -> bright yellow
      return [
        Math.round(100 + 155 * t),
        Math.round(20 + 200 * t),
        Math.round(10 + 30 * t),
      ];
    } else if (btype === 2) {
      // Divergent: dark purple -> bright magenta
      return [
        Math.round(60 + 150 * t),
        Math.round(10 + 40 * t),
        Math.round(100 + 155 * t),
      ];
    } else {
      // Transform: dark teal -> bright green
      return [
        Math.round(10 + 60 * t),
        Math.round(80 + 175 * t),
        Math.round(40 + 80 * t),
      ];
    }
  });

  return outPath;
}

// ---------------------------------------------------------------------------
// 5. Plate boundary co-occurrence analysis
// ---------------------------------------------------------------------------
function printPlateStats(plateMap, boundaryTypes) {
  const plateIds = new Set();
  for (let i = 0; i < TOTAL; i++) {
    plateIds.add(plateMap[i]);
  }
  console.log(`\n=== PLATE STATISTICS ===`);
  console.log(`  Total plates: ${plateIds.size}`);

  // Per-plate cell counts and boundary cell counts
  const plateCellCount = new Map();
  const plateBoundaryCells = new Map();
  for (let i = 0; i < TOTAL; i++) {
    const pid = plateMap[i];
    plateCellCount.set(pid, (plateCellCount.get(pid) || 0) + 1);
    if (boundaryTypes[i] !== 0) {
      plateBoundaryCells.set(pid, (plateBoundaryCells.get(pid) || 0) + 1);
    }
  }

  // Sort by size descending
  const plates = [...plateCellCount.entries()].sort((a, b) => b[1] - a[1]);
  console.log(`\n  Top plates by size:`);
  console.log(`  ${'ID'.padStart(5)}  ${'Cells'.padStart(9)}  ${'%Area'.padStart(6)}  ${'Boundary%'.padStart(10)}`);
  const showCount = Math.min(20, plates.length);
  for (let k = 0; k < showCount; k++) {
    const [pid, cnt] = plates[k];
    const bCnt = plateBoundaryCells.get(pid) || 0;
    const areaPct = ((cnt / TOTAL) * 100).toFixed(2);
    const bPct = ((bCnt / cnt) * 100).toFixed(1);
    console.log(`  ${String(pid).padStart(5)}  ${String(cnt).padStart(9)}  ${areaPct.padStart(6)}  ${bPct.padStart(10)}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function main() {
  const runDir = process.argv[2] || DEFAULT_RUN_DIR;

  console.log(`=== debug-boundaries.cjs ===`);
  console.log(`Run directory: ${runDir}\n`);

  // Read boundary_types.i8
  const boundaryPath = path.join(runDir, 'boundary_types.i8');
  if (!fs.existsSync(boundaryPath)) {
    console.error(`boundary_types.i8 not found at: ${boundaryPath}`);
    process.exit(1);
  }
  console.log(`Reading boundary_types.i8...`);
  const boundaryTypes = readInt8(boundaryPath);
  console.log(`  ${boundaryTypes.length} values (expected ${TOTAL})`);
  if (boundaryTypes.length !== TOTAL) {
    console.error(`ERROR: Size mismatch! Expected ${TOTAL}, got ${boundaryTypes.length}`);
    process.exit(1);
  }

  // Read plate_map.i16
  const platePath = path.join(runDir, 'plate_map.i16');
  if (!fs.existsSync(platePath)) {
    console.error(`plate_map.i16 not found at: ${platePath}`);
    process.exit(1);
  }
  console.log(`Reading plate_map.i16...`);
  const plateMap = readInt16LE(platePath);
  console.log(`  ${plateMap.length} values (expected ${TOTAL})`);
  if (plateMap.length !== TOTAL) {
    console.error(`ERROR: Size mismatch! Expected ${TOTAL}, got ${plateMap.length}`);
    process.exit(1);
  }

  // Read height_map.f32
  const heightPath = path.join(runDir, 'height_map.f32');
  if (!fs.existsSync(heightPath)) {
    console.error(`height_map.f32 not found at: ${heightPath}`);
    process.exit(1);
  }
  console.log(`Reading height_map.f32...`);
  const heightMap = readFloat32(heightPath);
  console.log(`  ${heightMap.length} values (expected ${TOTAL})`);
  if (heightMap.length !== TOTAL) {
    console.error(`ERROR: Size mismatch! Expected ${TOTAL}, got ${heightMap.length}`);
    process.exit(1);
  }

  // --- Visualization 1: Boundary types ---
  createBoundaryTypesVis(boundaryTypes, runDir);

  // --- Visualization 2: Boundary width ---
  console.log('\nComputing neighbor difference map...');
  const diffMap = computeNeighborDiffMap(boundaryTypes, plateMap);
  createBoundaryWidthVis(diffMap, runDir);

  // --- Statistics ---
  printBoundaryStats(boundaryTypes, diffMap);
  printPlateStats(plateMap, boundaryTypes);

  // --- Visualization 3: Pre-erosion relief ---
  createPreErosionRelief(boundaryTypes, heightMap, runDir);

  console.log('\nDone. All outputs saved to:', runDir);
}

main();
