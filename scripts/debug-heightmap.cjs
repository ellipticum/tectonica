/**
 * debug-heightmap.cjs
 *
 * Reads a raw height_map.f32 and produces two diagnostic visualizations:
 *   1) flat_height.jpg  — flat-colored elevation bands (no hillshade)
 *   2) slope_map_vis.jpg — per-cell slope magnitude for land pixels
 *
 * Also prints elevation histogram and slope statistics to stdout.
 *
 * Usage:
 *   node scripts/debug-heightmap.cjs [path-to-run-directory]
 *
 * If no path is given, uses the default run directory hard-coded below.
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

// ---------------------------------------------------------------------------
// Flat elevation color (no hillshade)
// ---------------------------------------------------------------------------
function flatElevationColor(h) {
  if (h <= 0) {
    // Ocean — dark blue
    return [15, 30, 100];
  }
  if (h <= 200) {
    // Low land — dark green
    return [20, 100, 40];
  }
  if (h <= 500) {
    // Mid-low — green
    return [50, 160, 50];
  }
  if (h <= 1500) {
    // Mid — light green / yellow blend
    const t = (h - 500) / 1000;
    return [
      50 + Math.round(170 * t),   // 50 -> 220
      160 + Math.round(40 * t),   // 160 -> 200
      50 - Math.round(30 * t),    // 50 -> 20
    ];
  }
  if (h <= 3000) {
    // High — orange / brown
    const t = (h - 1500) / 1500;
    return [
      220 - Math.round(100 * t),  // 220 -> 120
      200 - Math.round(130 * t),  // 200 -> 70
      20 + Math.round(30 * t),    // 20 -> 50
    ];
  }
  // Very high — white
  const t = clamp((h - 3000) / 2000, 0, 1);
  return [
    120 + Math.round(135 * t),  // 120 -> 255
    70 + Math.round(185 * t),   // 70 -> 255
    50 + Math.round(205 * t),   // 50 -> 255
  ];
}

// ---------------------------------------------------------------------------
// Compute slope map (max abs difference to 4 cardinal neighbors)
// ---------------------------------------------------------------------------
function computeSlopeMap(heightMap, width, height) {
  const slopes = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const h = heightMap[i];

      // Skip ocean for slope coloring, but still compute
      let maxDiff = 0;

      // Left (wrap horizontally — equirectangular)
      const xL = (x - 1 + width) % width;
      maxDiff = Math.max(maxDiff, Math.abs(h - heightMap[y * width + xL]));

      // Right (wrap)
      const xR = (x + 1) % width;
      maxDiff = Math.max(maxDiff, Math.abs(h - heightMap[y * width + xR]));

      // Up
      if (y > 0) {
        maxDiff = Math.max(maxDiff, Math.abs(h - heightMap[(y - 1) * width + x]));
      }

      // Down
      if (y < height - 1) {
        maxDiff = Math.max(maxDiff, Math.abs(h - heightMap[(y + 1) * width + x]));
      }

      slopes[i] = maxDiff;
    }
  }
  return slopes;
}

// ---------------------------------------------------------------------------
// Slope visualization color
// ---------------------------------------------------------------------------
function slopeColor(slope, isOcean) {
  if (isOcean) return [0, 0, 0];

  // Map slope 0..2000 to gray..white with a sqrt curve for better contrast
  const maxVis = 2000;
  const t = Math.sqrt(clamp(slope / maxVis, 0, 1));
  const v = Math.round(40 + 215 * t); // 40 (dark gray) to 255 (white)
  return [v, v, v];
}

// ---------------------------------------------------------------------------
// Elevation histogram
// ---------------------------------------------------------------------------
function printElevationHistogram(heightMap) {
  const bins = [
    { label: '< -4000m (deep ocean)', lo: -Infinity, hi: -4000, count: 0 },
    { label: '-4000 to -2000m',       lo: -4000,     hi: -2000, count: 0 },
    { label: '-2000 to -500m',        lo: -2000,     hi: -500,  count: 0 },
    { label: '-500 to 0m (shallow)',   lo: -500,      hi: 0,     count: 0 },
    { label: '0 to 200m',             lo: 0,         hi: 200,   count: 0 },
    { label: '200 to 500m',           lo: 200,       hi: 500,   count: 0 },
    { label: '500 to 1000m',          lo: 500,       hi: 1000,  count: 0 },
    { label: '1000 to 1500m',         lo: 1000,      hi: 1500,  count: 0 },
    { label: '1500 to 2000m',         lo: 1500,      hi: 2000,  count: 0 },
    { label: '2000 to 3000m',         lo: 2000,      hi: 3000,  count: 0 },
    { label: '3000 to 5000m',         lo: 3000,      hi: 5000,  count: 0 },
    { label: '5000m+',                lo: 5000,       hi: Infinity, count: 0 },
  ];

  for (let i = 0; i < heightMap.length; i++) {
    const h = heightMap[i];
    for (const bin of bins) {
      if (h > bin.lo && h <= bin.hi) {
        bin.count++;
        break;
      }
    }
    // Edge case: exactly at boundary of first bin
    if (h <= bins[0].hi && h <= bins[0].lo) {
      // won't happen with -Infinity, handled above
    }
  }

  // Handle h <= -Infinity edge (not real, but just in case)
  // Actually, -Infinity < h always, so first bin catches everything below -4000

  const total = heightMap.length;
  console.log('\n=== ELEVATION HISTOGRAM ===');
  const maxBar = 50;
  const maxCount = Math.max(...bins.map(b => b.count));
  for (const bin of bins) {
    const pct = ((bin.count / total) * 100).toFixed(1);
    const barLen = Math.round((bin.count / maxCount) * maxBar);
    const bar = '#'.repeat(barLen);
    console.log(`  ${bin.label.padEnd(28)} ${String(bin.count).padStart(8)}  (${pct.padStart(5)}%)  ${bar}`);
  }
}

// ---------------------------------------------------------------------------
// Slope statistics (land only)
// ---------------------------------------------------------------------------
function printSlopeStats(heightMap, slopes) {
  let landCount = 0;
  let slopeOver500 = 0;
  let slopeOver1000 = 0;
  let slopeOver2000 = 0;
  let slopeOver3000 = 0;
  let maxSlope = 0;
  let sumSlope = 0;

  // Also build slope histogram for land
  const slopeBins = [
    { label: '0-50 m/cell',     lo: 0,    hi: 50,   count: 0 },
    { label: '50-100 m/cell',   lo: 50,   hi: 100,  count: 0 },
    { label: '100-200 m/cell',  lo: 100,  hi: 200,  count: 0 },
    { label: '200-500 m/cell',  lo: 200,  hi: 500,  count: 0 },
    { label: '500-1000 m/cell', lo: 500,  hi: 1000, count: 0 },
    { label: '1000-2000 m/cell',lo: 1000, hi: 2000, count: 0 },
    { label: '2000-3000 m/cell',lo: 2000, hi: 3000, count: 0 },
    { label: '3000+ m/cell',    lo: 3000, hi: Infinity, count: 0 },
  ];

  for (let i = 0; i < heightMap.length; i++) {
    if (heightMap[i] <= 0) continue;
    landCount++;
    const s = slopes[i];
    sumSlope += s;
    if (s > maxSlope) maxSlope = s;
    if (s > 500) slopeOver500++;
    if (s > 1000) slopeOver1000++;
    if (s > 2000) slopeOver2000++;
    if (s > 3000) slopeOver3000++;

    for (const bin of slopeBins) {
      if (s >= bin.lo && s < bin.hi) {
        bin.count++;
        break;
      }
    }
    // catch exactly at 3000+
    if (s >= 3000) {
      // already counted in last bin via Infinity
    }
  }

  console.log('\n=== SLOPE STATISTICS (land cells only) ===');
  console.log(`  Total land cells:    ${landCount}`);
  console.log(`  Mean slope:          ${(sumSlope / landCount).toFixed(1)} m/cell`);
  console.log(`  Max slope:           ${maxSlope.toFixed(1)} m/cell`);
  console.log(`  Slope > 500 m/cell:  ${slopeOver500} (${((slopeOver500 / landCount) * 100).toFixed(2)}%) -- sharp features`);
  console.log(`  Slope > 1000 m/cell: ${slopeOver1000} (${((slopeOver1000 / landCount) * 100).toFixed(2)}%) -- very sharp`);
  console.log(`  Slope > 2000 m/cell: ${slopeOver2000} (${((slopeOver2000 / landCount) * 100).toFixed(2)}%)`);
  console.log(`  Slope > 3000 m/cell: ${slopeOver3000} (${((slopeOver3000 / landCount) * 100).toFixed(2)}%)`);

  console.log('\n=== LAND SLOPE HISTOGRAM ===');
  const maxBar = 50;
  const maxCount = Math.max(...slopeBins.map(b => b.count));
  for (const bin of slopeBins) {
    const pct = landCount > 0 ? ((bin.count / landCount) * 100).toFixed(1) : '0.0';
    const barLen = maxCount > 0 ? Math.round((bin.count / maxCount) * maxBar) : 0;
    const bar = '#'.repeat(barLen);
    console.log(`  ${bin.label.padEnd(22)} ${String(bin.count).padStart(8)}  (${pct.padStart(5)}%)  ${bar}`);
  }
}

// ---------------------------------------------------------------------------
// Find top-N steepest locations (for manual inspection)
// ---------------------------------------------------------------------------
function printSteepestLocations(heightMap, slopes, width, topN = 20) {
  const landSlopes = [];
  for (let i = 0; i < heightMap.length; i++) {
    if (heightMap[i] <= 0) continue;
    landSlopes.push({ i, slope: slopes[i] });
  }
  landSlopes.sort((a, b) => b.slope - a.slope);

  console.log(`\n=== TOP ${topN} STEEPEST LAND CELLS ===`);
  console.log('  (x, y)       slope m/cell   elevation m');
  for (let k = 0; k < Math.min(topN, landSlopes.length); k++) {
    const { i, slope } = landSlopes[k];
    const x = i % width;
    const y = Math.floor(i / width);
    const h = heightMap[i];
    console.log(`  (${String(x).padStart(4)}, ${String(y).padStart(4)})   ${slope.toFixed(1).padStart(10)}   ${h.toFixed(1).padStart(10)}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function main() {
  const runDir = process.argv[2] || DEFAULT_RUN_DIR;
  const heightFile = path.join(runDir, 'height_map.f32');

  if (!fs.existsSync(heightFile)) {
    console.error(`height_map.f32 not found at: ${heightFile}`);
    process.exit(1);
  }

  console.log(`Reading height map from: ${heightFile}`);
  console.log(`Expected: ${WIDTH}x${HEIGHT} = ${TOTAL} float32 values = ${TOTAL * 4} bytes`);

  const buf = fs.readFileSync(heightFile);
  console.log(`File size: ${buf.length} bytes`);

  if (buf.length !== TOTAL * 4) {
    console.error(`ERROR: Expected ${TOTAL * 4} bytes, got ${buf.length}`);
    process.exit(1);
  }

  const heightMap = new Float32Array(buf.buffer, buf.byteOffset, TOTAL);

  // Basic stats
  let minH = Infinity, maxH = -Infinity;
  let oceanCount = 0;
  for (let i = 0; i < TOTAL; i++) {
    const h = heightMap[i];
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
    if (h <= 0) oceanCount++;
  }
  const landCount = TOTAL - oceanCount;

  console.log(`\nHeight range: ${minH.toFixed(1)}m to ${maxH.toFixed(1)}m`);
  console.log(`Ocean cells: ${oceanCount} (${((oceanCount / TOTAL) * 100).toFixed(1)}%)`);
  console.log(`Land cells:  ${landCount} (${((landCount / TOTAL) * 100).toFixed(1)}%)`);

  // Elevation histogram
  printElevationHistogram(heightMap);

  // Compute slopes
  console.log('\nComputing slope map...');
  const slopes = computeSlopeMap(heightMap, WIDTH, HEIGHT);

  // Slope stats
  printSlopeStats(heightMap, slopes);

  // Steepest locations
  printSteepestLocations(heightMap, slopes, WIDTH, 20);

  // --- Visualization A: Flat elevation coloring ---
  const flatPath = path.join(runDir, 'flat_height.jpg');
  console.log(`\nWriting flat elevation map: ${flatPath}`);
  writeJpg(flatPath, WIDTH, HEIGHT, (i, x, y) => {
    return flatElevationColor(heightMap[i]);
  });

  // --- Visualization B: Slope map ---
  const slopePath = path.join(runDir, 'slope_map_vis.jpg');
  console.log(`Writing slope map: ${slopePath}`);
  writeJpg(slopePath, WIDTH, HEIGHT, (i, x, y) => {
    return slopeColor(slopes[i], heightMap[i] <= 0);
  });

  console.log('\nDone. Check the output images to compare with height_preview.jpg.');
  console.log('If flat_height.jpg shows smooth continents without spiky patterns,');
  console.log('the dendritic artifacts are from the hillshade algorithm.');
  console.log('If flat_height.jpg also shows spiky edges, the raw data has the problem.');
}

main();
