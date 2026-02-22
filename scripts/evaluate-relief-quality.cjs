const fs = require("fs");
const path = require("path");

const WIDTH = 2048;
const HEIGHT = 1024;
const SIZE = WIDTH * HEIGHT;

function listRunDirs() {
  const root = path.resolve(__dirname, "..", "generations", "runs");
  if (!fs.existsSync(root)) {
    throw new Error(`Runs directory not found: ${root}`);
  }
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function readFloat32(filePath) {
  const buf = fs.readFileSync(filePath);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function readInt8(filePath) {
  const buf = fs.readFileSync(filePath);
  return new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

function quantileSorted(sorted, q) {
  if (!sorted.length) return 0;
  const t = Math.max(0, Math.min(1, q)) * (sorted.length - 1);
  const lo = Math.floor(t);
  const hi = Math.min(sorted.length - 1, lo + 1);
  const k = t - lo;
  return sorted[lo] * (1 - k) + sorted[hi] * k;
}

function sphericalIndex(x, y) {
  let sx = x;
  let sy = y;
  while (sy < 0 || sy >= HEIGHT) {
    if (sy < 0) {
      sy = -sy - 1;
      sx += WIDTH / 2;
    } else {
      sy = 2 * HEIGHT - sy - 1;
      sx += WIDTH / 2;
    }
  }
  sx %= WIDTH;
  if (sx < 0) sx += WIDTH;
  return sy * WIDTH + sx;
}

function evaluateRun(runDir) {
  const metaPath = path.join(runDir, "meta.json");
  const heightPath = path.join(runDir, "height_map.f32");
  const boundaryPath = path.join(runDir, "boundary_types.i8");
  if (!fs.existsSync(metaPath) || !fs.existsSync(heightPath) || !fs.existsSync(boundaryPath)) {
    return null;
  }

  const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
  const heights = readFloat32(heightPath);
  const boundaries = readInt8(boundaryPath);
  if (heights.length !== SIZE || boundaries.length !== SIZE) {
    return null;
  }

  const land = [];
  let oceanCells = 0;
  let boundaryCells = 0;
  for (let i = 0; i < SIZE; i++) {
    if (heights[i] > 0) {
      land.push(heights[i]);
    } else {
      oceanCells++;
    }
    if (boundaries[i] !== 0) boundaryCells++;
  }
  land.sort((a, b) => a - b);
  const p90 = quantileSorted(land, 0.9);
  const p98 = quantileSorted(land, 0.98);

  let high90 = 0;
  let high98 = 0;
  let high90Boundary = 0;
  let high98Boundary = 0;
  let linearness90 = 0;
  let linearness98 = 0;

  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = y * WIDTH + x;
      const h = heights[i];
      if (h <= p90) continue;

      high90++;
      if (boundaries[i] !== 0) high90Boundary++;

      const d0 = Math.abs(
        heights[sphericalIndex(x + 1, y)] - heights[sphericalIndex(x - 1, y)],
      );
      const d1 = Math.abs(
        heights[sphericalIndex(x, y + 1)] - heights[sphericalIndex(x, y - 1)],
      );
      const d2 = Math.abs(
        heights[sphericalIndex(x + 1, y + 1)] - heights[sphericalIndex(x - 1, y - 1)],
      );
      const d3 = Math.abs(
        heights[sphericalIndex(x + 1, y - 1)] - heights[sphericalIndex(x - 1, y + 1)],
      );
      const max = Math.max(d0, d1, d2, d3);
      const sum = d0 + d1 + d2 + d3 + 1e-6;
      linearness90 += (max * 4 - sum) / sum;

      if (h > p98) {
        high98++;
        if (boundaries[i] !== 0) high98Boundary++;
        linearness98 += (max * 4 - sum) / sum;
      }
    }
  }

  return {
    run: path.basename(runDir),
    durationMs: meta.durationMs,
    minHeight: meta.stats.minHeight,
    maxHeight: meta.stats.maxHeight,
    landCells: land.length,
    oceanCells,
    boundaryCoverage: boundaryCells / SIZE,
    p90,
    p98,
    high90BoundaryShare: high90 ? high90Boundary / high90 : 0,
    high98BoundaryShare: high98 ? high98Boundary / high98 : 0,
    linearness90: high90 ? linearness90 / high90 : 0,
    linearness98: high98 ? linearness98 / high98 : 0,
  };
}

function main() {
  const input = process.argv.slice(2);
  let runDirs;
  if (input.length > 0) {
    runDirs = input.map((arg) =>
      path.isAbsolute(arg)
        ? arg
        : path.resolve(__dirname, "..", "generations", "runs", arg),
    );
  } else {
    const all = listRunDirs();
    runDirs = all.slice(-5);
  }

  const rows = runDirs.map(evaluateRun).filter(Boolean);
  if (!rows.length) {
    console.error("No valid run directories found.");
    process.exit(1);
  }

  console.log(JSON.stringify(rows, null, 2));
}

main();

