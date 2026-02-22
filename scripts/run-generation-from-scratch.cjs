const fs = require('fs');
const path = require('path');
const jpeg = require('jpeg-js');

const pkgPath = path.resolve(__dirname, '..', 'generations', 'pkg', 'planet_engine.js');
if (!fs.existsSync(pkgPath)) {
  console.error('WASM pkg not found at generations/pkg. Build it first with: wasm-pack build rust/planet_engine --target nodejs --out-dir ../../generations/pkg --release');
  process.exit(1);
}

const { run_simulation_with_progress } = require(pkgPath);

const WIDTH = 2048;
const HEIGHT = 1024;

function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}

function sampleStops(stops, t) {
  const k = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    const a = stops[i - 1];
    const b = stops[i];
    if (k <= b.t) {
      const local = (k - a.t) / Math.max(1e-6, b.t - a.t);
      return [
        Math.round(a.color[0] + (b.color[0] - a.color[0]) * local),
        Math.round(a.color[1] + (b.color[1] - a.color[1]) * local),
        Math.round(a.color[2] + (b.color[2] - a.color[2]) * local),
      ];
    }
  }
  return stops[stops.length - 1].color;
}

const OCEAN_STOPS = [
  { t: 0, color: [176, 206, 232] },
  { t: 0.18, color: [146, 184, 220] },
  { t: 0.45, color: [95, 145, 198] },
  { t: 0.72, color: [48, 93, 155] },
  { t: 1, color: [15, 40, 95] },
];

const LAND_STOPS = [
  { t: 0, color: [101, 196, 108] },
  { t: 0.2, color: [168, 214, 112] },
  { t: 0.42, color: [230, 213, 119] },
  { t: 0.62, color: [216, 164, 95] },
  { t: 0.8, color: [190, 116, 90] },
  { t: 0.9, color: [210, 161, 166] },
  { t: 1, color: [247, 245, 242] },
];

function heightColor(value, min, max) {
  if (value < 0 && max > 0) {
    const t = Math.pow(clamp(-value / Math.max(1, -min), 0, 1), 0.68);
    return sampleStops(OCEAN_STOPS, t);
  }
  if (max <= 0) {
    return sampleStops(OCEAN_STOPS, 1);
  }
  const t = clamp(value / Math.max(1, max), 0, 1);
  return sampleStops(LAND_STOPS, t);
}

function biomeColor(id) {
  switch (id) {
    case 0: return [17, 42, 82];
    case 1: return [204, 213, 238];
    case 2: return [34, 139, 87];
    case 3: return [21, 109, 61];
    case 4: return [196, 168, 84];
    case 5: return [219, 179, 94];
    case 6: return [132, 173, 93];
    case 7: return [130, 106, 74];
    case 8: return [63, 104, 61];
    default: return [255, 255, 255];
  }
}

function plateColor(plateId) {
  const id = Math.max(0, plateId | 0);
  const hue = ((id * 37) % 360) * (Math.PI / 180);
  const r = Math.round(127 + 100 * Math.sin(hue + 0));
  const g = Math.round(127 + 100 * Math.sin(hue + 2.094));
  const b = Math.round(127 + 100 * Math.sin(hue + 4.188));
  return [r, g, b];
}

function writeTypedArray(filePath, arr) {
  fs.writeFileSync(filePath, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
}

function writeBmp(filePath, width, height, pixelAt) {
  const rowStride = ((width * 3 + 3) & ~3);
  const pixelDataSize = rowStride * height;
  const fileSize = 54 + pixelDataSize;
  const out = Buffer.alloc(fileSize);

  out.write('BM', 0, 2, 'ascii');
  out.writeUInt32LE(fileSize, 2);
  out.writeUInt32LE(54, 10);
  out.writeUInt32LE(40, 14);
  out.writeInt32LE(width, 18);
  out.writeInt32LE(height, 22);
  out.writeUInt16LE(1, 26);
  out.writeUInt16LE(24, 28);
  out.writeUInt32LE(0, 30);
  out.writeUInt32LE(pixelDataSize, 34);
  out.writeInt32LE(2835, 38);
  out.writeInt32LE(2835, 42);

  let off = 54;
  for (let y = height - 1; y >= 0; y--) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const [r, g, b] = pixelAt(i, x, y);
      out[off++] = b;
      out[off++] = g;
      out[off++] = r;
    }
    while ((off - 54) % rowStride !== 0) {
      out[off++] = 0;
    }
  }

  fs.writeFileSync(filePath, out);
}

function writeJpg(filePath, width, height, pixelAt, quality = 92) {
  const rgba = Buffer.alloc(width * height * 4);
  let off = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const [r, g, b] = pixelAt(i, x, y);
      rgba[off++] = r;
      rgba[off++] = g;
      rgba[off++] = b;
      rgba[off++] = 255;
    }
  }

  const encoded = jpeg.encode({ data: rgba, width, height }, quality);
  fs.writeFileSync(filePath, encoded.data);
}

function run() {
  const seed = Number(process.env.SEED || Math.floor(Math.random() * 2_147_483_647));
  const generationPreset = process.env.PRESET || 'detailed';

  const config = {
    seed,
    planet: {
      radiusKm: 6371,
      gravity: 9.81,
      density: 5510,
      rotationHours: 24,
      axialTiltDeg: 23.5,
      eccentricity: 0.016,
      atmosphereBar: 1,
      oceanPercent: 67,
    },
    tectonics: {
      plateCount: 11,
      plateSpeedCmPerYear: 5,
      mantleHeat: 55,
    },
    events: [],
    generationPreset,
  };

  let last = -1;
  const t0 = Date.now();
  const result = run_simulation_with_progress(config, 'global', (p) => {
    const k = Math.max(0, Math.min(99, Math.floor(p)));
    if (k !== last) {
      process.stdout.write(`\rGenerating: ${k}%`);
      last = k;
    }
  });
  process.stdout.write('\rGenerating: 100%\n');

  const width = result.width;
  const height = result.height;
  if (width !== WIDTH || height !== HEIGHT) {
    console.warn(`Unexpected map size: ${width}x${height}`);
  }

  const heightMap = result.heightMap();
  const slopeMap = result.slopeMap();
  const riverMap = result.riverMap();
  const lakeMap = result.lakeMap();
  const flowDirection = result.flowDirection();
  const flowAccumulation = result.flowAccumulation();
  const temperatureMap = result.temperatureMap();
  const precipitationMap = result.precipitationMap();
  const biomeMap = result.biomeMap();
  const settlementMap = result.settlementMap();
  const plates = result.plates();
  const boundaryTypes = result.boundaryTypes();

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const runDir = path.resolve(__dirname, '..', 'generations', 'runs', `${stamp}_seed-${seed}`);
  fs.mkdirSync(runDir, { recursive: true });

  const meta = {
    createdAt: new Date().toISOString(),
    durationMs: Date.now() - t0,
    seed,
    generationPreset,
    config,
    width,
    height,
    seaLevel: result.seaLevel(),
    recomputedLayers: Array.from(result.recomputedLayers()),
    stats: {
      minHeight: result.minHeight(),
      maxHeight: result.maxHeight(),
      minSlope: result.minSlope(),
      maxSlope: result.maxSlope(),
      minTemperature: result.minTemperature(),
      maxTemperature: result.maxTemperature(),
      minPrecipitation: result.minPrecipitation(),
      maxPrecipitation: result.maxPrecipitation(),
    },
  };

  fs.writeFileSync(path.join(runDir, 'meta.json'), JSON.stringify(meta, null, 2));

  writeTypedArray(path.join(runDir, 'height_map.f32'), heightMap);
  writeTypedArray(path.join(runDir, 'slope_map.f32'), slopeMap);
  writeTypedArray(path.join(runDir, 'river_map.f32'), riverMap);
  writeTypedArray(path.join(runDir, 'lake_map.u8'), lakeMap);
  writeTypedArray(path.join(runDir, 'flow_direction.i32'), flowDirection);
  writeTypedArray(path.join(runDir, 'flow_accumulation.f32'), flowAccumulation);
  writeTypedArray(path.join(runDir, 'temperature_map.f32'), temperatureMap);
  writeTypedArray(path.join(runDir, 'precipitation_map.f32'), precipitationMap);
  writeTypedArray(path.join(runDir, 'biome_map.u8'), biomeMap);
  writeTypedArray(path.join(runDir, 'settlement_map.f32'), settlementMap);
  writeTypedArray(path.join(runDir, 'plate_map.i16'), plates);
  writeTypedArray(path.join(runDir, 'boundary_types.i8'), boundaryTypes);

  const minH = meta.stats.minHeight;
  const maxH = meta.stats.maxHeight;

  writeBmp(path.join(runDir, 'height_preview.bmp'), width, height, (i) => {
    return heightColor(heightMap[i], minH, maxH);
  });
  writeJpg(path.join(runDir, 'height_preview.jpg'), width, height, (i) => {
    return heightColor(heightMap[i], minH, maxH);
  });

  writeBmp(path.join(runDir, 'plates_preview.bmp'), width, height, (i) => {
    const base = plateColor(plates[i]);
    if (boundaryTypes[i] === 1) return [255, 175, 175];
    if (boundaryTypes[i] === 2) return [180, 220, 255];
    if (boundaryTypes[i] === 3) return [235, 220, 170];
    return base;
  });
  writeJpg(path.join(runDir, 'plates_preview.jpg'), width, height, (i) => {
    const base = plateColor(plates[i]);
    if (boundaryTypes[i] === 1) return [255, 175, 175];
    if (boundaryTypes[i] === 2) return [180, 220, 255];
    if (boundaryTypes[i] === 3) return [235, 220, 170];
    return base;
  });

  writeBmp(path.join(runDir, 'biomes_preview.bmp'), width, height, (i) => biomeColor(biomeMap[i]));
  writeJpg(path.join(runDir, 'biomes_preview.jpg'), width, height, (i) => biomeColor(biomeMap[i]));

  result.free();

  console.log(`Saved generation to: ${runDir}`);
}

run();
