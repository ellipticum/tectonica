const fs = require('fs');
const path = require('path');
const jpeg = require('jpeg-js');

const pkgPath = path.resolve(__dirname, '..', 'generations', 'pkg', 'planet_engine.js');
if (!fs.existsSync(pkgPath)) {
  console.error('WASM pkg not found at generations/pkg. Build it first with: wasm-pack build rust/planet_engine --target nodejs --out-dir ../../generations/pkg --release');
  process.exit(1);
}

const { run_simulation_with_progress } = require(pkgPath);

const EXPECTED_SIZE = {
  planet: { width: 2048, height: 1024 },
  island: { width: 1024, height: 512 },
};

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
  { t: 0, color: [220, 236, 250] },
  { t: 0.14, color: [188, 216, 243] },
  { t: 0.3, color: [150, 190, 232] },
  { t: 0.5, color: [108, 156, 218] },
  { t: 0.72, color: [70, 120, 195] },
  { t: 0.88, color: [46, 87, 169] },
  { t: 1, color: [30, 61, 143] },
];

const LAND_STOPS = [
  { t: 0, color: [4, 104, 64] },       // 0 m
  { t: 0.118, color: [36, 129, 53] },  // 200 m
  { t: 0.294, color: [215, 179, 95] }, // 500 m
  { t: 0.471, color: [147, 51, 10] },  // 800 m
  { t: 0.706, color: [99, 96, 94] },   // 1200 m
  { t: 0.824, color: [219, 218, 218] }, // 1400 m
  { t: 0.882, color: [253, 253, 251] }, // 1500 m
  { t: 1, color: [247, 246, 244] },    // 1700 m
];

function estimateLandToneRange(heightMap, maxHeight) {
  if (maxHeight <= 0) {
    return { minRef: 0, maxRef: 1 };
  }
  const bins = 2048;
  const hist = new Uint32Array(bins);
  const denom = Math.max(1, maxHeight);
  let landCount = 0;
  for (let i = 0; i < heightMap.length; i++) {
    const h = heightMap[i];
    if (h <= 0) continue;
    const t = clamp(h / denom, 0, 1);
    const bin = Math.min(bins - 1, Math.floor(t * (bins - 1)));
    hist[bin] += 1;
    landCount += 1;
  }

  if (landCount < 16) {
    return { minRef: 0, maxRef: denom };
  }

  const qLow = Math.floor(landCount * 0.02);
  const qHigh = Math.floor(landCount * 0.98);

  let acc = 0;
  let lowBin = 0;
  for (let i = 0; i < bins; i++) {
    acc += hist[i] || 0;
    if (acc >= qLow) {
      lowBin = i;
      break;
    }
  }

  acc = 0;
  let highBin = bins - 1;
  for (let i = 0; i < bins; i++) {
    acc += hist[i] || 0;
    if (acc >= qHigh) {
      highBin = i;
      break;
    }
  }

  let minRef = (lowBin / (bins - 1)) * denom;
  let maxRef = (highBin / (bins - 1)) * denom;
  if (maxRef - minRef < 120) {
    minRef = 0;
    maxRef = denom;
  }
  return { minRef, maxRef };
}

function landHillshade(heightMap, width, height, x, y, wrapX = true) {
  const yUp = Math.max(0, y - 1);
  const yDown = Math.min(height - 1, y + 1);
  const xLeft = wrapX ? (x - 1 + width) % width : Math.max(0, x - 1);
  const xRight = wrapX ? (x + 1) % width : Math.min(width - 1, x + 1);
  const left = heightMap[y * width + xLeft] || 0;
  const right = heightMap[y * width + xRight] || 0;
  const up = heightMap[yUp * width + x] || 0;
  const down = heightMap[yDown * width + x] || 0;

  const dzdx = (right - left) * 0.5;
  const dzdy = (down - up) * 0.5;
  const k = 1 / 1800;
  let nx = -dzdx * k;
  let ny = -dzdy * k;
  let nz = 1;
  const nLen = Math.hypot(nx, ny, nz) || 1;
  nx /= nLen;
  ny /= nLen;
  nz /= nLen;

  const lx = -0.45;
  const ly = -0.6;
  const lz = 0.66;
  const lLen = Math.hypot(lx, ly, lz) || 1;
  const dot = clamp((nx * lx + ny * ly + nz * lz) / lLen, 0, 1);
  return clamp(0.9 + (dot - 0.55) * 0.2, 0.82, 1.04);
}

function heightColor(value, min, max, landMinRef, landMaxRef) {
  if (value < 0 && max > 0) {
    const t = Math.pow(clamp(-value / Math.max(1, -min), 0, 1), 0.68);
    return sampleStops(OCEAN_STOPS, t);
  }
  if (max <= 0) {
    return sampleStops(OCEAN_STOPS, 1);
  }
  const tLinear = clamp((value - landMinRef) / Math.max(1, landMaxRef - landMinRef), 0, 1);
  const t = Math.pow(tLinear, 0.72);
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
  const scope = process.env.SCOPE || 'planet';
  const islandType = process.env.ISLAND_TYPE || 'continental';
  const islandScaleKm = Number(process.env.ISLAND_SCALE_KM || 400);

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
    scope,
    ...(scope === 'island' && { islandType, islandScaleKm }),
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
  const expected = EXPECTED_SIZE[scope] || EXPECTED_SIZE.planet;
  if (width !== expected.width || height !== expected.height) {
    console.warn(`Unexpected map size for scope=${scope}: ${width}x${height}`);
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
    scope,
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
  const landToneRange = estimateLandToneRange(heightMap, maxH);
  const wrapXForHillshade = scope !== 'island';

  writeBmp(path.join(runDir, 'height_preview.bmp'), width, height, (i, x, y) => {
    const h = heightMap[i];
    const base = heightColor(h, minH, maxH, landToneRange.minRef, landToneRange.maxRef);
    if (h < 0) {
      return base;
    }
    const shade = landHillshade(heightMap, width, height, x, y, wrapXForHillshade);
    return [
      Math.round(clamp(base[0] * shade, 0, 255)),
      Math.round(clamp(base[1] * shade, 0, 255)),
      Math.round(clamp(base[2] * shade, 0, 255)),
    ];
  });
  writeJpg(path.join(runDir, 'height_preview.jpg'), width, height, (i, x, y) => {
    const h = heightMap[i];
    const base = heightColor(h, minH, maxH, landToneRange.minRef, landToneRange.maxRef);
    if (h < 0) {
      return base;
    }
    const shade = landHillshade(heightMap, width, height, x, y, wrapXForHillshade);
    return [
      Math.round(clamp(base[0] * shade, 0, 255)),
      Math.round(clamp(base[1] * shade, 0, 255)),
      Math.round(clamp(base[2] * shade, 0, 255)),
    ];
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
