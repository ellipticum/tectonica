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
  planet: { width: 4096, height: 2048 },
  island: { width: 1024, height: 512 },
  continent: { width: 7680, height: 4320 },
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
  // Spherical Y-wrapping at poles: reflect y and shift x by half-width
  // (same as equirectangular → sphere mapping used in the Rust engine).
  let yUp, yDown, xForUp, xForDown;
  if (y === 0) {
    yUp = 0;
    xForUp = wrapX ? (x + (width >> 1)) % width : x;
  } else {
    yUp = y - 1;
    xForUp = x;
  }
  if (y === height - 1) {
    yDown = height - 1;
    xForDown = wrapX ? (x + (width >> 1)) % width : x;
  } else {
    yDown = y + 1;
    xForDown = x;
  }
  const xLeft = wrapX ? (x - 1 + width) % width : Math.max(0, x - 1);
  const xRight = wrapX ? (x + 1) % width : Math.min(width - 1, x + 1);
  const left = heightMap[y * width + xLeft] || 0;
  const right = heightMap[y * width + xRight] || 0;
  const up = heightMap[yUp * width + xForUp] || 0;
  const down = heightMap[yDown * width + xForDown] || 0;

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
  // 12 Whittaker biome types — must match BIOME_COLORS in types.ts
  switch (id) {
    case 0:  return [17,  42,  82];   // Ocean
    case 1:  return [204, 213, 238];  // Tundra/Ice
    case 2:  return [63,  104, 61];   // Boreal/Taiga
    case 3:  return [21,  109, 61];   // Temperate Forest
    case 4:  return [196, 168, 84];   // Temperate Grassland
    case 5:  return [168, 142, 60];   // Mediterranean
    case 6:  return [1,   87,  50];   // Tropical Rainforest
    case 7:  return [132, 173, 93];   // Tropical Savanna
    case 8:  return [219, 179, 94];   // Desert
    case 9:  return [34,  139, 87];   // Subtropical Forest
    case 10: return [130, 106, 74];   // Alpine
    case 11: return [178, 162, 108];  // Steppe
    default: return [128, 128, 128];
  }
}

// --- Satellite / natural-color palette ---
// Landsat-like base colors per biome (R, G, B).
// These mimic what Landsat true-color composites show for each biome.
const SAT_BIOME = [
  [15,  45,  90],   // 0  Ocean (deep dark blue)
  [215, 220, 228],  // 1  Tundra/Ice (pale gray-white)
  [45,  72,  42],   // 2  Boreal/Taiga (dark olive)
  [32,  88,  38],   // 3  Temperate Forest (medium green)
  [145, 148, 72],   // 4  Temperate Grassland (olive yellow-green)
  [128, 118, 62],   // 5  Mediterranean (dry olive-tan)
  [12,  62,  28],   // 6  Tropical Rainforest (deep dark green)
  [95,  120, 55],   // 7  Tropical Savanna (olive green)
  [195, 170, 125],  // 8  Desert (sandy tan)
  [28,  95,  48],   // 9  Subtropical Forest (green)
  [115, 100, 82],   // 10 Alpine (gray-brown rock)
  [162, 155, 95],   // 11 Steppe (dry yellow)
];

// Ocean gradient for satellite view (shallow → deep)
const SAT_OCEAN_STOPS = [
  { t: 0,    color: [110, 160, 190] }, // very shallow (turquoise-ish)
  { t: 0.08, color: [55,  105, 155] }, // shelf
  { t: 0.25, color: [30,  72,  130] }, // mid-depth
  { t: 0.50, color: [18,  52,  108] }, // deep
  { t: 0.80, color: [10,  35,  82]  }, // abyss
  { t: 1.0,  color: [8,   25,  65]  }, // trench
];

function satelliteColor(i, x, y, heightMap, biomeMap, temperatureMap, precipitationMap,
                        riverMap, lakeMap, slopeMap, width, height, minH, maxH, wrapX) {
  const h = heightMap[i];

  // --- Ocean ---
  if (h < 0) {
    const depthT = Math.pow(clamp(-h / Math.max(1, -minH), 0, 1), 0.55);
    const base = sampleStops(SAT_OCEAN_STOPS, depthT);
    // Shallow water near coast: turquoise tint
    if (depthT < 0.08) {
      const shallowMix = 1.0 - depthT / 0.08;
      base[0] = Math.round(base[0] + (80 - base[0]) * shallowMix * 0.35);
      base[1] = Math.round(base[1] + (155 - base[1]) * shallowMix * 0.35);
      base[2] = Math.round(base[2] + (150 - base[2]) * shallowMix * 0.2);
    }
    // Ocean hillshade for underwater terrain
    const oShade = landHillshade(heightMap, width, height, x, y, wrapX);
    const oFactor = clamp(0.92 + (oShade - 0.82) * 0.6, 0.88, 1.06);
    return [
      Math.round(clamp(base[0] * oFactor, 0, 255)),
      Math.round(clamp(base[1] * oFactor, 0, 255)),
      Math.round(clamp(base[2] * oFactor, 0, 255)),
    ];
  }

  // --- Land: continuous color from climate + elevation ---
  const precip = precipitationMap[i];
  const temp = temperatureMap[i];
  const biome = biomeMap[i];

  // NDVI proxy: vegetation density from temperature and precipitation.
  // Plants need warmth + water.  Miami model NPP simplified.
  const vegT = clamp((temp - (-5)) / 30, 0, 1);     // 0 at -5°C, 1 at 25°C
  const vegP = clamp(precip / 900, 0, 1);             // 0 at 0mm, 1 at 900mm (temperate forests)
  const ndvi = Math.pow(vegT * vegP, 0.65);           // 0=barren, 1=lush; concave to boost mid-range

  // Base terrain palette: interpolate between dry ground and lush vegetation.
  // Dry ground (NDVI≈0): warm tan
  const dryR = 168, dryG = 152, dryB = 118;
  // Lush vegetation (NDVI≈1): rich green
  const lushR = 15, lushG = 85, lushB = 25;

  // Smooth NDVI curve (avoid sharp cutoffs)
  const v = Math.pow(ndvi, 0.7);
  let r = dryR + (lushR - dryR) * v;
  let g = dryG + (lushG - dryG) * v;
  let b = dryB + (lushB - dryB) * v;

  // Desert biome override: force sandy even if precip has some moisture
  if (biome === 8) {
    const desertMix = 0.6;
    r = r + (195 - r) * desertMix;
    g = g + (172 - g) * desertMix;
    b = b + (130 - b) * desertMix;
  }

  // Tropical rainforest: extra deep green
  if (biome === 6 && ndvi > 0.4) {
    const deepMix = 0.3;
    r = r + (8 - r) * deepMix;
    g = g + (55 - g) * deepMix;
    b = b + (22 - b) * deepMix;
  }

  // --- Elevation: higher → exposed rock ---
  // Gradual transition to rock color at high elevation.
  const rockStart = 1200;
  const rockFull = 4500;
  const rockT = clamp((h - rockStart) / (rockFull - rockStart), 0, 1);
  if (rockT > 0) {
    // Rock color varies with slope (steeper = darker)
    const slope = slopeMap[i];
    const steepness = clamp(slope / 0.8, 0, 1);
    const rockR = 140 - steepness * 35;
    const rockG = 125 - steepness * 30;
    const rockB = 105 - steepness * 25;
    // Mix: vegetation thins out before full rock
    const treeline = clamp(rockT * 1.5, 0, 1); // treeline fades early
    const vegRemaining = (1.0 - treeline) * ndvi;
    const rockMix = rockT * (1.0 - vegRemaining * 0.5);
    r = r + (rockR - r) * rockMix;
    g = g + (rockG - g) * rockMix;
    b = b + (rockB - b) * rockMix;
  }

  // --- Snow/ice ---
  const latDeg = 90 - (y + 0.5) * (180 / height);
  const absLat = Math.abs(latDeg);
  const snowline = 5200 - 62 * absLat;
  // Snow above ELA: gradual onset over 1200 m
  const snowElev = clamp((h - snowline) / 1200, 0, 1);
  // Polar/tundra snow at ground level
  const polarSnow = (biome === 1) ? clamp((absLat - 60) / 20, 0, 0.7) : 0;
  // Cold high peaks
  const coldPeak = clamp((-temp - 10) / 20, 0, 0.4) * clamp(h / 2000, 0, 1);
  const snowAmount = clamp(Math.max(snowElev, polarSnow, coldPeak), 0, 0.95);
  if (snowAmount > 0) {
    // Snow: slightly blue-white (varies with amount)
    const sr = 235 + snowAmount * 10;
    const sg = 238 + snowAmount * 8;
    const sb = 245 + snowAmount * 5;
    r = r + (sr - r) * snowAmount;
    g = g + (sg - g) * snowAmount;
    b = b + (sb - b) * snowAmount;
  }

  // --- Hillshade (dramatic for satellite) ---
  const shade = landHillshade(heightMap, width, height, x, y, wrapX);
  // Strong contrast: shadows dark, lit slopes bright
  const satShade = clamp(0.62 + (shade - 0.82) * 2.0, 0.55, 1.15);
  r = clamp(r * satShade, 0, 255);
  g = clamp(g * satShade, 0, 255);
  b = clamp(b * satShade, 0, 255);

  // --- Rivers & lakes (drawn AFTER hillshade for visibility) ---
  const river = riverMap[i];
  const lake = lakeMap[i];
  if (lake > 0) {
    // Lakes: solid water color, slightly darker for endorheic (lake==2)
    const lakeR = lake === 2 ? 50 : 35;
    const lakeG = lake === 2 ? 75 : 65;
    const lakeB = lake === 2 ? 95 : 120;
    r = r * 0.15 + lakeR * 0.85;
    g = g * 0.15 + lakeG * 0.85;
    b = b * 0.15 + lakeB * 0.85;
  } else if (river > 0.08) {
    // Rivers: bright blue, strength from log-discharge
    const riverStr = clamp((river - 0.08) / 0.45, 0, 0.95);
    // Brighter, more saturated blue than surrounding terrain
    r = r + (25 - r) * riverStr;
    g = g + (65 - g) * riverStr;
    b = b + (140 - b) * riverStr;
  }

  // --- Atmospheric scattering (subtle blue haze at distance / high latitude) ---
  const hazeFactor = clamp((absLat - 60) / 25, 0, 0.08);
  if (hazeFactor > 0) {
    r = r + (195 - r) * hazeFactor;
    g = g + (205 - g) * hazeFactor;
    b = b + (225 - b) * hazeFactor;
  }

  return [Math.round(r), Math.round(g), Math.round(b)];
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
  const continentScaleKm = Number(process.env.CONTINENT_SCALE_KM || 3000);

  // Planet parameters (all overridable via env)
  const radiusKm = Number(process.env.RADIUS_KM || 6371);
  const oceanPercent = Number(process.env.OCEAN_PERCENT || 67);
  const gravity = Number(process.env.GRAVITY || 9.81);
  const rotationHours = Number(process.env.ROTATION_HOURS || 24);
  const axialTiltDeg = Number(process.env.AXIAL_TILT || 23.5);
  const plateCount = Number(process.env.PLATE_COUNT || 11);
  const plateSpeed = Number(process.env.PLATE_SPEED || 5);
  const mantleHeat = Number(process.env.MANTLE_HEAT || 55);

  // Resolution: RESOLUTION=8192 → 8192x4096, RESOLUTION=2048 → 2048x1024
  const planetWidth = Number(process.env.RESOLUTION || 4096);
  const planetHeight = Math.round(planetWidth / 2);

  const config = {
    seed,
    planetWidth,
    planetHeight,
    planet: {
      radiusKm,
      gravity,
      density: 5510,
      rotationHours,
      axialTiltDeg,
      eccentricity: 0.016,
      atmosphereBar: 1,
      oceanPercent,
    },
    tectonics: {
      plateCount,
      plateSpeedCmPerYear: plateSpeed,
      mantleHeat,
    },
    events: [],
    generationPreset,
    scope,
    ...(scope === 'island' && { islandType, islandScaleKm }),
    ...(scope === 'continent' && { continentScaleKm }),
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

  // Satellite / natural-color view (Landsat-style)
  const satPixel = (i, x, y) => satelliteColor(
    i, x, y, heightMap, biomeMap, temperatureMap, precipitationMap,
    riverMap, lakeMap, slopeMap, width, height, minH, maxH, wrapXForHillshade
  );
  writeBmp(path.join(runDir, 'satellite_preview.bmp'), width, height, satPixel);
  writeJpg(path.join(runDir, 'satellite_preview.jpg'), width, height, satPixel);

  result.free();

  console.log(`Saved generation to: ${runDir}`);
}

run();
