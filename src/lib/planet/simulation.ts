export const WORLD_WIDTH = 360;
export const WORLD_HEIGHT = 180;
const WORLD_SIZE = WORLD_WIDTH * WORLD_HEIGHT;
const RADIANS = Math.PI / 180;
const KILOMETERS_PER_DEGREE = 111.319_490_793; // Equatorial distance around Earth-equivalent sphere.

export type LayerId =
  | "planet"
  | "plates"
  | "relief"
  | "hydrology"
  | "climate"
  | "biomes"
  | "settlement";

export type RecomputeTrigger = "global" | "tectonics" | "events";

export interface LayerSpec {
  id: LayerId;
  name: string;
  formula: string;
  inputs: string[];
  dependsOn: LayerId[];
}

export const LAYER_GRAPH: LayerSpec[] = [
  {
    id: "planet",
    name: "Планетарная геометрия",
    formula: "Сфера в км, глобальные константы (R, g, плотность, атмосфера, наклон, эксцентриситет).",
    inputs: [
      "Радиус",
      "g",
      "Плотность",
      "Период вращения",
      "Наклон оси",
      "Эксцентриситет",
      "Атм. давление",
      "Океан%",
    ],
    dependsOn: [],
  },
  {
    id: "plates",
    name: "Тектоника",
    formula: "Тип границы определён по векторной разнице скоростей плит. Горная высота H = k × v_convergence / g.",
    inputs: ["Кол-во плит", "Скорость плит", "Тепловой поток", "Глобальная геометрия"],
    dependsOn: ["planet"],
  },
  {
    id: "relief",
    name: "Рельеф",
    formula:
      "HeightMap = базовая тектоника + ограниченный стохастический шум + итеративная эрозия + события.",
    inputs: ["Тектоника", "Тепловой поток", "События", "Океанский уровень"],
    dependsOn: ["plates", "planet"],
  },
  {
    id: "hydrology",
    name: "Гидрология",
    formula:
      "Схема: уклон -> направление потока -> накопление воды -> рельефный разрез и озёра в локальных минимумах.",
    inputs: ["Рельеф", "События", "Океан%"],
    dependsOn: ["relief"],
  },
  {
    id: "climate",
    name: "Климат",
    formula:
      "T, P по широте (инсоляция), ветру и высоте; аэрозоли от метеоритов уменьшают температуру.",
    inputs: ["Планетарная геометрия", "Рельеф", "Гидрология", "События"],
    dependsOn: ["planet", "relief", "hydrology"],
  },
  {
    id: "biomes",
    name: "Биомы",
    formula: "Biome = f(Temperature, Precipitation, Elevation).",
    inputs: ["Температура", "Осадки", "Высота"],
    dependsOn: ["climate", "relief"],
  },
  {
    id: "settlement",
    name: "Расселение",
    formula:
      "Потенциал заселения = мягкая функция комфорта по T, P и уклону + штраф за высоту и подводные клетки.",
    inputs: ["Биомы", "Топография", "Климат"],
    dependsOn: ["biomes", "climate", "relief"],
  },
];

export const BIOME_NAMES = [
  "Океан",
  "Тундра",
  "Субтропический лес",
  "Лес",
  "Саванна",
  "Пустыня",
  "Степь",
  "Высокогорье",
  "Тайга",
];

export const BIOME_COLORS: Record<number, [number, number, number]> = {
  0: [17, 42, 82],
  1: [204, 213, 238],
  2: [34, 139, 87],
  3: [21, 109, 61],
  4: [196, 168, 84],
  5: [219, 179, 94],
  6: [132, 173, 93],
  7: [130, 106, 74],
  8: [63, 104, 61],
};

export type WorldDisplayLayer =
  | "plates"
  | "height"
  | "slope"
  | "rivers"
  | "precipitation"
  | "temperature"
  | "biomes"
  | "settlement"
  | "events";

export interface PlanetInputs {
  radiusKm: number;
  gravity: number;
  density: number;
  rotationHours: number;
  axialTiltDeg: number;
  eccentricity: number;
  atmosphereBar: number;
  oceanPercent: number;
}

export interface TectonicInputs {
  plateCount: number;
  plateSpeedCmPerYear: number;
  mantleHeat: number;
}

export type EventKind = "meteorite" | "rift" | "subduction" | "uplift" | "oceanShift";

export interface MeteoriteEvent {
  kind: "meteorite";
  latitude: number;
  longitude: number;
  diameterKm: number;
  speedKms: number;
  angleDeg: number;
  densityKgM3: number;
}

export interface RegionEvent {
  kind: "rift" | "subduction" | "uplift" | "oceanShift";
  latitude: number;
  longitude: number;
  radiusKm: number;
  magnitude: number;
}

export type WorldEvent = MeteoriteEvent | RegionEvent;

export type WorldEventRecord = WorldEvent & {
  id: string;
  createdAt: string;
  summary: string;
  energyJoule?: number;
};

export interface SimulationConfig {
  seed: number;
  planet: PlanetInputs;
  tectonics: TectonicInputs;
  events: WorldEventRecord[];
  generationPreset?: "fast" | "balanced" | "detailed";
}

export interface SimulationStats {
  minHeight: number;
  maxHeight: number;
  minTemperature: number;
  maxTemperature: number;
  minPrecipitation: number;
  maxPrecipitation: number;
  minSlope: number;
  maxSlope: number;
}

export interface SimulationResult {
  width: number;
  height: number;
  seed: number;
  specs: LayerSpec[];
  recomputedLayers: LayerId[];
  planet: {
    seaLevel: number;
    radiusKm: number;
    oceanPercent: number;
  };
  plates: Int16Array;
  boundaryTypes: Int8Array;
  heightMap: Float32Array;
  slopeMap: Float32Array;
  riverMap: Float32Array;
  lakeMap: Uint8Array;
  flowDirection: Int32Array;
  flowAccumulation: Float32Array;
  temperatureMap: Float32Array;
  precipitationMap: Float32Array;
  biomeMap: Uint8Array;
  settlementMap: Float32Array;
  eventHistory: WorldEventRecord[];
  stats: SimulationStats;
}

export interface HeightExportPayload {
  format: "float32" | "int16";
  width: number;
  height: number;
  radiusKm: number;
  scaleFactor?: number;
  values: number[];
}

interface PlateSpec {
  lat: number;
  lon: number;
  speed: number;
  dirX: number;
  dirY: number;
  heat: number;
  buoyancy: number;
}

interface RNG {
  next: () => number;
}

function makeRng(seed: number): RNG {
  let state = seed >>> 0;
  return {
    next() {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    },
  };
}

function clamp(value: number, min: number, max: number) {
  return value < min ? min : value > max ? max : value;
}

const INDEX = (x: number, y: number) => y * WORLD_WIDTH + x;

function sphericalWrap(x: number, y: number) {
  let wrappedX = x;
  let wrappedY = y;

  while (wrappedY < 0 || wrappedY >= WORLD_HEIGHT) {
    if (wrappedY < 0) {
      wrappedY = -wrappedY - 1;
      wrappedX += WORLD_WIDTH / 2;
    } else {
      wrappedY = 2 * WORLD_HEIGHT - wrappedY - 1;
      wrappedX += WORLD_WIDTH / 2;
    }
  }

  const normalizedX = ((Math.round(wrappedX) % WORLD_WIDTH) + WORLD_WIDTH) % WORLD_WIDTH;
  const normalizedY = clamp(Math.round(wrappedY), 0, WORLD_HEIGHT - 1);
  return { x: normalizedX, y: normalizedY };
}

function INDEX_SPHERICAL(x: number, y: number) {
  const p = sphericalWrap(x, y);
  return INDEX(p.x, p.y);
}

const latByY = new Float32Array(WORLD_SIZE);
const lonByX = new Float32Array(WORLD_SIZE);
const xByCell = new Float32Array(WORLD_SIZE);
const yByCell = new Float32Array(WORLD_SIZE);
const zByCell = new Float32Array(WORLD_SIZE);

for (let y = 0; y < WORLD_HEIGHT; y++) {
  const latDeg = 90 - (y + 0.5) * (180 / WORLD_HEIGHT);
  const latRad = latDeg * RADIANS;
  for (let x = 0; x < WORLD_WIDTH; x++) {
    const lonDeg = (x + 0.5) * (360 / WORLD_WIDTH) - 180;
    const lonRad = lonDeg * RADIANS;
    const i = INDEX(x, y);
    latByY[i] = latDeg;
    lonByX[i] = lonDeg;
    const cosLat = Math.cos(latRad);
    xByCell[i] = cosLat * Math.cos(lonRad);
    yByCell[i] = cosLat * Math.sin(lonRad);
    zByCell[i] = Math.sin(latRad);
  }
}

export const DEFAULT_PLANET: PlanetInputs = {
  radiusKm: 6371,
  gravity: 9.81,
  density: 5510,
  rotationHours: 24,
  axialTiltDeg: 23.5,
  eccentricity: 0.016,
  atmosphereBar: 1,
  oceanPercent: 67,
};

export const DEFAULT_TECTONICS: TectonicInputs = {
  plateCount: 11,
  plateSpeedCmPerYear: 5,
  mantleHeat: 55,
};

export const DEFAULT_SIMULATION: SimulationConfig = {
  seed: 2026,
  planet: DEFAULT_PLANET,
  tectonics: DEFAULT_TECTONICS,
  events: [],
  generationPreset: "balanced",
};

function randomRange(rng: RNG, min: number, max: number) {
  return min + rng.next() * (max - min);
}

function quantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  const t = clamp(q, 0, 1) * (sortedValues.length - 1);
  const lo = Math.floor(t);
  const hi = Math.min(sortedValues.length - 1, lo + 1);
  const k = t - lo;
  return sortedValues[lo] * (1 - k) + sortedValues[hi] * k;
}

function normalizeHeightRange(
  relief: Float32Array,
  planet: PlanetInputs,
  tectonics: TectonicInputs,
) {
  const landHeights: number[] = [];
  const oceanDepths: number[] = [];

  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    if (h > 0) {
      landHeights.push(h);
    } else if (h < 0) {
      oceanDepths.push(-h);
    }
  }

  if (landHeights.length < 8 || oceanDepths.length < 8) {
    return;
  }

  landHeights.sort((a, b) => a - b);
  oceanDepths.sort((a, b) => a - b);

  const landRef = Math.max(1, quantile(landHeights, 0.995));
  const oceanRef = Math.max(1, quantile(oceanDepths, 0.995));

  const speedFactor = clamp(tectonics.plateSpeedCmPerYear / 5, 0.35, 2.8);
  const heatFactor = clamp(tectonics.mantleHeat / 55, 0.45, 2.2);
  const gravityFactor = clamp(9.81 / Math.max(1, planet.gravity), 0.45, 2.4);
  const oceanFactor = clamp(planet.oceanPercent / 67, 0.45, 1.9);

  const targetLandMax = clamp(
    6000 * Math.pow(speedFactor, 0.55) * Math.pow(heatFactor, 0.45) * Math.pow(gravityFactor, 0.7),
    1800,
    9000,
  );
  const targetOceanDepth = clamp(
    7500 * Math.pow(speedFactor, 0.5) * Math.pow(heatFactor, 0.35) * Math.pow(oceanFactor, 0.55),
    1400,
    12000,
  );

  const landScale = targetLandMax / landRef;
  const oceanScale = targetOceanDepth / oceanRef;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    if (h > 0) {
      relief[i] = clamp(h * landScale, -12000, 9000);
    } else if (h < 0) {
      relief[i] = clamp(-(-h * oceanScale), -12000, 9000);
    }
  }
}

function reshapeOceanBoundaries(
  relief: Float32Array,
  boundaryTypes: Int8Array,
  boundaryStrength: Float32Array,
) {
  let deepest = 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    if (relief[i] < 0) {
      deepest = Math.max(deepest, -relief[i]);
    }
  }
  const depthScale = Math.max(1, deepest);

  for (let i = 0; i < WORLD_SIZE; i++) {
    if (relief[i] >= 0) continue;
    const type = boundaryTypes[i];
    if (type === 0) continue;

    const depthT = clamp(-relief[i] / depthScale, 0, 1);
    const strength = clamp(boundaryStrength[i], 0, 1);

    if (type === 2) {
      // Divergence below sea level should bias toward mid-ocean ridges instead of deep trenches.
      const uplift = (520 + 2200 * strength) * (0.7 + 0.3 * (1 - depthT));
      relief[i] = Math.min(-15, relief[i] + uplift);
    } else if (type === 3) {
      const uplift = (120 + 480 * strength) * (0.65 + 0.35 * (1 - depthT));
      relief[i] += uplift;
    }
  }

  const oceanSmoothed = new Float32Array(relief);
  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      if (relief[i] >= 0) continue;
      let sum = relief[i] * 3.4;
      let wsum = 3.4;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          if (relief[j] >= 0) continue;
          const w = ox === 0 || oy === 0 ? 0.6 : 0.45;
          sum += relief[j] * w;
          wsum += w;
        }
      }
      oceanSmoothed[i] = sum / Math.max(1e-6, wsum);
    }
  }
  relief.set(oceanSmoothed);
}

function applyCoastalDetail(relief: Float32Array, seed: number) {
  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    const nearSea = Math.exp(-Math.abs(h) / 900);
    if (nearSea < 0.035) continue;

    const lat = latByY[i];
    const lon = lonByX[i];
    const warpLat = lat + 2.4 * Math.sin(lon * 0.065 + seed * 0.0013);
    const warpLon = lon + 3.6 * Math.cos(lat * 0.058 - seed * 0.0011);
    const n =
      Math.sin(warpLat * 0.19 + warpLon * 0.23 + seed * 0.0022) * 0.62 +
      Math.cos(warpLat * 0.31 - warpLon * 0.17 - seed * 0.0016) * 0.38;
    const coastWeight = Math.pow(nearSea, 1.18);
    relief[i] += coastWeight * n * 92;
  }
}

function cleanupCoastalSpeckles(relief: Float32Array) {
  const next = new Float32Array(relief);

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = relief[i];
      if (Math.abs(h) > 260) {
        continue;
      }

      let land = 0;
      let water = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          if (relief[j] >= 0) {
            land++;
          } else {
            water++;
          }
        }
      }

      if (h >= 0 && water >= 7) {
        next[i] = Math.min(next[i], -18 - Math.abs(h) * 0.15);
      } else if (h < 0 && land >= 7) {
        next[i] = Math.max(next[i], 18 + Math.abs(h) * 0.15);
      }
    }
  }

  relief.set(next);
}

interface FrontierNode {
  cost: number;
  index: number;
  plate: number;
}

class MinCostQueue {
  private readonly data: FrontierNode[] = [];

  get size() {
    return this.data.length;
  }

  push(item: FrontierNode) {
    this.data.push(item);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p].cost <= this.data[i].cost) break;
      const tmp = this.data[p];
      this.data[p] = this.data[i];
      this.data[i] = tmp;
      i = p;
    }
  }

  pop(): FrontierNode | undefined {
    if (this.data.length === 0) return undefined;
    const root = this.data[0];
    const tail = this.data.pop();
    if (tail && this.data.length > 0) {
      this.data[0] = tail;
      let i = 0;
      while (true) {
        const left = i * 2 + 1;
        const right = left + 1;
        if (left >= this.data.length) break;
        let minChild = left;
        if (right < this.data.length && this.data[right].cost < this.data[left].cost) {
          minChild = right;
        }
        if (this.data[i].cost <= this.data[minChild].cost) break;
        const tmp = this.data[i];
        this.data[i] = this.data[minChild];
        this.data[minChild] = tmp;
        i = minChild;
      }
    }
    return root;
  }
}

function latLonToIndex(lat: number, lon: number) {
  const y = clamp(Math.round((90 - lat) / (180 / WORLD_HEIGHT)), 0, WORLD_HEIGHT - 1);
  const x = ((Math.round((lon + 180) / (360 / WORLD_WIDTH)) % WORLD_WIDTH) + WORLD_WIDTH) % WORLD_WIDTH;
  return INDEX(x, y);
}

function nearestFreeIndex(start: number, occupied: Uint8Array) {
  if (!occupied[start]) return start;

  const sx = start % WORLD_WIDTH;
  const sy = (start - sx) / WORLD_WIDTH;
  const maxRadius = Math.max(WORLD_WIDTH, WORLD_HEIGHT);

  for (let radius = 1; radius < maxRadius; radius++) {
    for (let dy = -radius; dy <= radius; dy++) {
      const y = sy + dy;
      if (y < 0 || y >= WORLD_HEIGHT) continue;
      const span = radius - Math.abs(dy);
      const candidates = [sx - span, sx + span];
      for (const cx of candidates) {
        const x = (cx % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH;
        const idx = INDEX(x, y);
        if (!occupied[idx]) return idx;
      }
    }
  }

  return start;
}

function buildIrregularPlateField(plates: PlateSpec[], seed: number) {
  const plateField = new Int16Array(WORLD_SIZE);
  plateField.fill(-1);

  const openCost = new Float32Array(WORLD_SIZE);
  openCost.fill(Number.POSITIVE_INFINITY);

  const occupiedSeeds = new Uint8Array(WORLD_SIZE);
  const growthRng = makeRng((seed ^ 0x9e3779b9) >>> 0);
  const queue = new MinCostQueue();

  const growthParams = plates.map((plate) => {
    const len = Math.hypot(plate.dirX, plate.dirY) || 1;
    return {
      driftX: plate.dirX / len,
      driftY: plate.dirY / len,
      spread: randomRange(growthRng, 0.85, 1.25),
      roughness: randomRange(growthRng, 0.28, 1.05),
      freqA: randomRange(growthRng, 0.045, 0.145),
      freqB: randomRange(growthRng, 0.055, 0.16),
      freqC: randomRange(growthRng, 0.08, 0.22),
      freqD: randomRange(growthRng, 0.07, 0.2),
      phaseA: randomRange(growthRng, -Math.PI, Math.PI),
      phaseB: randomRange(growthRng, -Math.PI, Math.PI),
    };
  });

  for (let plate = 0; plate < plates.length; plate++) {
    const p = plates[plate];
    const seedIndex = nearestFreeIndex(latLonToIndex(p.lat, p.lon), occupiedSeeds);
    occupiedSeeds[seedIndex] = 1;
    openCost[seedIndex] = 0;
    queue.push({ cost: 0, index: seedIndex, plate });
  }

  const steps = [
    { dx: 1, dy: 0, w: 1 },
    { dx: -1, dy: 0, w: 1 },
    { dx: 0, dy: 1, w: 1 },
    { dx: 0, dy: -1, w: 1 },
    { dx: 1, dy: 1, w: Math.SQRT2 },
    { dx: -1, dy: 1, w: Math.SQRT2 },
    { dx: 1, dy: -1, w: Math.SQRT2 },
    { dx: -1, dy: -1, w: Math.SQRT2 },
  ];

  let assigned = 0;
  while (queue.size > 0 && assigned < WORLD_SIZE) {
    const node = queue.pop();
    if (!node) break;
    const { cost, index, plate } = node;
    if (cost > openCost[index] + 1e-6) continue;
    if (plateField[index] !== -1) continue;

    plateField[index] = plate;
    assigned++;

    const x = index % WORLD_WIDTH;
    const y = (index - x) / WORLD_WIDTH;
    const gp = growthParams[plate];

    for (const step of steps) {
      const j = INDEX_SPHERICAL(x + step.dx, y + step.dy);
      if (plateField[j] !== -1) continue;

      const lat = latByY[j];
      const lon = lonByX[j];
      const waveA = Math.sin(lat * gp.freqA + lon * gp.freqB + gp.phaseA);
      const waveB = Math.cos(lat * gp.freqC - lon * gp.freqD + gp.phaseB);
      const roughFactor = 1 + gp.roughness * (0.22 * waveA + 0.18 * waveB);
      const driftAlign = step.dx * gp.driftX + step.dy * gp.driftY;
      const driftFactor = 1.03 - 0.12 * driftAlign;
      const polarFactor = 1 + (Math.abs(lat) / 90) * 0.1;
      const stepCost = Math.max(0.08, step.w * gp.spread * roughFactor * driftFactor * polarFactor);
      const nextCost = cost + stepCost;

      if (nextCost + 1e-6 < openCost[j]) {
        openCost[j] = nextCost;
        queue.push({ cost: nextCost, index: j, plate });
      }
    }
  }

  if (assigned < WORLD_SIZE) {
    for (let i = 0; i < WORLD_SIZE; i++) {
      if (plateField[i] !== -1) continue;
      let bestPlate = 0;
      let bestCost = Number.POSITIVE_INFINITY;
      const cellX = xByCell[i];
      const cellY = yByCell[i];
      const cellZ = zByCell[i];
      for (let p = 0; p < plates.length; p++) {
        const latR = plates[p].lat * RADIANS;
        const lonR = plates[p].lon * RADIANS;
        const c = Math.cos(latR);
        const px = c * Math.cos(lonR);
        const py = c * Math.sin(lonR);
        const pz = Math.sin(latR);
        const d = 1 - (cellX * px + cellY * py + cellZ * pz);
        if (d < bestCost) {
          bestCost = d;
          bestPlate = p;
        }
      }
      plateField[i] = bestPlate;
    }
  }

  return plateField;
}

function computePlates(planet: PlanetInputs, tectonics: TectonicInputs, seed: number) {
  const plateCount = clamp(Math.round(tectonics.plateCount), 2, 20);
  const rng = makeRng(seed + plateCount * 7919);
  const plates: PlateSpec[] = [];
  for (let i = 0; i < plateCount; i++) {
    const lat = randomRange(rng, -90, 90);
    const lon = randomRange(rng, -180, 180);
    const speed = Math.max(0.001, randomRange(rng, 0.5, 1.5) * tectonics.plateSpeedCmPerYear);
    const dir = randomRange(rng, 0, Math.PI * 2);
    plates.push({
      lat,
      lon,
      speed,
      dirX: Math.cos(dir) * speed,
      dirY: Math.sin(dir) * speed,
      heat: randomRange(rng, Math.max(1, tectonics.mantleHeat * 0.5), tectonics.mantleHeat * 1.5),
      buoyancy: randomRange(rng, -1, 1),
    });
  }

  const plateField = buildIrregularPlateField(plates, seed);
  const plateVectors = plates.map((plate) => {
    const latR = plate.lat * RADIANS;
    const lonR = plate.lon * RADIANS;
    const c = Math.cos(latR);
    return {
      x: c * Math.cos(lonR),
      y: c * Math.sin(lonR),
      z: Math.sin(latR),
      speed: plate.speed,
      dirX: plate.dirX,
      dirY: plate.dirY,
      heat: plate.heat,
      buoyancy: plate.buoyancy,
    };
  });

  const boundaryTypes = new Int8Array(WORLD_SIZE);
  const boundaryScores = new Float32Array(WORLD_SIZE);
  const boundaryNormalX = new Float32Array(WORLD_SIZE);
  const boundaryNormalY = new Float32Array(WORLD_SIZE);
  const boundaryStrength = new Float32Array(WORLD_SIZE);
  const boundaryScale = Math.max(1.2, tectonics.plateSpeedCmPerYear * 1.25);

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const plateA = plateField[i];
      const a = plateVectors[plateA];
      let score = 0;
      let normalX = 0;
      let normalY = 0;
      let hasDifferentNeighbor = false;
      const neighbors = [
        { nx: (x + 1) % WORLD_WIDTH, ny: y, dx: 1, dy: 0, w: 1 },
        { nx: x === 0 ? WORLD_WIDTH - 1 : x - 1, ny: y, dx: -1, dy: 0, w: 1 },
        { nx: x, ny: y - 1, dx: 0, dy: -1, w: 1 },
        { nx: x, ny: y + 1, dx: 0, dy: 1, w: 1 },
        { nx: (x + 1) % WORLD_WIDTH, ny: y - 1, dx: 1, dy: -1, w: Math.SQRT1_2 },
        { nx: x === 0 ? WORLD_WIDTH - 1 : x - 1, ny: y - 1, dx: -1, dy: -1, w: Math.SQRT1_2 },
        { nx: (x + 1) % WORLD_WIDTH, ny: y + 1, dx: 1, dy: 1, w: Math.SQRT1_2 },
        { nx: x === 0 ? WORLD_WIDTH - 1 : x - 1, ny: y + 1, dx: -1, dy: 1, w: Math.SQRT1_2 },
      ];

      for (const n of neighbors) {
        const j = INDEX_SPHERICAL(n.nx, n.ny);
        const plateB = plateField[j];
        if (plateB === plateA) {
          continue;
        }
        hasDifferentNeighbor = true;
        const b = plateVectors[plateB];
        const relX = b.dirX - a.dirX;
        const relY = b.dirY - a.dirY;
        const edgeScore = (relX * n.dx + relY * n.dy) * n.w;
        if (Math.abs(edgeScore) > Math.abs(score)) {
          score = edgeScore;
          normalX = n.dx;
          normalY = n.dy;
        }
      }

      boundaryScores[i] = score;
      boundaryNormalX[i] = normalX;
      boundaryNormalY[i] = normalY;
      boundaryStrength[i] = clamp(Math.abs(score) / boundaryScale, 0, 1);
      if (score > 0.2 * boundaryScale) {
        boundaryTypes[i] = 1; // convergence/orogeny
      } else if (score < -0.2 * boundaryScale) {
        boundaryTypes[i] = 2; // divergence/rift
      } else if (hasDifferentNeighbor) {
        boundaryTypes[i] = 3; // transform
      } else {
        boundaryTypes[i] = 0;
      }
    }
  }

  return {
    plateField,
    boundaryTypes,
    boundaryScores,
    boundaryNormalX,
    boundaryNormalY,
    boundaryStrength,
    plateVectors,
  };
}

function computeRelief(
  planet: PlanetInputs,
  tectonics: TectonicInputs,
  plates: ReturnType<typeof computePlates>,
  seed: number,
) {
  const {
    plateField,
    boundaryTypes,
    boundaryScores,
    boundaryNormalX,
    boundaryNormalY,
    boundaryStrength,
    plateVectors,
  } = plates;
  const relief = new Float32Array(WORLD_SIZE);
  const randomSeed = makeRng(
    (((planet.radiusKm * 1000) | 0) ^
      ((tectonics.plateSpeedCmPerYear * 1000) | 0) ^
      ((tectonics.mantleHeat * 1000) | 0) ^
      ((seed * 2654435761) | 0)) >>> 0,
  );
  const macroA = randomRange(randomSeed, 0.012, 0.03);
  const macroB = randomRange(randomSeed, 0.008, 0.02);
  const macroC = randomRange(randomSeed, 0.006, 0.016);
  const phaseA = randomRange(randomSeed, -Math.PI, Math.PI);
  const phaseB = randomRange(randomSeed, -Math.PI, Math.PI);
  const phaseC = randomRange(randomSeed, -Math.PI, Math.PI);
  const kRelief = 560;
  const maxBoundaryScore = Math.max(1, tectonics.plateSpeedCmPerYear * 2.5);
  const baseKernelRadius = 3;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const plateId = plateField[i];
    const plateSpeed = plateVectors[plateId].speed;
    const heat = plateVectors[plateId].heat;
    const plateBuoyancy = plateVectors[plateId].buoyancy;
    const x = i % WORLD_WIDTH;
    const y = Math.floor(i / WORLD_WIDTH);

    let convInfluence = 0;
    let divInfluence = 0;
    let transformInfluence = 0;

    const localKernel = Math.min(
      6,
      baseKernelRadius + Math.floor(plateSpeed / 1.8) + Math.floor((heat / 30) * 2),
    );

    for (let oy = -localKernel; oy <= localKernel; oy++) {
      for (let ox = -localKernel; ox <= localKernel; ox++) {
        const j = INDEX_SPHERICAL(x + ox, y + oy);
        const t = boundaryTypes[j];
        if (t === 0) continue;

        const sourcePlate = plateField[j];
        const sourceHeat = plateVectors[sourcePlate].heat;
        const ridgeNoise = 0.65 + 0.35 * Math.sin((latByY[j] * 0.11 + lonByX[j] * 0.17 + seed * 0.0027));
        const sourceHeatNorm = sourceHeat / Math.max(1, tectonics.mantleHeat);
        const heatWidth = 0.75 + sourceHeatNorm * 1.6 + (sourceHeatNorm - 0.55) * 0.9;
        const localWidth = clamp(0.65 + heatWidth * ridgeNoise, 0.5, 4.8);
        const s = clamp(Math.abs(boundaryScores[j]) / maxBoundaryScore, 0, 1);
        const nNormalX = boundaryNormalX[j] || 0;
        const nNormalY = boundaryNormalY[j] || 0;
        const normalLen = Math.hypot(nNormalX, nNormalY) || 1;
        const normalX = nNormalX / normalLen;
        const normalY = nNormalY / normalLen;
        const tangentX = -normalY;
        const tangentY = normalX;
        const across = ox * normalX + oy * normalY;
        const along = ox * tangentX + oy * tangentY;
        const boundaryBias = clamp(boundaryStrength[j] * 1.25, 0, 1.4);
        const sigmaAcross = Math.max(0.55, localWidth * (0.85 + 1.1 * s) + 0.05 * Math.sin(ox * 3.1 + oy * 2.3));
        const sigmaAlong = Math.max(
          1.6,
          localWidth * (3.0 + 3.2 * boundaryBias + 1.6 * s) + 1.0 * Math.cos(oy * 2.1 - ox * 1.9),
        );
        const anisotropy =
          (across * across) / (sigmaAcross * sigmaAcross) + (along * along) / (sigmaAlong * sigmaAlong);
        const w = Math.exp(-anisotropy);
        const widthTuning = 0.32 + 0.95 * s + 0.28 * boundaryBias;
        const chainSegment =
          0.35 + 0.65 * (0.5 + 0.5 * Math.sin(latByY[j] * 0.27 + lonByX[j] * 0.34 + phaseA));
        const chainCluster =
          0.45 + 0.55 * (0.5 + 0.5 * Math.cos(latByY[j] * 0.22 - lonByX[j] * 0.19 + phaseB));
        let intensity = w * widthTuning;

        if (t === 1) {
          intensity *= chainSegment * chainCluster;
          convInfluence += intensity;
        } else if (t === 2) {
          intensity *= 0.55 + 0.45 * chainSegment;
          divInfluence += intensity;
        } else {
          transformInfluence += intensity;
        }
      }
    }

    const lat = latByY[i];
    const lon = lonByX[i];
    const warpLat = lat + 11 * Math.sin(lon * 0.045 + phaseA) + 6 * Math.cos(lat * 0.062 + phaseB);
    const warpLon = lon + 14 * Math.cos(lat * 0.037 + phaseC) - 7 * Math.sin(lon * 0.053 + phaseA);
    const continentalSignal =
      Math.sin(warpLat * 0.018 + phaseA) * 0.95 +
      Math.cos(warpLon * 0.013 + phaseB) * 0.8 +
      Math.sin((warpLat + warpLon) * 0.0105 + phaseC) * 0.55 +
      Math.cos(warpLat * 0.023 - warpLon * 0.019 + phaseA * 0.65) * 0.4;
    const regionalSignal = Math.sin((lat + lon) * 0.072 + phaseB) * Math.cos((lat - lon) * 0.059 + phaseC);
    const crustMaskRaw = continentalSignal * 0.95 + regionalSignal * 0.45 + plateBuoyancy * 1.2;
    const crustMask = 1 / (1 + Math.exp(-crustMaskRaw));

    let base = 0;
    if (convInfluence > 0.03) {
      const convShape = Math.pow(convInfluence, 0.78);
      base += ((kRelief * plateSpeed) / Math.max(1, planet.gravity)) * (0.04 + 0.12 * Math.atan(convShape));
      base += 105 * Math.log1p(convShape);
    }
    if (divInfluence > 0.03) {
      const divShape = Math.pow(divInfluence, 0.84);
      base -= (95 + 6 * plateSpeed) * (0.06 + 0.18 * Math.atan(divShape));
    }
    if (transformInfluence > 0.03) {
      base += 18 * (plateSpeed - 1) * Math.atan(transformInfluence);
    }
    const tectonicWeight = 0.16 + 0.84 * crustMask;
    base *= tectonicWeight;
    const intraplateSignal =
      Math.sin((lat * 0.083 + lon * 0.064) + plateId * 1.77 + phaseA * 0.25) *
      Math.cos((lat * 0.059 - lon * 0.051) + plateId * 0.91 + phaseB * 0.3);
    const macroNoise =
      Math.sin(lat * macroA + lon * macroB + phaseA) * 140 +
      Math.sin(lat * (macroA * 0.55) - lon * macroC + phaseB) * 90 +
      Math.cos(lon * (macroB * 1.6) + phaseC) * 70;
    const continentalBase = (crustMask - 0.5) * 3600;
    const macroBase = continentalBase + intraplateSignal * 260 + plateBuoyancy * 420 + regionalSignal * 110;
    const noise = (randomSeed.next() - 0.5) * 78 + macroNoise * 0.45;
    const heatTerm = heat * 1.8;
    relief[i] = base + macroBase + noise + heatTerm;
  }

  const macroBlend = new Float32Array(WORLD_SIZE);
  const macroRadius = 1;
  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      let sum = 0;
      let wsum = 0;
      for (let oy = -macroRadius; oy <= macroRadius; oy++) {
        for (let ox = -macroRadius; ox <= macroRadius; ox++) {
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const dist2 = ox * ox + oy * oy;
          const w = Math.exp(-dist2 / 1.9);
          sum += relief[j] * w;
          wsum += w;
        }
      }
      macroBlend[i] = sum / Math.max(1e-6, wsum);
    }
  }
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] = relief[i] * 0.82 + macroBlend[i] * 0.18;
  }

  const erosionRounds = 2;
  const smoothed = new Float32Array(relief);
  const scratch = new Float32Array(WORLD_SIZE);

  for (let round = 0; round < erosionRounds; round++) {
    for (let y = 0; y < WORLD_HEIGHT; y++) {
      for (let x = 0; x < WORLD_WIDTH; x++) {
        const i = INDEX(x, y);
        let sum = 0;
        let count = 0;
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const j = INDEX_SPHERICAL(x + ox, y + oy);
            sum += smoothed[j];
            count++;
          }
        }
        const avg = sum / count;
        scratch[i] = smoothed[i] * 0.92 + avg * 0.08;
      }
    }

    for (let i = 0; i < WORLD_SIZE; i++) {
      const drop = Math.max(0, smoothed[i] - scratch[i]);
      const heightLoss = Math.min(drop * 0.18, 22);
      smoothed[i] = scratch[i] - heightLoss;
    }
  }

  relief.set(smoothed);

  // Ocean percent normalization.
  const sorted = Array.from(relief).sort((a, b) => a - b);
  const oceanCut = Math.max(
    0,
    Math.min(WORLD_SIZE - 1, Math.floor((planet.oceanPercent / 100) * WORLD_SIZE))
  );
  const seaLevel = sorted[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= seaLevel;
  }

  applyCoastalDetail(relief, seed);
  cleanupCoastalSpeckles(relief);
  const sortedAfterCoast = Array.from(relief).sort((a, b) => a - b);
  const coastRecenter = sortedAfterCoast[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= coastRecenter;
  }

  normalizeHeightRange(relief, planet, tectonics);
  reshapeOceanBoundaries(relief, boundaryTypes, boundaryStrength);
  const sortedFinal = Array.from(relief).sort((a, b) => a - b);
  const finalRecenter = sortedFinal[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= finalRecenter;
  }

  return { relief, seaLevel };
}

function applyEvents(
  planet: PlanetInputs,
  relief: Float32Array,
  events: WorldEventRecord[],
): { relief: Float32Array; aerosol: number } {
  const updated = new Float32Array(relief);
  let aerosolIndex = 0;

  const nearestCell = (lat: number, lon: number) => {
    const y = clamp(Math.round((90 - lat) / (180 / WORLD_HEIGHT)), 0, WORLD_HEIGHT - 1);
    const x =
      ((Math.round((lon + 180) / (360 / WORLD_WIDTH)) % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH);
    return INDEX(x, y);
  };

  const kmPerCellLat = KILOMETERS_PER_DEGREE * planet.radiusKm / 6371;
  const kmPerCellLon = (lat: number, lon: number) =>
    Math.max(1, KILOMETERS_PER_DEGREE * Math.cos(lat * RADIANS) * Math.cos(lon * RADIANS) * planet.radiusKm / 6371);

  for (const event of events) {
    if (event.kind === "meteorite") {
      const radiusM = ((event.diameterKm * 1000) / 2) || 1;
      const mass = (4 / 3) * Math.PI * Math.pow(radiusM, 3) * event.densityKgM3;
      const energy = 0.5 * mass * Math.pow(event.speedKms * 1000, 2);
      const craterRadiusKm = Math.max(8, Math.pow(energy, 1 / 5) / 2500);
      const craterDepth = Math.min(9000, 800 + (Math.log10(energy + 1) - 10) * 250);
      const centerIndex = nearestCell(event.latitude, event.longitude);
      const cx = centerIndex % WORLD_WIDTH;
      const cy = Math.floor(centerIndex / WORLD_WIDTH);
      const latSpan = craterRadiusKm / kmPerCellLat;
      const lonSpan = craterRadiusKm / kmPerCellLon(event.latitude, event.longitude);

      const minX = Math.floor(cx - lonSpan - 1);
      const maxX = Math.ceil(cx + lonSpan + 1);
      const minY = clamp(Math.floor(cy - latSpan - 1), 0, WORLD_HEIGHT - 1);
      const maxY = clamp(Math.ceil(cy + latSpan + 1), 0, WORLD_HEIGHT - 1);

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const wrappedX = (x % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH;
          const target = INDEX(wrappedX, y);
          const dLat = latByY[target] - event.latitude;
          const dLon =
            (((((lonByX[target] - event.longitude + 180) % 360) + 360) % 360) - 180);
          const dKm = Math.sqrt(
            Math.pow(dLat * kmPerCellLat, 2) +
              Math.pow(dLon * KILOMETERS_PER_DEGREE * Math.cos(latByY[target] * RADIANS) * planet.radiusKm / 6371, 2),
          );
          if (dKm <= craterRadiusKm) {
            const falloff = 1 - Math.pow(dKm / craterRadiusKm, 2);
            updated[target] -= craterDepth * Math.max(0, falloff) * 0.5;
            updated[target] = Math.max(-planet.radiusKm * 10, updated[target]);
          }
        }
      }
      aerosolIndex += Math.min(0.45, Math.log10(energy + 1) / 18);
    } else if (event.kind === "oceanShift") {
      const idx = nearestCell(event.latitude, event.longitude);
      updated[idx] += event.magnitude * 0.5;
      for (let i = 0; i < WORLD_SIZE; i++) {
        updated[i] += event.magnitude * 0.15;
      }
    } else {
      const centerIndex = nearestCell(event.latitude, event.longitude);
      const cx = centerIndex % WORLD_WIDTH;
      const cy = Math.floor(centerIndex / WORLD_WIDTH);
      const sign = event.kind === "rift" ? -1 : 1;
      const magnitude = event.magnitude * 40 * sign;
      const radiusCells = Math.max(
        1,
        Math.round(event.radiusKm / (KILOMETERS_PER_DEGREE * planet.radiusKm / 6371))
      );
      for (let y = cy - radiusCells; y <= cy + radiusCells; y++) {
        if (y < 0 || y >= WORLD_HEIGHT) continue;
        for (let x = cx - radiusCells; x <= cx + radiusCells; x++) {
          const wrappedX = (x % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH;
          const target = INDEX(wrappedX, y);
          const dx = x - cx;
          const dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radiusCells) {
            const falloff = Math.exp(-dist / Math.max(1, radiusCells));
            if (event.kind === "uplift") {
              updated[target] += magnitude * falloff;
            } else {
              updated[target] += magnitude * falloff;
            }
          }
        }
      }
    }
  }

  return { relief: updated, aerosol: clamp(aerosolIndex, 0, 1) };
}

function computeSlope(heights: Float32Array) {
  const slope = new Float32Array(WORLD_SIZE);
  let minSlope = Number.POSITIVE_INFINITY;
  let maxSlope = 0;

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = heights[i];
      let maxDrop = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const drop = Math.max(0, h - heights[j]);
          if (drop > maxDrop) {
            maxDrop = drop;
          }
        }
      }
      slope[i] = maxDrop;
      if (maxDrop < minSlope) minSlope = maxDrop;
      if (maxDrop > maxSlope) maxSlope = maxDrop;
    }
  }

  return { slope, minSlope, maxSlope };
}

function computeHydrology(heights: Float32Array, slope: Float32Array) {
  const flowDirection = new Int32Array(WORLD_SIZE);
  flowDirection.fill(-1);
  const flowAccumulation = new Float32Array(WORLD_SIZE);

  for (let i = 0; i < WORLD_SIZE; i++) {
    flowAccumulation[i] = 1;
  }

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = heights[i];
      let bestDrop = 0;
      let best = -1;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const drop = h - heights[j];
          if (drop > bestDrop) {
            bestDrop = drop;
            best = j;
          }
        }
      }
      flowDirection[i] = best;
    }
  }

  const order = Array.from({ length: WORLD_SIZE }, (_, i) => i).sort((a, b) => heights[b] - heights[a]);
  for (const i of order) {
    const to = flowDirection[i];
    if (to >= 0) {
      flowAccumulation[to] += flowAccumulation[i] * (1 + slope[i] / 1000);
    }
  }

  const sorted = Array.from(flowAccumulation).sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, Math.floor(sorted.length * 0.985))] ?? 1;

  const rivers = new Float32Array(WORLD_SIZE);
  const lakes = new Uint8Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    rivers[i] = flowAccumulation[i] / threshold;
    if (flowDirection[i] < 0 && heights[i] < 0 && slope[i] < 40) {
      lakes[i] = 1;
    }
  }

  return { flowDirection, flowAccumulation, rivers, lakes };
}

function computeClimate(
  planet: PlanetInputs,
  heights: Float32Array,
  slope: Float32Array,
  rivers: Float32Array,
  flow: Float32Array,
  aerosol: number,
) {
  const temperature = new Float32Array(WORLD_SIZE);
  const precipitation = new Float32Array(WORLD_SIZE);
  const tiltRad = planet.axialTiltDeg * RADIANS;
  const rotationFactor = 20 / Math.max(1, planet.rotationHours);

  let minTemp = Number.POSITIVE_INFINITY;
  let maxTemp = -Number.POSITIVE_INFINITY;
  let minPrec = Number.POSITIVE_INFINITY;
  let maxPrec = -Number.POSITIVE_INFINITY;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const lat = latByY[i] * RADIANS;
    const baseInsolation = Math.max(0, Math.cos(lat - tiltRad)) * (1 + planet.eccentricity * 0.35);
    const elevationForTemperature = Math.max(0, heights[i]);
    const baseTemp = 55 * baseInsolation - 0.0065 * elevationForTemperature - Math.abs(lat) * 0.4;
    const pressureTerm = 12 * Math.log1p(Math.max(0, planet.atmosphereBar));
    const oceanTerm = heights[i] < 0 ? 6 : 0;
    const seasonal = 1 + 0.08 * Math.sin((lat + 0.2 * Math.PI * (tiltRad)) * 2);
    const temp = baseTemp + pressureTerm + oceanTerm + 0.12 * rotationFactor * 100 * seasonal - aerosol * 8;
    const slopeOrographic = 1 + Math.min(1.4, Math.abs(slope[i]) / 400);
    const precipRaw =
      600 + 1800 * baseInsolation * seasonal + 1200 * Math.min(1, rivers[i]) * slopeOrographic + flow[i] * 0.1;
    const precip = Math.max(0, precipRaw - aerosol * 250);

    temperature[i] = clamp(temp, -90, 70);
    precipitation[i] = precip;
    if (temperature[i] < minTemp) minTemp = temperature[i];
    if (temperature[i] > maxTemp) maxTemp = temperature[i];
    if (precip < minPrec) minPrec = precip;
    if (precip > maxPrec) maxPrec = precip;
  }

  return { temperature, precipitation, minTemp, maxTemp, minPrec, maxPrec };
}

function classifyBiome(temp: number, precip: number, height: number): number {
  if (height < 0) return 0;
  if (temp < -5) return 1;
  if (precip > 1600 && temp >= 5 && temp < 22) return 2;
  if (temp > 12 && precip > 900) return 3;
  if (temp > 24 && precip > 500 && precip <= 1400) return 4;
  if (precip < 400 && temp > 18) return 5;
  if (precip > 700 && temp > 5 && temp < 22) return 6;
  if (height > 1800) return 7;
  return 8;
}

function computeBiomes(temperature: Float32Array, precipitation: Float32Array, heights: Float32Array) {
  const biomes = new Uint8Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    biomes[i] = classifyBiome(temperature[i], precipitation[i], heights[i]);
  }
  return biomes;
}

function computeSettlement(biomes: Uint8Array, heights: Float32Array, temperature: Float32Array, precipitation: Float32Array) {
  const settlement = new Float32Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    if (biomes[i] === 0) {
      settlement[i] = 0;
      continue;
    }
    const comfortT = 1 - Math.abs(temperature[i] - 18) / 45;
    const comfortP = 1 - Math.abs(precipitation[i] - 1400) / 2200;
    const elevationPenalty = Math.max(0, heights[i] - 1200) / 2600;
    const value = Math.max(0, Math.min(1, (comfortT + comfortP) / 2 - elevationPenalty));
    settlement[i] = value;
  }
  return settlement;
}

function evaluateRecompute(reason: RecomputeTrigger): LayerId[] {
  const all: LayerId[] = ["planet", "plates", "relief", "hydrology", "climate", "biomes", "settlement"];
  const active = new Set<LayerId>();

  if (reason === "global") {
    for (const id of all) active.add(id);
  } else if (reason === "tectonics") {
    const dependency: LayerId[] = ["planet", "plates", "relief", "hydrology", "climate", "biomes", "settlement"];
    for (const id of dependency) {
      active.add(id);
    }
  } else {
    const dependency: LayerId[] = ["relief", "hydrology", "climate", "biomes", "settlement"];
    for (const id of dependency) {
      active.add(id);
    }
  }

  return all.filter((id) => active.has(id));
  
}

function ensureEventEnergy(event: WorldEventRecord): WorldEventRecord {
  if (event.kind === "meteorite" && event.energyJoule === undefined) {
    const r = (event.diameterKm * 1000) / 2;
    const mass = (4 / 3) * Math.PI * Math.pow(r, 3) * event.densityKgM3;
    return {
      ...event,
      energyJoule: 0.5 * mass * Math.pow(event.speedKms * 1000, 2),
    };
  }
  return event;
}

function minMax(values: Float32Array) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

export function runSimulation(config: SimulationConfig, reason: RecomputeTrigger = "global"): SimulationResult {
  const planet = { ...config.planet };
  const tectonics = { ...config.tectonics };
  const events = config.events.map((event) => ensureEventEnergy(event as WorldEventRecord));

  const layerOrder = evaluateRecompute(reason);
  const platesLayer = computePlates(planet, tectonics, config.seed);
  const reliefRaw = computeRelief(planet, tectonics, platesLayer, config.seed);
  const eventRelief = applyEvents(planet, reliefRaw.relief, events);
  const { slope, minSlope, maxSlope } = computeSlope(eventRelief.relief);
  const hydro = computeHydrology(eventRelief.relief, slope);
  const climate = computeClimate(
    planet,
    eventRelief.relief,
    slope,
    hydro.rivers,
    hydro.flowAccumulation,
    eventRelief.aerosol,
  );
  const biomes = computeBiomes(climate.temperature, climate.precipitation, eventRelief.relief);
  const settlement = computeSettlement(biomes, eventRelief.relief, climate.temperature, climate.precipitation);
  const heightRange = minMax(eventRelief.relief);

  return {
    width: WORLD_WIDTH,
    height: WORLD_HEIGHT,
    seed: config.seed,
    specs: LAYER_GRAPH.map((s) => ({
      ...s,
      dependsOn: [...s.dependsOn],
    })),
    recomputedLayers: layerOrder,
    planet: {
      seaLevel: reliefRaw.seaLevel,
      radiusKm: planet.radiusKm,
      oceanPercent: planet.oceanPercent,
    },
    plates: platesLayer.plateField,
    boundaryTypes: platesLayer.boundaryTypes,
    heightMap: eventRelief.relief,
    slopeMap: slope,
    riverMap: hydro.rivers,
    lakeMap: hydro.lakes,
    flowDirection: hydro.flowDirection,
    flowAccumulation: hydro.flowAccumulation,
    temperatureMap: climate.temperature,
    precipitationMap: climate.precipitation,
    biomeMap: biomes,
    settlementMap: settlement,
    eventHistory: events,
    stats: {
      minHeight: heightRange.min,
      maxHeight: heightRange.max,
      minTemperature: climate.minTemp,
      maxTemperature: climate.maxTemp,
      minPrecipitation: climate.minPrec,
      maxPrecipitation: climate.maxPrec,
      minSlope,
      maxSlope,
    },
  };
}

export function toHeightExportPayload(
  result: SimulationResult,
  mode: "float32" | "int16",
): HeightExportPayload {
  if (mode === "int16") {
    const scale = 1;
    const values = Array.from(result.heightMap, (v) =>
      Math.round(clamp(v / scale, -32768, 32767)),
    );
    return {
      format: "int16",
      width: result.width,
      height: result.height,
      radiusKm: result.planet.radiusKm,
      scaleFactor: scale,
      values,
    };
  }

  return {
    format: "float32",
    width: result.width,
    height: result.height,
    radiusKm: result.planet.radiusKm,
    values: Array.from(result.heightMap),
  };
}
