import {
  INDEX,
  INDEX_SPHERICAL,
  RADIANS,
  WORLD_HEIGHT,
  WORLD_SIZE,
  WORLD_WIDTH,
  clamp,
  latByY,
  lonByX,
  makeRng,
  randomRange,
  xByCell,
  yByCell,
  zByCell,
} from "./core";
import type { PlanetInputs, TectonicInputs } from "./types";

interface PlateSpec {
  lat: number;
  lon: number;
  speed: number;
  dirX: number;
  dirY: number;
  heat: number;
  buoyancy: number;
}

export interface PlateVector {
  x: number;
  y: number;
  z: number;
  speed: number;
  dirX: number;
  dirY: number;
  heat: number;
  buoyancy: number;
}

export interface PlatesComputation {
  plateField: Int16Array;
  boundaryTypes: Int8Array;
  boundaryScores: Float32Array;
  boundaryNormalX: Float32Array;
  boundaryNormalY: Float32Array;
  boundaryStrength: Float32Array;
  plateVectors: PlateVector[];
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

export function computePlates(
  _planet: PlanetInputs,
  tectonics: TectonicInputs,
  seed: number,
): PlatesComputation {
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
