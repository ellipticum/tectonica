export const WORLD_WIDTH = 360;
export const WORLD_HEIGHT = 180;
export const WORLD_SIZE = WORLD_WIDTH * WORLD_HEIGHT;
export const RADIANS = Math.PI / 180;
export const KILOMETERS_PER_DEGREE = 111.319_490_793; // Equatorial distance around Earth-equivalent sphere.

export interface RNG {
  next: () => number;
}

export function makeRng(seed: number): RNG {
  let state = seed >>> 0;
  return {
    next() {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    },
  };
}

export function clamp(value: number, min: number, max: number) {
  return value < min ? min : value > max ? max : value;
}

export const INDEX = (x: number, y: number) => y * WORLD_WIDTH + x;

export function sphericalWrap(x: number, y: number) {
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

export function INDEX_SPHERICAL(x: number, y: number) {
  const p = sphericalWrap(x, y);
  return INDEX(p.x, p.y);
}

export const latByY = new Float32Array(WORLD_SIZE);
export const lonByX = new Float32Array(WORLD_SIZE);
export const xByCell = new Float32Array(WORLD_SIZE);
export const yByCell = new Float32Array(WORLD_SIZE);
export const zByCell = new Float32Array(WORLD_SIZE);

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

export function randomRange(rng: RNG, min: number, max: number) {
  return min + rng.next() * (max - min);
}

export function quantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  const t = clamp(q, 0, 1) * (sortedValues.length - 1);
  const lo = Math.floor(t);
  const hi = Math.min(sortedValues.length - 1, lo + 1);
  const k = t - lo;
  return sortedValues[lo] * (1 - k) + sortedValues[hi] * k;
}

export function minMax(values: Float32Array) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}
