/// <reference lib="webworker" />

import type { RecomputeTrigger, SimulationConfig, SimulationResult } from "@/lib/planet/simulation";
import type { runSimulationRust as RunSimulationRustFn } from "@/lib/planet/rust-simulation";

type GenerateMessage = {
  type: "generate";
  requestId: number;
  config: SimulationConfig;
  reason: RecomputeTrigger;
  attempts?: number;
};

type WorkerInput = GenerateMessage;

type ProgressMessage = {
  type: "progress";
  requestId: number;
  progress: number;
};

type ResultMessage = {
  type: "result";
  requestId: number;
  result: SimulationResult;
};

type ErrorMessage = {
  type: "error";
  requestId: number;
  message: string;
};

type WorkerOutput = ProgressMessage | ResultMessage | ErrorMessage;

const scope = globalThis as unknown as DedicatedWorkerGlobalScope;
let runSimulationRustRef: typeof RunSimulationRustFn | null = null;
let fetchPatched = false;

function indexSpherical(x: number, y: number, width: number, height: number): number {
  let xx = x;
  let yy = y;
  while (yy < 0 || yy >= height) {
    if (yy < 0) {
      yy = -yy - 1;
      xx += Math.floor(width / 2);
    } else {
      yy = 2 * height - yy - 1;
      xx += Math.floor(width / 2);
    }
  }
  xx %= width;
  if (xx < 0) {
    xx += width;
  }
  return yy * width + xx;
}

function earthLikeScore(result: SimulationResult, oceanTargetPercent: number): number {
  const width = result.width;
  const height = result.height;
  const size = width * height;
  const heights = result.heightMap;
  const landMask = new Uint8Array(size);
  let totalLand = 0;
  for (let i = 0; i < size; i++) {
    if (heights[i] >= 0) {
      landMask[i] = 1;
      totalLand += 1;
    }
  }
  if (totalLand < size * 0.05 || totalLand > size * 0.95) {
    return -1e9;
  }

  const visited = new Uint8Array(size);
  const queue = new Int32Array(size);
  const componentSizes: number[] = [];
  let tinyLand = 0;
  let coastlineEdges = 0;
  let componentCount = 0;

  const tinyThreshold = Math.max(300, Math.floor(totalLand * 0.005));
  const majorThreshold = Math.max(1200, Math.floor(totalLand * 0.075));
  let majorCount = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (landMask[i] !== 1 || visited[i] !== 0) {
        continue;
      }

      componentCount += 1;
      let head = 0;
      let tail = 0;
      let compSize = 0;
      visited[i] = 1;
      queue[tail++] = i;

      while (head < tail) {
        const node = queue[head++];
        compSize += 1;
        const nx = node % width;
        const ny = Math.floor(node / width);

        const n4 = [
          indexSpherical(nx - 1, ny, width, height),
          indexSpherical(nx + 1, ny, width, height),
          indexSpherical(nx, ny - 1, width, height),
          indexSpherical(nx, ny + 1, width, height),
        ];
        for (const j of n4) {
          if (landMask[j] === 0) {
            coastlineEdges += 1;
          }
        }

        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            if (ox === 0 && oy === 0) {
              continue;
            }
            const j = indexSpherical(nx + ox, ny + oy, width, height);
            if (landMask[j] === 1 && visited[j] === 0) {
              visited[j] = 1;
              queue[tail++] = j;
            }
          }
        }
      }

      componentSizes.push(compSize);
      if (compSize < tinyThreshold) {
        tinyLand += compSize;
      }
      if (compSize >= majorThreshold) {
        majorCount += 1;
      }
    }
  }

  componentSizes.sort((a, b) => b - a);
  const top5 = componentSizes.slice(0, 5).reduce((acc, n) => acc + n, 0);
  const largest = componentSizes[0] ?? 0;

  const tinyLandShare = tinyLand / Math.max(1, totalLand);
  const top5Share = top5 / Math.max(1, totalLand);
  const largestShare = largest / Math.max(1, totalLand);
  const shorelineRatio = coastlineEdges / Math.max(1, totalLand);

  let score = 0;
  score += top5Share * 220;
  score -= tinyLandShare * 300;
  score -= Math.abs(majorCount - 6) * 14;
  score -= Math.max(0, componentCount - 26) * 1.3;
  score -= Math.abs(largestShare - 0.34) * 90;
  score -= Math.abs(shorelineRatio - 2.2) * 24;
  score -= Math.abs(result.planet.oceanPercent - oceanTargetPercent) * 1.5;
  return score;
}

function nextSeed(seed: number): number {
  const n = (Math.imul(seed >>> 0, 1664525) + 1013904223) >>> 0;
  return n === 0 ? 1 : n;
}

function inferWorkerOrigin(): string | null {
  try {
    const origin = scope.location?.origin;
    if (origin && origin !== "null") {
      return origin;
    }

    const href = scope.location?.href;
    if (href && href.startsWith("blob:")) {
      const match = /^blob:(https?:\/\/[^/]+)/i.exec(href);
      if (match?.[1]) {
        return match[1];
      }
    }
  } catch {
    return null;
  }
  return null;
}

function patchFetchForAbsoluteAssetUrls() {
  if (fetchPatched || typeof scope.fetch !== "function") {
    return;
  }

  const origin = inferWorkerOrigin();
  if (!origin) {
    fetchPatched = true;
    return;
  }

  const nativeFetch = scope.fetch.bind(scope);
  (scope as unknown as { fetch: typeof fetch }).fetch = (
    input: RequestInfo | URL,
    init?: RequestInit,
  ) => {
    if (typeof input === "string" && input.startsWith("/")) {
      return nativeFetch(new URL(input, origin).toString(), init);
    }
    if (input instanceof URL && input.protocol === "file:") {
      return nativeFetch(new URL(input.pathname, origin).toString(), init);
    }
    return nativeFetch(input as RequestInfo, init);
  };

  fetchPatched = true;
}

async function getRunSimulationRust(): Promise<typeof RunSimulationRustFn> {
  if (runSimulationRustRef) {
    return runSimulationRustRef;
  }

  patchFetchForAbsoluteAssetUrls();
  const wasmModule = await import("@/lib/planet/rust-simulation");
  runSimulationRustRef = wasmModule.runSimulationRust;
  return runSimulationRustRef;
}

function transferables(result: SimulationResult): Transferable[] {
  return [
    result.plates.buffer,
    result.boundaryTypes.buffer,
    result.heightMap.buffer,
    result.slopeMap.buffer,
    result.riverMap.buffer,
    result.lakeMap.buffer,
    result.flowDirection.buffer,
    result.flowAccumulation.buffer,
    result.temperatureMap.buffer,
    result.precipitationMap.buffer,
    result.biomeMap.buffer,
    result.settlementMap.buffer,
  ];
}

scope.onmessage = async (event: MessageEvent<WorkerInput>) => {
  const message = event.data;
  if (!message || message.type !== "generate") {
    return;
  }

  try {
    const runSimulationRust = await getRunSimulationRust();
    const isIslandScope = message.config.scope === "tasmania";
    const attempts = isIslandScope ? 1 : Math.max(1, Math.min(24, Math.floor(message.attempts ?? 1)));
    let seed = message.config.seed >>> 0;
    let bestResult: SimulationResult | null = null;
    let bestScore = -1e12;

    for (let attempt = 0; attempt < attempts; attempt++) {
      const config = {
        ...message.config,
        seed,
      };
      const result = runSimulationRust(config, message.reason, (percent) => {
        const overall = ((attempt + percent / 100) / attempts) * 100;
        const payload: ProgressMessage = {
          type: "progress",
          requestId: message.requestId,
          progress: overall,
        };
        scope.postMessage(payload);
      });

      const score = isIslandScope ? 0 : earthLikeScore(result, message.config.planet.oceanPercent);
      if (bestResult === null || score > bestScore) {
        if (!isIslandScope) {
          bestScore = score;
        }
        bestResult = result;
      }

      const endPayload: ProgressMessage = {
        type: "progress",
        requestId: message.requestId,
        progress: ((attempt + 1) / attempts) * 100,
      };
      scope.postMessage(endPayload);
      seed = nextSeed(seed);
    }

    const result = bestResult;
    if (!result) {
      throw new Error("failed to generate world");
    }

    const payload: ResultMessage = {
      type: "result",
      requestId: message.requestId,
      result,
    };
    const transfers = transferables(result);
    try {
      scope.postMessage(payload as WorkerOutput, transfers);
    } catch {
      scope.postMessage(payload as WorkerOutput);
    }
  } catch (error) {
    const payload: ErrorMessage = {
      type: "error",
      requestId: message.requestId,
      message: error instanceof Error ? error.message : "simulation failed",
    };
    scope.postMessage(payload as WorkerOutput);
  }
};

export {};
