/// <reference lib="webworker" />

import type { RecomputeTrigger, SimulationConfig, SimulationResult } from "@/lib/planet/simulation";
import type { runSimulationRust as RunSimulationRustFn } from "@/lib/planet/rust-simulation";

type GenerateMessage = {
  type: "generate";
  requestId: number;
  config: SimulationConfig;
  reason: RecomputeTrigger;
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

const scope = self as DedicatedWorkerGlobalScope;
let runSimulationRustRef: typeof RunSimulationRustFn | null = null;
let fetchPatched = false;

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
  const module = await import("@/lib/planet/rust-simulation");
  runSimulationRustRef = module.runSimulationRust;
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
    const result = runSimulationRust(message.config, message.reason, (percent) => {
      const payload: ProgressMessage = {
        type: "progress",
        requestId: message.requestId,
        progress: percent,
      };
      scope.postMessage(payload);
    });

    const payload: ResultMessage = {
      type: "result",
      requestId: message.requestId,
      result,
    };
    const transfers = transferables(result);
    try {
      // Prefer transfer for large typed arrays, but fall back if runtime rejects transfer options.
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
