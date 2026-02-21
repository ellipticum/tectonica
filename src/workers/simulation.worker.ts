/// <reference lib="webworker" />

import type { RecomputeTrigger, SimulationConfig, SimulationResult } from "@/lib/planet/simulation";
import { runSimulationRust } from "@/lib/planet/rust-simulation";

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

scope.onmessage = (event: MessageEvent<WorkerInput>) => {
  const message = event.data;
  if (!message || message.type !== "generate") {
    return;
  }

  try {
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
    scope.postMessage(payload as WorkerOutput, { transfer: transferables(result) });
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
