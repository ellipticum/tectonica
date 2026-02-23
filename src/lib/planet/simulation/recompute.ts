import type { LayerId, RecomputeTrigger } from "./types";

export function evaluateRecompute(reason: RecomputeTrigger): LayerId[] {
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
