import type { WorldDisplayLayer } from "@/lib/planet/simulation";

export const LAYER_OPTIONS: Array<{ id: WorldDisplayLayer; label: string }> = [
  { id: "plates", label: "Плиты" },
  { id: "height", label: "Высоты" },
  { id: "slope", label: "Уклон" },
  { id: "rivers", label: "Реки" },
  { id: "precipitation", label: "Осадки" },
  { id: "temperature", label: "Температура" },
  { id: "biomes", label: "Биомы" },
  { id: "settlement", label: "Расселение" },
  { id: "events", label: "События" },
];

export const PREVIEW_PRESETS = [
  { id: "low", label: "Быстро x0.25 (~512×256)", scale: 0.25 },
  { id: "balanced", label: "Баланс x0.5 (~1024×512)", scale: 0.5 },
  { id: "native", label: "Натив (2048×1024)", scale: 1 },
] as const;

export type PreviewPresetId = (typeof PREVIEW_PRESETS)[number]["id"];

export const GENERATION_PRESETS = [
  { id: "ultra", label: "Сверхбыстро (тест)" },
  { id: "fast", label: "Быстро (черновик)" },
  { id: "balanced", label: "Баланс" },
  { id: "detailed", label: "Детально" },
] as const;

export type GenerationPresetId = (typeof GENERATION_PRESETS)[number]["id"];

export const SEED_SEARCH_PRESETS = [
  { id: "1", label: "1 seed (fast)", attempts: 1 },
  { id: "4", label: "4 seeds (selection)", attempts: 4 },
  { id: "8", label: "8 seeds (earth-like)", attempts: 8 },
  { id: "12", label: "12 seeds (max selection)", attempts: 12 },
] as const;

export type SeedSearchPresetId = (typeof SEED_SEARCH_PRESETS)[number]["id"];
