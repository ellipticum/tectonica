"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { BellRing, Layers, Mountain, Waves, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PlanetLayerCanvas, type ProjectionMode } from "@/components/planet/layer-canvas";
import {
  BIOME_COLORS,
  BIOME_NAMES,
  DEFAULT_PLANET,
  DEFAULT_SIMULATION,
  DEFAULT_TECTONICS,
  LAYER_GRAPH,
  type LayerId,
  type RecomputeTrigger,
  type WorldEvent,
  type WorldEventRecord,
  type WorldDisplayLayer,
  toHeightExportPayload,
  type SimulationConfig,
  type SimulationResult,
} from "@/lib/planet/simulation";
import { runSimulationRust } from "@/lib/planet/rust-simulation";

const LAYER_OPTIONS: Array<{ id: WorldDisplayLayer; label: string }> = [
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

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

const PREVIEW_PRESETS = [
  { id: "low", label: "Быстро x0.25 (~512×256)", scale: 0.25 },
  { id: "balanced", label: "Баланс x0.5 (~1024×512)", scale: 0.5 },
  { id: "native", label: "Натив (2048×1024)", scale: 1 },
] as const;

type PreviewPresetId = (typeof PREVIEW_PRESETS)[number]["id"];
const GENERATION_PRESETS = [
  { id: "fast", label: "Быстро (черновик)" },
  { id: "balanced", label: "Баланс" },
  { id: "detailed", label: "Детально" },
] as const;
type GenerationPresetId = (typeof GENERATION_PRESETS)[number]["id"];
type ViewMode = "map" | "globe";
type FlatProjection = "equirectangular" | "mercator";

type SimulationWorkerRequest = {
  type: "generate";
  requestId: number;
  config: SimulationConfig;
  reason: RecomputeTrigger;
};

type SimulationWorkerProgress = {
  type: "progress";
  requestId: number;
  progress: number;
};

type SimulationWorkerResult = {
  type: "result";
  requestId: number;
  result: SimulationResult;
};

type SimulationWorkerError = {
  type: "error";
  requestId: number;
  message: string;
};

type SimulationWorkerEvent =
  | SimulationWorkerProgress
  | SimulationWorkerResult
  | SimulationWorkerError;

type ColorStop = {
  t: number;
  color: [number, number, number];
};

const OCEAN_STOPS: ColorStop[] = [
  { t: 0, color: [176, 206, 232] },
  { t: 0.18, color: [146, 184, 220] },
  { t: 0.45, color: [95, 145, 198] },
  { t: 0.72, color: [48, 93, 155] },
  { t: 1, color: [15, 40, 95] },
];

const LAND_STOPS: ColorStop[] = [
  { t: 0, color: [101, 196, 108] },
  { t: 0.2, color: [168, 214, 112] },
  { t: 0.42, color: [230, 213, 119] },
  { t: 0.62, color: [216, 164, 95] },
  { t: 0.8, color: [190, 116, 90] },
  { t: 0.9, color: [210, 161, 166] },
  { t: 1, color: [247, 245, 242] },
];

const formatNum = (value: number) => value.toLocaleString("ru-RU", { maximumFractionDigits: 2 });
type LegendItem = { color: string; label: string };

const hslToCss = (hue: number, saturation: number, lightness: number) =>
  `hsl(${hue.toFixed(0)} ${Math.round(saturation)}% ${Math.round(lightness)}%)`;

const rgbCss = (rgb: [number, number, number]) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

const sampleStops = (stops: ColorStop[], t: number) => {
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
      ] as [number, number, number];
    }
  }

  return stops[stops.length - 1]?.color ?? [255, 255, 255];
};

const heightColor = (value: number, min: number, max: number) => {
  if (value < 0 && max > 0) {
    const t = Math.pow(clamp(-value / Math.max(1, -min), 0, 1), 0.68);
    return rgbCss(sampleStops(OCEAN_STOPS, t));
  }

  if (max <= 0) {
    return rgbCss(sampleStops(OCEAN_STOPS, 1));
  }

  const t = clamp(value / Math.max(1, max), 0, 1);
  return rgbCss(sampleStops(LAND_STOPS, t));
};

const buildHeightGradient = (minHeight: number, maxHeight: number) => {
  if (maxHeight <= 0) {
    const oceanOnlyStops = OCEAN_STOPS.map((s) => {
      const color = sampleStops(OCEAN_STOPS, 1 - s.t);
      return `${rgbCss(color)} ${(s.t * 100).toFixed(2)}%`;
    });
    return `linear-gradient(90deg, ${oceanOnlyStops.join(", ")})`;
  }
  if (minHeight >= 0) {
    return `linear-gradient(90deg, ${LAND_STOPS.map((s) => `${rgbCss(s.color)} ${(s.t * 100).toFixed(2)}%`).join(", ")})`;
  }

  const seaSplit = clamp((0 - minHeight) / Math.max(1, maxHeight - minHeight), 0.08, 0.92);
  const oceanStops = OCEAN_STOPS.map((s) => {
    const color = sampleStops(OCEAN_STOPS, 1 - s.t);
    return `${rgbCss(color)} ${(s.t * seaSplit * 100).toFixed(2)}%`;
  });
  const landStops = LAND_STOPS.map((s) =>
    `${rgbCss(s.color)} ${(seaSplit * 100 + s.t * (1 - seaSplit) * 100).toFixed(2)}%`,
  );
  return `linear-gradient(90deg, ${[...oceanStops, ...landStops].join(", ")})`;
};

const slopeLegendColor = (slope: number, maxSlope: number) => {
  const t = clamp(slope, 0, maxSlope) / Math.max(1, maxSlope);
  return hslToCss(25 + t * 120, 80, 18 + t * 50);
};

const climateLegendColor = (value: number, min: number, max: number, temp = false) => {
  const t = (value - min) / Math.max(1, max - min);
  if (temp) {
    return hslToCss(240 - t * 260, 75, 20 + t * 55);
  }
  return hslToCss(210 - t * 180, 85, 20 + t * 55);
};

const buildLayerLegend = (layer: WorldDisplayLayer, result: SimulationResult): LegendItem[] => {
  const minHeight = result.stats.minHeight;
  const maxHeight = result.stats.maxHeight;
  const maxSlope = result.stats.maxSlope;
  const minTemp = result.stats.minTemperature;
  const maxTemp = result.stats.maxTemperature;
  const minPrec = result.stats.minPrecipitation;
  const maxPrec = result.stats.maxPrecipitation;

  if (layer === "biomes") {
    return BIOME_NAMES.map((name, index) => {
      const color = BIOME_COLORS[index] ?? [255, 255, 255];
      return {
        color: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
        label: `${index}. ${name}`,
      };
    });
  }

  if (layer === "plates") {
    const plateSet = new Set(result.plates);
    const plateIds = Array.from(plateSet).sort((a, b) => a - b);
    const limit = Math.min(plateIds.length, 12);

    const entries: LegendItem[] = plateIds.slice(0, limit).map((plateId) => {
      const hue = ((plateId / Math.max(1, plateIds.length)) * 360 + 20) % 360;
      return {
        color: hslToCss(hue, 60, 42),
        label: `Плита ${plateId + 1}`,
      };
    });

    if (plateIds.length > limit) {
      entries.push({
        color: "rgba(255,255,255,0.25)",
        label: `...и еще ${plateIds.length - limit} шт.`,
      });
    }

    return entries;
  }

  if (layer === "height") {
    const vMin = minHeight;
    const vMax = maxHeight;
    return [
      { color: heightColor(vMin, vMin, vMax), label: `Глубокий океан (min ${formatNum(vMin)} м)` },
      { color: heightColor(vMin * 0.35, vMin, vMax), label: "Мелкая вода (светло-голубая)" },
      { color: heightColor(vMin * 0.75, vMin, vMax), label: "Глубокая вода (тёмно-голубая)" },
      { color: heightColor(0, vMin, vMax), label: "Уровень моря: 0 м (тёмно-бледно-жёлтый)" },
      { color: heightColor(vMax * 0.2, vMin, vMax), label: "Низины суши (приподнятая кора)" },
      { color: heightColor(vMax * 0.55, vMin, vMax), label: "Хребты и холмы" },
      { color: heightColor(vMax * 0.85, vMin, vMax), label: "Вершины (жёлтый)" },
      { color: heightColor(vMax, vMin, vMax), label: `Пики: ~${formatNum(vMax)} м (белый)` },
    ];
  }

  if (layer === "slope") {
    const max = Math.max(1, maxSlope);
    return [
      { color: slopeLegendColor(0, maxSlope), label: "Пологое / слабый уклон" },
      { color: slopeLegendColor(0.2 * max, maxSlope), label: "Лёгкий уклон" },
      { color: slopeLegendColor(0.45 * max, maxSlope), label: "Умеренный уклон" },
      { color: slopeLegendColor(0.7 * max, maxSlope), label: "Крутой" },
      { color: slopeLegendColor(max, maxSlope), label: "Очень крутой" },
    ];
  }

  if (layer === "rivers") {
    return [
      { color: "rgb(18, 38, 84)", label: "Низкая/низкоуровневая вода" },
      { color: "rgb(44, 108, 140)", label: "Озера (впадины)" },
      { color: "rgb(128, 174, 255)", label: "Река, средняя сила потока" },
      { color: "rgb(245, 170, 255)", label: "Сильные русла" },
      { color: "rgb(14, 14, 14)", label: "Остальные области (подложка)" },
    ];
  }

  if (layer === "precipitation") {
    const max = maxPrec;
    const min = minPrec;
    return [
      { color: climateLegendColor(minPrec, min, max), label: "Сухо" },
      { color: climateLegendColor(minPrec + (max - min) * 0.25, min, max), label: "Низкие осадки" },
      { color: climateLegendColor(minPrec + (max - min) * 0.5, min, max), label: "Умеренные осадки" },
      { color: climateLegendColor(minPrec + (max - min) * 0.75, min, max), label: "Высокие осадки" },
      { color: climateLegendColor(max, min, max), label: "Экстремально влажные" },
    ];
  }

  if (layer === "temperature") {
    return [
      { color: climateLegendColor(minTemp, minTemp, maxTemp, true), label: "Самый холодный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.25, minTemp, maxTemp, true), label: "Холодный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.5, minTemp, maxTemp, true), label: "Умеренный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.75, minTemp, maxTemp, true), label: "Тёплый" },
      { color: climateLegendColor(maxTemp, minTemp, maxTemp, true), label: "Очень тёплый" },
    ];
  }

  if (layer === "settlement") {
    return [
      { color: "rgb(16, 16, 16)", label: "Нет поселений" },
      { color: "rgb(90, 90, 90)", label: "Слабая активность" },
      { color: "rgb(145, 145, 145)", label: "Умеренная активность" },
      { color: "rgb(190, 190, 190)", label: "Развитая колония" },
      { color: "rgb(236, 236, 236)", label: "Максимальная плотность" },
    ];
  }

  return [
    { color: "rgb(14, 14, 14)", label: "Подложка" },
    { color: "rgb(70, 80, 130)", label: "Признаки гидро- и рельефной динамики" },
    { color: "rgb(0, 190, 255)", label: "Сильные потоки после пересчета событий" },
  ];
};

export default function HomePage() {
  const [planet, setPlanet] = useState(DEFAULT_PLANET);
  const [tectonics, setTectonics] = useState(DEFAULT_TECTONICS);
  const [events, setEvents] = useState<WorldEventRecord[]>([]);
  const [seed, setSeed] = useState(DEFAULT_SIMULATION.seed);
  const [reason, setReason] = useState<RecomputeTrigger>("global");
  const [displayLayer, setDisplayLayer] = useState<WorldDisplayLayer>("height");
  const [inspectLayer, setInspectLayer] = useState<LayerId>("relief");
  const [showConstraint, setShowConstraint] = useState(true);
  const [previewMode, setPreviewMode] = useState<PreviewPresetId>("balanced");
  const [generationPreset, setGenerationPreset] = useState<GenerationPresetId>(
    (DEFAULT_SIMULATION.generationPreset as GenerationPresetId) ?? "balanced",
  );
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [flatProjection, setFlatProjection] = useState<FlatProjection>("equirectangular");
  const [viewCenterLon, setViewCenterLon] = useState(0);
  const [viewCenterLat, setViewCenterLat] = useState(0);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const simulationWorkerRef = useRef<Worker | null>(null);
  const generationRequestIdRef = useRef(0);

  const [meteorForm, setMeteorForm] = useState({
    latitude: 8,
    longitude: 33,
    diameterKm: 9,
    speedKms: 22,
    angleDeg: 46,
    densityKgM3: 3200,
  });

  const [regionForm, setRegionForm] = useState({
    latitude: -14,
    longitude: 84,
    radiusKm: 500,
    magnitude: 400,
  });

  const [result, setResult] = useState<SimulationResult>(() =>
    runSimulationRust(
      {
        seed: DEFAULT_SIMULATION.seed,
        planet: DEFAULT_PLANET,
        tectonics: DEFAULT_TECTONICS,
        events: [],
        generationPreset: (DEFAULT_SIMULATION.generationPreset as GenerationPresetId) ?? "balanced",
      },
      "global",
    ),
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [generationProgress, setGenerationProgress] = useState<number>(0);
  const [generationError, setGenerationError] = useState<string | null>(null);

  const createSimulationWorker = () => {
    const worker = new Worker(new URL("../workers/simulation.worker.ts", import.meta.url), {
      type: "module",
    });

    worker.onmessage = (event: MessageEvent<SimulationWorkerEvent>) => {
      const message = event.data;
      if (!message || message.requestId !== generationRequestIdRef.current) {
        return;
      }

      if (message.type === "progress") {
        setGenerationError(null);
        setGenerationProgress(clamp(message.progress, 0, 100));
        return;
      }

      if (message.type === "result") {
        setResult(message.result);
        setIsDirty(false);
        setGenerationProgress(100);
        setIsGenerating(false);
        setGenerationError(null);
        return;
      }

      setIsGenerating(false);
      setGenerationProgress(0);
      setGenerationError(message.message || "Ошибка генерации");
      // eslint-disable-next-line no-console
      console.error(message.message);
    };

    const handleWorkerFailure = () => {
      setIsGenerating(false);
      setGenerationProgress(0);
      setGenerationError("Ошибка worker во время генерации");
    };

    worker.onerror = handleWorkerFailure;
    worker.onmessageerror = handleWorkerFailure;

    simulationWorkerRef.current = worker;
    return worker;
  };

  useEffect(() => {
    const worker = createSimulationWorker();
    return () => {
      worker.terminate();
      simulationWorkerRef.current = null;
    };
  }, []);

  const generateWorld = () => {
    if (isGenerating) {
      return;
    }

    let worker = simulationWorkerRef.current;
    if (!worker) {
      worker = createSimulationWorker();
    }
    if (!worker) {
      return;
    }

    const shouldGenerateNewSeed = !isDirty;
    const nextSeed = shouldGenerateNewSeed ? Math.floor(Math.random() * 2_147_483_647) : seed;
    if (shouldGenerateNewSeed) {
      setSeed(nextSeed);
    }

    const config: SimulationConfig = {
      seed: nextSeed,
      planet,
      tectonics,
      events,
      generationPreset,
    };

    const requestId = generationRequestIdRef.current + 1;
    generationRequestIdRef.current = requestId;
    setGenerationProgress(0);
    setIsGenerating(true);
    setGenerationError(null);

    const payload: SimulationWorkerRequest = {
      type: "generate",
      requestId,
      config,
      reason,
    };
    worker.postMessage(payload);
  };
  const inspectNodeIndex = useMemo(() => {
    const x = Math.floor(result.width / 2);
    const y = Math.floor(result.height / 2);
    return y * result.width + x;
  }, [result.width, result.height]);

  const inspectValue = useMemo(() => {
    const i = inspectNodeIndex;
    const toNum = (value: number | undefined) => {
      if (value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
        return "—";
      }
      return value.toFixed(2);
    };

    switch (inspectLayer) {
      case "planet":
        return `Радиус: ${result.planet.radiusKm} км, g: ${planet.gravity}, океан: ${result.planet.oceanPercent}%`;
      case "plates":
        return `Плита: ${result.plates[i]}`;
      case "relief":
        return `Рельеф: ${toNum(result.heightMap[i])} м`;
      case "hydrology":
        return `Река: ${toNum(result.riverMap[i])}, бассейн: ${result.lakeMap[i] ? "да" : "нет"}`;
      case "climate":
        return `T: ${toNum(result.temperatureMap[i])}°C, P: ${toNum(result.precipitationMap[i])}`;
      case "biomes":
        return `Биом: ${BIOME_NAMES[result.biomeMap[i]] || "—"}`;
      case "settlement":
        return `Расселение: ${toNum(result.settlementMap[i])}`;
      default:
        return `События: heightFlow=${toNum(result.riverMap[i])}, dir=${result.flowDirection[i]}`;
    }
  }, [inspectLayer, inspectNodeIndex, planet.gravity, result]);

  const preview = useMemo(
    () => PREVIEW_PRESETS.find((item) => item.id === previewMode) ?? PREVIEW_PRESETS[0],
    [previewMode],
  );
  const projectionMode: ProjectionMode = viewMode === "globe" ? "orthographic" : flatProjection;
  const deferredDisplayLayer = useDeferredValue(displayLayer);
  const deferredProjectionMode = useDeferredValue(projectionMode);
  const deferredPreviewScale = useDeferredValue(preview.scale);
  const deferredViewCenterLon = useDeferredValue(viewCenterLon);
  const deferredViewCenterLat = useDeferredValue(viewCenterLat);
  const layerLegend = useMemo(() => buildLayerLegend(deferredDisplayLayer, result), [deferredDisplayLayer, result]);

  const setPlanetField = (patch: Partial<typeof planet>) => {
    setPlanet((prev) => ({ ...prev, ...patch }));
    setReason("global");
    setIsDirty(true);
  };

  const setTectonicsField = (patch: Partial<typeof tectonics>) => {
    setTectonics((prev) => ({ ...prev, ...patch }));
    setReason("tectonics");
    setIsDirty(true);
  };

  const addEventRecord = (event: WorldEvent, summary: string, energy?: number) => {
    const record: WorldEventRecord = {
      ...event,
      id: `ev-${Date.now()}-${Math.round(Math.random() * 1_000_000)}`,
      createdAt: new Date().toLocaleString("ru-RU"),
      summary,
      energyJoule: energy,
    };
    setEvents((prev) => [...prev, record]);
    setReason("events");
    setIsDirty(true);
  };

  const addMeteor = () => {
    const event: WorldEvent = {
      kind: "meteorite",
      latitude: clamp(meteorForm.latitude, -90, 90),
      longitude: clamp(meteorForm.longitude, -180, 180),
      diameterKm: clamp(meteorForm.diameterKm, 0.2, 120),
      speedKms: clamp(meteorForm.speedKms, 0.1, 150),
      angleDeg: clamp(meteorForm.angleDeg, 1, 89),
      densityKgM3: clamp(meteorForm.densityKgM3, 100, 19_000),
    };
    const r = (event.diameterKm * 500) / 2;
    const mass = (4 / 3) * Math.PI * Math.pow(r * 1000, 3) * event.densityKgM3;
    const energy = 0.5 * mass * Math.pow(event.speedKms * 1000, 2);
    addEventRecord(
      event,
      `Метеорит: D=${event.diameterKm} км, v=${event.speedKms} км/с, угол=${event.angleDeg}°`,
      energy,
    );
  };

  const addRegion = (kind: "rift" | "subduction" | "uplift") => {
    const event: WorldEvent = {
      kind,
      latitude: clamp(regionForm.latitude, -90, 90),
      longitude: clamp(regionForm.longitude, -180, 180),
      radiusKm: clamp(regionForm.radiusKm, 50, 3000),
      magnitude: clamp(regionForm.magnitude, 20, 2600),
    };
    const names = {
      rift: "Рифт",
      subduction: "Субдукция",
      uplift: "Поднятие региона",
    };
    addEventRecord(event, `${names[kind]} · радиус ${event.radiusKm} км, мощность ${event.magnitude}`);
  };

  const addOceanShift = () => {
    const level = clamp(regionForm.magnitude, 10, 2500) / 100;
    addEventRecord(
      {
        kind: "oceanShift",
        latitude: clamp(regionForm.latitude, -90, 90),
        longitude: clamp(regionForm.longitude, -180, 180),
        radiusKm: 0,
        magnitude: level,
      },
      `Смена уровня океана на ${level.toFixed(2)} м`,
    );
  };

  const removeEvent = (id: string) => {
    setEvents((prev) => prev.filter((item) => item.id !== id));
    setReason("events");
    setIsDirty(true);
  };

  const download = (mode: "float32" | "int16") => {
    const payload = toHeightExportPayload(result, mode);
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `height_${mode}_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const downloadPreviewImage = (format: "png" | "jpg") => {
    const canvas = previewCanvasRef.current;
    if (!canvas) {
      return;
    }
    const mimeType = format === "png" ? "image/png" : "image/jpeg";
    const dataUrl = canvas.toDataURL(mimeType, format === "jpg" ? 0.92 : undefined);
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = `preview_${displayLayer}_${new Date().toISOString().replace(/[:.]/g, "-")}.${format}`;
    anchor.click();
  };

  return (
    <main className="flex min-h-screen w-full flex-col gap-4 p-3 pb-12 md:p-6 lg:p-8">
      <section className="rounded-3xl border border-white/10 bg-card/90 p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-cyan-200/80">Процедурный геофизический конструктор</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Planetary Causality Engine</h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-200">
              Любые правки происходят через параметры и события. Ручное рисование запрещено для сохранения физической причинности.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={generateWorld} disabled={isGenerating}>
              {isGenerating
                ? `Генерация ${Math.round(generationProgress)}%`
                : isDirty
                  ? "Сгенерировать"
                  : "Новый мир"}
            </Button>
            <Button onClick={() => download("float32")}>Экспорт 32-bit</Button>
            <Button variant="secondary" onClick={() => download("int16")}>
              Экспорт 16-bit
            </Button>
            <Button variant="outline" onClick={() => downloadPreviewImage("png")}>
              Экспорт PNG
            </Button>
            <Button variant="outline" onClick={() => downloadPreviewImage("jpg")}>
              Экспорт JPG
            </Button>
          </div>
        </div>
        {generationError ? (
          <p className="mt-2 text-sm text-rose-300">{generationError}</p>
        ) : null}
        <Separator className="my-4 bg-white/20" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-slate-400 text-xs">Высота, м</p>
            <p className="text-white">
              {formatNum(result.stats.minHeight)} / {formatNum(result.stats.maxHeight)}
            </p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Температура, °C</p>
            <p className="text-white">
              {formatNum(result.stats.minTemperature)} / {formatNum(result.stats.maxTemperature)}
            </p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Океанический порог</p>
            <p className="text-white">{result.planet.oceanPercent}%</p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Событий</p>
            <p className="text-white">{events.length}</p>
          </div>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-[320px_minmax(0,1fr)] xl:grid-cols-[320px_minmax(0,1fr)_320px]">
        <section className="space-y-4 md:order-2 xl:order-2">

          <Card className="border border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle className="text-white">Карта мира</CardTitle>
              <CardDescription className="text-slate-300">Слой и проекция настраиваются в правом сайдбаре</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative">
                <PlanetLayerCanvas
                  result={result}
                  layer={deferredDisplayLayer}
                  previewScale={deferredPreviewScale}
                  projection={deferredProjectionMode}
                  centerLongitudeDeg={deferredViewCenterLon}
                  centerLatitudeDeg={viewMode === "globe" ? deferredViewCenterLat : 0}
                  externalCanvasRef={previewCanvasRef}
                />
                {isGenerating ? (
                  <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-slate-950/58 backdrop-blur-[1px]">
                    <div className="w-[min(86%,440px)] rounded-xl border border-white/20 bg-slate-900/90 p-4 shadow-lg">
                      <div className="mb-2 flex items-center justify-between text-sm">
                        <span className="text-slate-200">Генерация мира</span>
                        <span className="font-semibold text-cyan-200">{Math.round(generationProgress)}%</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-700/70">
                        <div
                          className="h-full rounded-full bg-cyan-400 transition-[width] duration-150"
                          style={{ width: `${clamp(generationProgress, 0, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
              <div>
                <p className="text-xs text-slate-400">Легенда</p>
                {deferredDisplayLayer === "height" ? (
                  <HeightGradientLegend minHeight={result.stats.minHeight} maxHeight={result.stats.maxHeight} />
                ) : (
                  <div className="mt-1 grid gap-2 text-xs sm:grid-cols-3">
                    {layerLegend.map((item) => (
                      <div
                        key={`${item.label}-${item.color}`}
                        className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-2 py-1"
                      >
                        <span className="inline-block size-3 rounded-full" style={{ backgroundColor: item.color }} />
                        <span className="text-slate-200">{item.label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="text-xs text-slate-300">Пересчитаны слои: {result.recomputedLayers.join(", ")}</CardFooter>
          </Card>

          <Card className="border border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle className="text-white">Граф зависимостей</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Слой</TableHead>
                    <TableHead>Формула</TableHead>
                    <TableHead>Статус</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {LAYER_GRAPH.map((layer) => (
                    <TableRow key={layer.id}>
                      <TableCell>
                        <p className="text-white">{layer.name}</p>
                        <p className="text-xs text-slate-400">Зависит от: {layer.dependsOn.join(", ") || "—"}</p>
                      </TableCell>
                      <TableCell className="text-xs text-slate-300">{layer.formula}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={result.recomputedLayers.includes(layer.id) ? "bg-cyan-300/20 text-white" : ""}
                        >
                          {result.recomputedLayers.includes(layer.id) ? "пересчитан" : "не изменялся"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </section>

        <aside className="space-y-4 self-start md:sticky md:top-6 md:order-1">
          <Card className="border border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle className="text-white">Генерация</CardTitle>
              <CardDescription className="text-slate-300">Параметры качества расчета мира</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1.5">
              <Label>Пресет генерации</Label>
              <Select
                value={generationPreset}
                onValueChange={(value) => setGenerationPreset(value as GenerationPresetId)}
                disabled={isGenerating}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Выберите пресет" />
                </SelectTrigger>
                <SelectContent>
                  {GENERATION_PRESETS.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          <Tabs defaultValue="planet" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="planet" className="gap-2 text-xs">
                <Mountain className="h-4 w-4" /> Планета
              </TabsTrigger>
              <TabsTrigger value="tectonics" className="gap-2 text-xs">
                <Layers className="h-4 w-4" /> Тектоника
              </TabsTrigger>
              <TabsTrigger value="events" className="gap-2 text-xs">
                <BellRing className="h-4 w-4" /> События
              </TabsTrigger>
            </TabsList>

            <TabsContent value="planet">
              <Card className="border border-white/10 bg-card/80">
                <CardHeader>
                  <CardTitle className="text-white">Планетарные константы</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <NumericRange
                    label="Радиус, км"
                    min={1000}
                    max={20000}
                    step={10}
                    value={planet.radiusKm}
                    onChange={(value) => setPlanetField({ radiusKm: value })}
                  />
                  <NumericRange
                    label="g, м/с²"
                    min={1}
                    max={25}
                    step={0.1}
                    value={planet.gravity}
                    onChange={(value) => setPlanetField({ gravity: value })}
                  />
                  <NumericRange
                    label="Плотность, кг/м³"
                    min={2500}
                    max={12000}
                    step={10}
                    value={planet.density}
                    onChange={(value) => setPlanetField({ density: value })}
                  />
                  <NumericRange
                    label="Период вращения, ч"
                    min={1}
                    max={300}
                    step={1}
                    value={planet.rotationHours}
                    onChange={(value) => setPlanetField({ rotationHours: value })}
                  />
                  <NumericRange
                    label="Наклон оси, °"
                    min={0}
                    max={90}
                    step={0.1}
                    value={planet.axialTiltDeg}
                    onChange={(value) => setPlanetField({ axialTiltDeg: value })}
                  />
                  <NumericRange
                    label="Эксцентриситет"
                    min={0}
                    max={0.9}
                    step={0.001}
                    value={planet.eccentricity}
                    onChange={(value) => setPlanetField({ eccentricity: value })}
                  />
                  <NumericRange
                    label="Атмосфера, бар"
                    min={0.1}
                    max={8}
                    step={0.05}
                    value={planet.atmosphereBar}
                    onChange={(value) => setPlanetField({ atmosphereBar: value })}
                  />
                  <NumericRange
                    label="Океанский %"
                    min={0}
                    max={100}
                    step={1}
                    value={planet.oceanPercent}
                    onChange={(value) => setPlanetField({ oceanPercent: value })}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="tectonics">
              <Card className="border border-white/10 bg-card/80">
                <CardHeader>
                  <CardTitle className="text-white">Параметры тектоники</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <NumericRange
                    label="Число плит"
                    min={5}
                    max={20}
                    step={1}
                    value={tectonics.plateCount}
                    onChange={(value) => setTectonicsField({ plateCount: Math.round(value) })}
                  />
                  <NumericRange
                    label="Скорость плит, см/год"
                    min={0.1}
                    max={20}
                    step={0.1}
                    value={tectonics.plateSpeedCmPerYear}
                    onChange={(value) => setTectonicsField({ plateSpeedCmPerYear: value })}
                  />
                  <NumericRange
                    label="Тепловой поток"
                    min={10}
                    max={120}
                    step={1}
                    value={tectonics.mantleHeat}
                    onChange={(value) => setTectonicsField({ mantleHeat: value })}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="events">
              <Card className="border border-white/10 bg-card/80">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-white">
                    <Waves className="h-4 w-4" /> Событийная редакция
                  </CardTitle>
                  <CardDescription className="text-slate-300">Причинные правки: метеорит, рифт, субдукция, подъем, уровень океана</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="rounded-lg border border-cyan-300/25 bg-slate-900/20 p-2">
                    <p className="mb-2 text-sm text-slate-100">Метеорит</p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      <LabeledInput
                        label="Широта"
                        value={meteorForm.latitude}
                        min={-90}
                        max={90}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, latitude: value }))}
                      />
                      <LabeledInput
                        label="Долгота"
                        value={meteorForm.longitude}
                        min={-180}
                        max={180}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, longitude: value }))}
                      />
                      <LabeledInput
                        label="Диаметр, км"
                        value={meteorForm.diameterKm}
                        min={0.2}
                        max={120}
                        step={0.1}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, diameterKm: value }))}
                      />
                      <LabeledInput
                        label="Скорость, км/с"
                        value={meteorForm.speedKms}
                        min={0.1}
                        max={150}
                        step={0.1}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, speedKms: value }))}
                      />
                      <LabeledInput
                        label="Угол, град"
                        value={meteorForm.angleDeg}
                        min={1}
                        max={89}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, angleDeg: value }))}
                      />
                      <LabeledInput
                        label="Плотность, кг/м3"
                        value={meteorForm.densityKgM3}
                        min={100}
                        max={19000}
                        onChange={(value) => setMeteorForm((prev) => ({ ...prev, densityKgM3: value }))}
                      />
                    </div>
                    <Button className="mt-2 w-full" onClick={addMeteor}>
                      Добавить удар
                    </Button>
                  </div>

                  <div className="rounded-lg border border-amber-300/25 bg-slate-900/20 p-2">
                    <p className="mb-2 text-sm text-slate-100">Регионы</p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      <LabeledInput
                        label="Широта"
                        value={regionForm.latitude}
                        min={-90}
                        max={90}
                        onChange={(value) => setRegionForm((prev) => ({ ...prev, latitude: value }))}
                      />
                      <LabeledInput
                        label="Долгота"
                        value={regionForm.longitude}
                        min={-180}
                        max={180}
                        onChange={(value) => setRegionForm((prev) => ({ ...prev, longitude: value }))}
                      />
                      <LabeledInput
                        label="Радиус, км"
                        value={regionForm.radiusKm}
                        min={50}
                        max={3000}
                        step={5}
                        onChange={(value) => setRegionForm((prev) => ({ ...prev, radiusKm: value }))}
                      />
                      <LabeledInput
                        label="Мощность"
                        value={regionForm.magnitude}
                        min={20}
                        max={2600}
                        step={10}
                        onChange={(value) => setRegionForm((prev) => ({ ...prev, magnitude: value }))}
                      />
                    </div>
                    <div className="mt-2 grid gap-2 sm:grid-cols-4">
                      <Button variant="secondary" onClick={() => addRegion("rift")}>
                        Рифт
                      </Button>
                      <Button variant="secondary" onClick={() => addRegion("subduction")}>
                        Субдукция
                      </Button>
                      <Button variant="secondary" onClick={() => addRegion("uplift")}>
                        Поднять
                      </Button>
                      <Button variant="outline" onClick={addOceanShift}>
                        Океан
                      </Button>
                    </div>
                  </div>

                  <div className="rounded-lg border border-emerald-300/25 bg-slate-900/20 p-2">
                    <div className="flex items-center gap-2">
                      <Switch checked={showConstraint} onCheckedChange={setShowConstraint} />
                      <Label>Показывать ограничения физичности</Label>
                    </div>
                    {showConstraint ? (
                      <p className="mt-2 text-xs text-emerald-100">
                        Ручное рисование рек и гор отключено. Любое изменение возможно только через параметры планеты, тектоники или событие.
                      </p>
                    ) : null}
                  </div>

                  <div>
                    <Label>История событий</Label>
                    <div className="mt-2 space-y-2 max-h-52 overflow-auto">
                      {events.length === 0 ? <p className="text-xs text-slate-400">Пока событий нет.</p> : null}
                      {events.map((item) => (
                        <div key={item.id} className="rounded-md border border-white/15 bg-slate-900/20 p-2 text-xs text-slate-200">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span>
                              {item.kind.toUpperCase()} · {item.createdAt}
                            </span>
                            <Button size="icon" variant="ghost" onClick={() => removeEvent(item.id)}>
                              <X className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                          <p>{item.summary}</p>
                          {item.energyJoule ? <p>E={item.energyJoule.toExponential(2)} J</p> : null}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </aside>

        <aside className="space-y-4 self-start md:order-3 xl:sticky xl:top-6">
          <Card className="border border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle className="text-white">Просмотр</CardTitle>
              <CardDescription className="text-slate-300">Слой, проекция и центр камеры</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as ViewMode)}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="map">Карта</TabsTrigger>
                  <TabsTrigger value="globe">Глобус</TabsTrigger>
                </TabsList>
              </Tabs>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                <div className="space-y-1.5">
                  <Label>Слой карты</Label>
                  <Select value={displayLayer} onValueChange={(value) => setDisplayLayer(value as WorldDisplayLayer)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LAYER_OPTIONS.map((item) => (
                        <SelectItem key={item.id} value={item.id}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Инспекция узла</Label>
                  <Select value={inspectLayer} onValueChange={(value) => setInspectLayer(value as LayerId)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LAYER_GRAPH.map((item) => (
                        <SelectItem key={item.id} value={item.id}>
                          {item.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Проекция</Label>
                  {viewMode === "map" ? (
                    <Select value={flatProjection} onValueChange={(value) => setFlatProjection(value as FlatProjection)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="equirectangular">Equirectangular</SelectItem>
                        <SelectItem value="mercator">Mercator</SelectItem>
                      </SelectContent>
                    </Select>
                  ) : (
                    <div className="flex h-10 items-center rounded-md border border-input bg-background px-3 text-sm text-slate-200">
                      Orthographic sphere
                    </div>
                  )}
                </div>
                <div className="space-y-1.5">
                  <Label>Качество превью</Label>
                  <Select value={previewMode} onValueChange={(value) => setPreviewMode(value as PreviewPresetId)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PREVIEW_PRESETS.map((item) => (
                        <SelectItem key={item.id} value={item.id}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid gap-3">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Label className="text-slate-300">Центральный меридиан, °</Label>
                    <span className="text-xs text-cyan-200">{viewCenterLon.toFixed(0)}</span>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-[1fr_auto] xl:grid-cols-1">
                    <Slider
                      value={[viewCenterLon]}
                      min={-180}
                      max={180}
                      step={1}
                      onValueChange={(vals) => setViewCenterLon(clamp(vals[0] ?? 0, -180, 180))}
                    />
                    <Input
                      type="number"
                      value={viewCenterLon}
                      min={-180}
                      max={180}
                      step={1}
                      onChange={(event) => {
                        const next = Number(event.target.value);
                        if (Number.isFinite(next)) {
                          setViewCenterLon(clamp(next, -180, 180));
                        }
                      }}
                      className="w-24 xl:w-full"
                    />
                  </div>
                </div>
                {viewMode === "globe" ? (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <Label className="text-slate-300">Центральная широта, °</Label>
                      <span className="text-xs text-cyan-200">{viewCenterLat.toFixed(0)}</span>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-[1fr_auto] xl:grid-cols-1">
                      <Slider
                        value={[viewCenterLat]}
                        min={-85}
                        max={85}
                        step={1}
                        onValueChange={(vals) => setViewCenterLat(clamp(vals[0] ?? 0, -85, 85))}
                      />
                      <Input
                        type="number"
                        value={viewCenterLat}
                        min={-85}
                        max={85}
                        step={1}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          if (Number.isFinite(next)) {
                            setViewCenterLat(clamp(next, -85, 85));
                          }
                        }}
                        className="w-24 xl:w-full"
                      />
                    </div>
                  </div>
                ) : null}
              </div>
              <div className="rounded-md border border-white/10 bg-slate-900/15 px-2 py-1.5 text-xs sm:text-sm">
                <p className="text-slate-400">Инспекция эталонного узла</p>
                <p className="text-slate-100">{inspectValue}</p>
              </div>
            </CardContent>
          </Card>
        </aside>
      </div>
    </main>
  );
}

function HeightGradientLegend({
  minHeight,
  maxHeight,
}: {
  minHeight: number;
  maxHeight: number;
}) {
  const seaSplit = minHeight < 0 && maxHeight > 0
    ? clamp((0 - minHeight) / Math.max(1, maxHeight - minHeight), 0, 1)
    : null;

  return (
    <div className="mt-2 space-y-2">
      <div className="relative h-7 overflow-hidden rounded-md border border-white/15">
        <div className="absolute inset-0" style={{ background: buildHeightGradient(minHeight, maxHeight) }} />
        {seaSplit !== null ? (
          <div
            className="absolute inset-y-0 w-px bg-white/75"
            style={{ left: `${(seaSplit * 100).toFixed(2)}%` }}
            title="Уровень моря"
          />
        ) : null}
      </div>
      <div className="flex items-center justify-between text-[11px] text-slate-300">
        <span>{formatNum(minHeight)} м</span>
        <span>0 м</span>
        <span>{formatNum(maxHeight)} м</span>
      </div>
      <p className="text-[11px] text-slate-400">
        Океан: светло-голубой → тёмно-синий, суша: зелёный → жёлтый → светлые пики.
      </p>
    </div>
  );
}

function NumericRange({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const commit = (next: number) => {
    if (!Number.isFinite(next)) {
      return;
    }
    const clamped = clamp(next, min, max);
    setDraft(clamped);
    onChange(clamped);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-slate-300">{label}</Label>
        <span className="text-xs text-cyan-200">{draft.toFixed(step < 1 ? 2 : step < 10 ? 1 : 0)}</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
        <Slider
          value={[draft]}
          min={min}
          max={max}
          step={step}
          onValueChange={(vals) => setDraft(vals[0] ?? draft)}
          onValueCommit={(vals) => commit(vals[0] ?? draft)}
        />
        <Input
          type="number"
          value={draft}
          min={min}
          max={max}
          step={step}
          onChange={(event) => commit(Number(event.target.value))}
          className="w-24"
        />
      </div>
    </div>
  );
}

function LabeledInput({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-slate-300">{label}</Label>
      <Input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}
