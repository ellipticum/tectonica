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
import { HeightGradientLegend } from "@/components/planet/height-gradient-legend";
import { PlanetLayerCanvas, type ProjectionMode } from "@/components/planet/layer-canvas";
import { LabeledInput, NumericRange } from "@/components/planet/form-controls";
import { buildLayerLegend, formatNum } from "@/lib/planet/layer-legend";
import {
  GENERATION_PRESETS,
  LAYER_OPTIONS,
  PREVIEW_PRESETS,
  SEED_SEARCH_PRESETS,
  type GenerationPresetId,
  type PreviewPresetId,
  type SeedSearchPresetId,
} from "@/lib/planet/ui-presets";
import {
  BIOME_NAMES,
  DEFAULT_PLANET,
  DEFAULT_SIMULATION,
  DEFAULT_TECTONICS,
  LAYER_GRAPH,
  type GenerationScope,
  type IslandType,
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

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

type ViewMode = "map" | "globe";
type FlatProjection = "equirectangular" | "mercator";

type SimulationWorkerRequest = {
  type: "generate";
  requestId: number;
  config: SimulationConfig;
  reason: RecomputeTrigger;
  attempts?: number;
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
  const [generationScope, setGenerationScope] = useState<GenerationScope>(
    (DEFAULT_SIMULATION.scope as GenerationScope) ?? "planet",
  );
  const [islandType, setIslandType] = useState<IslandType>("continental");
  const [islandScaleKm, setIslandScaleKm] = useState(400);
  const [seedSearchPreset, setSeedSearchPreset] = useState<SeedSearchPresetId>("4");
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
        scope: (DEFAULT_SIMULATION.scope as GenerationScope) ?? "planet",
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
      scope: generationScope,
      ...(generationScope === "island" && {
        islandType,
        islandScaleKm,
      }),
    };
    const searchAttempts =
      generationScope === "island"
        ? 1
        : (SEED_SEARCH_PRESETS.find((item) => item.id === seedSearchPreset)?.attempts ?? 1);

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
      attempts: searchAttempts,
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
  // Island scope is a flat grid — orthographic and mercator don't apply.
  const effectiveViewMode: ViewMode = generationScope === "island" ? "map" : viewMode;
  const projectionMode: ProjectionMode =
    generationScope === "island"
      ? "equirectangular"
      : effectiveViewMode === "globe"
        ? "orthographic"
        : flatProjection;
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
              <Label>Scope</Label>
              <Select
                value={generationScope}
                onValueChange={(value) => {
                  setGenerationScope(value as GenerationScope);
                  setReason("global");
                  setIsDirty(true);
                }}
                disabled={isGenerating}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select generation scope" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="planet">Planet</SelectItem>
                  <SelectItem value="island">Island</SelectItem>
                </SelectContent>
              </Select>
              {generationScope === "island" && (
                <>
                  <div className="h-2" />
                  <Label>Тип острова</Label>
                  <Select
                    value={islandType}
                    onValueChange={(value) => {
                      setIslandType(value as IslandType);
                      setIsDirty(true);
                    }}
                    disabled={isGenerating}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="continental">Континентальный фрагмент</SelectItem>
                      <SelectItem value="arc">Островная дуга</SelectItem>
                      <SelectItem value="hotspot">Горячая точка</SelectItem>
                      <SelectItem value="rift">Рифтовый</SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="h-2" />
                  <Label>Масштаб острова (км)</Label>
                  <Input
                    type="number"
                    min={50}
                    max={2000}
                    step={50}
                    value={islandScaleKm}
                    onChange={(e) => {
                      const v = Math.max(50, Math.min(2000, Number(e.target.value) || 400));
                      setIslandScaleKm(v);
                      setIsDirty(true);
                    }}
                    disabled={isGenerating}
                  />
                </>
              )}
              <div className="h-2" />
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
              <div className="h-2" />
              <Label>Seed Search</Label>
              <Select
                value={seedSearchPreset}
                onValueChange={(value) => setSeedSearchPreset(value as SeedSearchPresetId)}
                disabled={isGenerating || generationScope === "island"}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select seed search mode" />
                </SelectTrigger>
                <SelectContent>
                  {SEED_SEARCH_PRESETS.map((item) => (
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
              <Tabs value={effectiveViewMode} onValueChange={(value) => setViewMode(value as ViewMode)}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="map">Карта</TabsTrigger>
                  <TabsTrigger value="globe" disabled={generationScope === "island"}>Глобус</TabsTrigger>
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


