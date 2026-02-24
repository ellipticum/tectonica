# 11. Единая Модель Для Всех Scope

## 1. Проблема

Сейчас в системе два независимых пайплайна:

1. **Planet scope** (2048x1024): `compute_plates` -> `compute_relief` -> hydrology -> climate -> biomes.
2. **Island scope** (1024x512): `generate_tasmania_height_map` — отдельная функция с эллипсом + spine + noise warping.

Island scope не использует тектонику, деформацию, коровую эволюцию. Результат — гладкий "бычий глаз" без физической обоснованности.

## 2. Целевая архитектура

**Один пайплайн, два режима координат.**

```
scope = planet | island

if planet:
    grid = сфера (equirectangular, spherical_wrap)
    plate_generation = full Voronoi + evolution
    coordinates = (lat, lon) on sphere
else:
    grid = плоский прямоугольник (clamped edges)
    plate_generation = synthetic tectonic context
    coordinates = (x_km, y_km) in local frame
```

Все последующие стадии (strain propagation, crust evolution, relief, hydrology, climate, biomes) работают на **абстрактной сетке** с двумя операциями:

1. `neighbor(i, dx, dy)` -> index (spherical_wrap или clamp).
2. `cell_area(i)` -> km^2 (зависит от широты на сфере, постоянно на flat grid).

## 3. Синтетический тектонический контекст для island scope

### 3.1 Что нужно подать в пайплайн

Пайплайн relief ожидает на входе `ComputePlatesResult`:

- `plate_field: Vec<i16>` — ID плиты на каждую ячейку.
- `boundary_types: Vec<i8>` — тип границы (0/1/2/3).
- `boundary_normal_x/y: Vec<f32>` — нормаль к границе.
- `boundary_strength: Vec<f32>` — сила взаимодействия.
- `plate_vectors: Vec<PlateVector>` — omega/speed/heat/buoyancy на плиту.

### 3.2 Как синтезировать для острова

Вместо полной симуляции плит на сфере, генерируем **локальный тектонический контекст**:

```
fn generate_island_plate_context(
    seed: u32,
    island_type: IslandType,  // Continental, Arc, Hotspot, Rift
    width: usize,
    height: usize,
    params: &TectonicInputs,
) -> ComputePlatesResult
```

#### Continental Fragment (Tasmania-like)

1. Генерируем 2-3 плиты с **линейными/криволинейными границами**, проходящими через или рядом с островом.
2. Ориентация границ — NNW-тренд (или из seed).
3. Одна плита — "континентальная" (высокая buoyancy, низкий heat), другая — "океаническая" (низкая buoyancy, высокий heat).
4. Скорости задаются через Euler poles, как для planet scope.
5. Boundary types: mix of transform + divergent (рифтовый контекст).

```
Пример для Tasmania-like:
- Plate A (west): buoyancy=0.7, heat=40, omega направлен на NNW
- Plate B (east): buoyancy=0.3, heat=60, omega направлен на ESE
- Boundary: transform + divergent (NNW-SSE trending)
- Plate C (oceanic): buoyancy=-0.6, heat=80
```

#### Island Arc (Japan-like)

1. Генерируем 2 плиты: океаническая субдуцирующая + верхняя (континентальная/океаническая).
2. Граница — криволинейная дуга (arc), проходящая через центр grid.
3. Boundary type: convergent (subduction).
4. Высокая convergent velocity (5-10 cm/yr).
5. Добавляется back-arc divergent зона за дугой.

#### Hotspot Volcanic (Hawaii-like)

1. Одна плита, без границ в пределах grid.
2. Источник uplift — синтетический hotspot blob (Gaussian в центре grid).
3. `boundary_strength = 0` всюду.
4. Деформация задается через `mantle_heat` field с Gaussian maximum.

#### Rift (Corsica-like)

1. 2 плиты с divergent границей.
2. Граница проходит вдоль одной стороны острова.
3. Asymmetric: rift shoulder uplift на одной стороне, subsidence на другой.

### 3.3 Параметризация через UI

Вместо "tasmania" scope, пользователь выбирает:

```
scope: "planet" | "island"
island_type: "continental" | "arc" | "hotspot" | "rift"  (только для island)
island_scale_km: 50-1000  (масштаб острова)
```

Остальные параметры (plate_count, plate_speed, mantle_heat) переиспользуются.

## 4. Абстракция сетки

### 4.1 Trait GridTopology

Все функции пайплайна работают через абстракцию:

```rust
struct GridConfig {
    width: usize,
    height: usize,
    is_spherical: bool,
    km_per_cell_x: f32,     // для flat grid — постоянно
    km_per_cell_y: f32,     // для сферы — вычисляется по широте
}

impl GridConfig {
    fn neighbor(&self, x: i32, y: i32) -> usize {
        if self.is_spherical {
            index_spherical(x, y)
        } else {
            grid_index_clamped(x, y, self.width, self.height)
        }
    }

    fn cell_area_km2(&self, y: usize) -> f32 {
        if self.is_spherical {
            let lat = ... // from y
            self.km_per_cell_x * self.km_per_cell_y * lat.cos()
        } else {
            self.km_per_cell_x * self.km_per_cell_y
        }
    }
}
```

### 4.2 Функции, требующие адаптации

| Функция | Сейчас | Нужно |
|---------|--------|-------|
| `compute_plates` | Hardcoded WORLD_*, WorldCache | Отдельная `generate_island_plate_context` для island |
| `compute_relief` | Hardcoded WORLD_*, WorldCache, spherical_fbm | Параметризовать через GridConfig |
| `compute_slope` | Hardcoded WORLD_* | Уже есть `compute_slope_grid` |
| `compute_hydrology` | Hardcoded WORLD_* | Уже есть `compute_hydrology_grid` (но с багами ISLAND_*) |
| `compute_climate` | Hardcoded WORLD_*, WorldCache | Уже есть `compute_climate_grid` (но с багами ISLAND_*) |
| `compute_biomes` | Hardcoded WORLD_SIZE | Уже есть `compute_biomes_grid` |
| `compute_settlement` | Hardcoded WORLD_SIZE | Уже есть `compute_settlement_grid` |

### 4.3 Приоритет адаптации

1. **Высший**: `compute_relief` — ядро системы, 1000+ строк, завязан на WorldCache и spherical_fbm.
2. **Средний**: `compute_plates` — нужна альтернативная входная функция для island.
3. **Низкий**: downstream функции — `_grid` варианты уже есть, нужно только исправить hardcoded ISLAND_*.

## 5. Шумовые функции на flat grid

`spherical_fbm(sx, sy, sz, seed_phase)` работает в 3D координатах на сфере. Для flat grid нужен аналог:

```rust
fn planar_fbm(x_km: f32, y_km: f32, seed_phase: f32) -> f32 {
    // Те же 5 октав, та же структура, но в 2D координатах
    // с третьей координатой = seed_phase (для разных слоев)
    let sx = x_km / scale;
    let sy = y_km / scale;
    let sz = seed_phase;
    // Далее идентично spherical_fbm но без сферической обертки
    ...
}
```

Или, проще: использовать `value_noise3(x/scale, y/scale, seed_phase, seed)` напрямую, что уже не зависит от сферы.

## 6. Continentality для island scope

На planet scope `continentality_field` определяет, где суша, а где океан. Для island scope continentality задается аналитически:

```rust
fn island_continentality(x: f32, y: f32, island_type: IslandType, seed: u32) -> f32 {
    // Positive = land, negative = ocean
    // Базовая форма — deformed ellipse (как сейчас в generate_tasmania_height_map)
    // НО: форма зависит от тектонического типа и plate boundaries

    match island_type {
        Continental => {
            // Форма определяется horst-graben структурой плит
            // Суша — где plate buoyancy > threshold и нет рифтового graben
            continentality_from_plates(plate_field, buoyancy, ...)
        }
        Arc => {
            // Узкая дуга суши вдоль arc axis
            // Ширина зависит от convergence rate
            arc_continentality(arc_axis, convergence, ...)
        }
        Hotspot => {
            // Circular/elliptical shield shape
            hotspot_continentality(center, radius, ...)
        }
        Rift => {
            // Asymmetric: tilted block
            rift_continentality(fault_line, tilt, ...)
        }
    }
}
```

## 7. Масштабирование физических параметров

При переходе от planet (111 km/cell) к island (0.5-1 km/cell), физические параметры пересчитываются:

| Параметр | Planet (111 km/cell) | Island (0.5 km/cell) | Формула |
|----------|---------------------|---------------------|---------|
| strain propagation passes | 9-26 | 50-200 | ~proportional to grid cells / deformation width |
| crust evolution steps | 12-40 | 20-60 | simulation time / dt |
| erosion rounds | 1-3 | 5-20 | proportional to detail and resolution |
| fault iterations | 720-3400 | not used | replaced by stream power |
| buoyancy smooth passes | 1-18 | 3-30 | proportional to grid size |

## 8. Roadmap интеграции

### Phase 1: GridConfig abstraction

1. Создать `GridConfig` struct.
2. Рефакторнуть `compute_relief` принимать `GridConfig` вместо `WorldCache`.
3. Рефакторнуть noise-функции работать с (x_km, y_km) координатами.
4. Исправить hardcoded ISLAND_* в `_grid` функциях.

### Phase 2: Island plate context generator

1. Реализовать `generate_island_plate_context()` для типа `Continental`.
2. Протестировать: island scope с синтетическим контекстом -> compute_relief -> hydrology -> climate -> biomes.
3. Сравнить с planet scope визуально.

### Phase 3: Stream power erosion для island

1. Реализовать Braun-Willett implicit solver (O(n)).
2. Заменить текущий `carve_fluvial_valleys_grid` на stream power erosion.
3. Добавить литологическую вариацию K(x,y).

### Phase 4: Остальные island types

1. Arc, Hotspot, Rift контексты.
2. UI для выбора типа острова.
3. Пресеты масштабов (50km, 200km, 500km).

## 9. Что удаляется

1. `generate_tasmania_height_map()` — полностью.
2. `apply_events_island()` — заменяется общим `apply_events` с GridConfig.
3. `run_tasmania_scope()` — заменяется единым `run_island_scope()`.
4. Все упоминания "tasmania" в коде и UI.
5. Константы `ISLAND_WIDTH`, `ISLAND_HEIGHT`, `ISLAND_SIZE` — заменяются динамическими из GridConfig.
