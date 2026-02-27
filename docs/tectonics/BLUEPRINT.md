# Tectonica — Техническая документация

Процедурная генерация планет на основе геофизики.
Движок: `rust/planet_engine/src/lib.rs` (Rust → WASM).

## Файлы

| # | Файл | Тема |
|---|------|------|
| 01 | [01_ОБЗОР](01_ОБЗОР.md) | Архитектура, pipeline, входы/выходы |
| 02 | [02_СЕТКА](02_СЕТКА.md) | Сетка, координаты, обёртка полюсов |
| 03 | [03_ПЛИТЫ](03_ПЛИТЫ.md) | Генерация и эволюция тектонических плит |
| 04 | [04_ГРАНИЦЫ](04_ГРАНИЦЫ.md) | Классификация границ плит |
| 05 | [05_КОРА](05_КОРА.md) | Мощность коры, типы пород |
| 06 | [06_ИЗОСТАЗИЯ](06_ИЗОСТАЗИЯ.md) | Изостазия Эйри, флексуральное сглаживание |
| 07 | [07_ДЕФОРМАЦИЯ](07_ДЕФОРМАЦИЯ.md) | Пропагация деформации, interior suppression |
| 08 | [08_ЭРОЗИЯ](08_ЭРОЗИЯ.md) | Stream power, гипсометрия, detail noise |
| 09 | [09_ТЕМПЕРАТУРА](09_ТЕМПЕРАТУРА.md) | Температурная модель |
| 10 | [10_ОСАДКИ](10_ОСАДКИ.md) | Осадки, ветер, rain shadow |
| 11 | [11_БИОМЫ](11_БИОМЫ.md) | Классификация Уиттакера, альпийская зона |
| 12 | [12_РЕКИ](12_РЕКИ.md) | Гидрология, долины, озёра |
| 13 | [13_ОСТРОВ](13_ОСТРОВ.md) | Island scope: кроп, апскейл, эрозия |
| 14 | [14_СОБЫТИЯ](14_СОБЫТИЯ.md) | Метеориты, рифты, ocean shift |
| 15 | [15_ХАКИ](15_ХАКИ.md) | Известные хаки и эвристики |
| 16 | [16_ЛИТЕРАТУРА](16_ЛИТЕРАТУРА.md) | Список источников |

## Принцип

Каждый параметр указан с точным значением из кода и источником.
Где физика упрощена или подобрана вручную — указано явно (см. [15_ХАКИ](15_ХАКИ.md)).

# 01. Обзор системы

## Что генерируется

На вход: seed, физические параметры планеты, тектонические настройки.
На выход: детерминированная планета с рельефом, климатом, биомами, реками, settlement.

## Входные параметры

| Параметр | Тип | Пример |
|----------|-----|--------|
| `seed` | u32 | 42 |
| `radius_km` | f32 | 6371 |
| `gravity` | f32 | 9.81 |
| `atmosphere_bar` | f32 | 1.0 |
| `ocean_percent` | f32 | 67.0 |
| `plate_count` | i32 | 12 |
| `plate_speed_cm_per_year` | f32 | 5.0 |
| `mantle_heat` | f32 | 50.0 |

## Pipeline (planet scope)

1. Генерация поля плит (Вороной + Dijkstra)
2. Эволюция плит (полулагранжева адвекция)
3. Классификация границ (conv/div/trans)
4. Континентальная/океаническая разметка
5. Пропагация деформации (England & McKenzie 1982)
6. Мощность коры (Christensen & Mooney 1995)
7. Изостатический рельеф (Turcotte & Schubert 2002)
8. Stream power эрозия (Braun & Willett 2013)
9. Гипсометрическая коррекция (Cogley 1984)
10. Климат: температура + осадки
11. Гидрология: реки + озёра
12. Биомы (Whittaker 1975)
13. Settlement (Lieth 1975, Miami model)

## Presets

| Preset | erosion_rounds | fluvial_rounds | plate_evolution_steps |
|--------|---------------|----------------|----------------------|
| Ultra | 0 | 0 | 2 |
| Fast | 1 | 1 | 4 |
| Balanced | 2 | 2 | 7 |
| Detailed | 3 | 3 | 10 |

## Два scope

- **Planet**: 4096x2048, сферическая обёртка, ~9.77 км/ячейка
- **Island**: 1024x512, плоская сетка, кроп из planet scope

# 02. Сетка и координаты

## Planet scope

- Размер: `4096 x 2048` (WORLD_WIDTH x WORLD_HEIGHT)
- Проекция: равнопромежуточная цилиндрическая (equirectangular)
- km_per_cell_x = 2 * pi * 6371 / 4096 = **9.77 км** (на экваторе)
- km_per_cell_y = pi * 6371 / 2048 = **9.77 км**

## Полярная обёртка

X оборачивается циклически (левый край → правый край).
Y использует **полярную рефлексию**: при y < 0 → y = -y - 1, x += width/2.
Это корректно моделирует переход через полюс на сфере.

Функция: `spherical_wrap` (строка ~158).

## Island scope

- Размер: `1024 x 512` (ISLAND_WIDTH x ISLAND_HEIGHT)
- Плоская сетка, без обёртки (clamped edges)
- km_per_cell задаётся через `island_scale_km` / ISLAND_WIDTH
- cos_lat_by_y = 1.0 для всех строк (нет широтной коррекции)

## Широтная коррекция

На equirectangular сетке E-W ячейки физически уже на cos(lat).
Это учитывается в:

- Сглаживании: EW вес = 1/cos(lat), NS вес = 1.0, диагональ = 1/sqrt(cos^2+1)
- Пропагации деформации: decay_ew = exp(-cos(lat)/L_cells)
- Dijkstra-росте плит

cos(lat) ограничен снизу:
- Для итеративного сглаживания: min 0.30 (~72.5 deg) — предотвращает накопление анизотропии
- Для одноразовых операций: min 0.087 (~85 deg) — полная коррекция

## CellCache

Предвычисленные координаты для каждой ячейки:
- `noise_x/y/z` — 3D координаты на единичной сфере (для шума)
- `lat_deg`, `lon_deg` — географические координаты

Для island scope координаты вычисляются из центра кропа через lat/lon span.

# 03. Тектоника плит

## Генерация поля плит

Алгоритм: взвешенный рост Дейкстры из seed-точек (Вороной с шумом).

### Seed-точки

- Количество: `plate_count.clamp(2, 20)` [хак: жёсткое ограничение]
- Позиции: случайные (lat, lon), первые ~5 проб — фильтр минимального расстояния

### Рост (build_irregular_plate_field)

Мульти-ядерная стратегия: каждая плита имеет 2-7 ядер (nuclei),
связанных «историческими путями» — случайными блужданиями от seed.

Стоимость перехода:
```
cost = base_spread * drift_factor * structure_factor * polar_factor
```

Все коэффициенты — **эвристика** (подобраны визуально, не из физики):
- `drift_factor = 1.03 - 0.12 * drift_align` — выравнивание с направлением дрейфа
- `structure_factor` — из 2-октавного noise поля
- `polar_factor = 1 + (|lat|/90) * 0.1` — небольшой полярный бонус

Источник формы: Bird 2003 (статистика форм плит Земли) — но реализация процедурная.

## Эволюция плит (evolve_plate_field)

Полулагранжева адвекция: метки плит перемещаются вдоль вектора скорости.

- Шаг: `random_range(900_000, 3_400_000)` лет * age factor
- Количество шагов: `plate_evolution_steps` из preset (2-10)
- Relaxation: 3x3 окно, замена если `best_count >= 5 && own_count <= 2`
- Очистка фрагментов: мин. размер = `WORLD_SIZE / (plate_count * 640)`, clamp [140, 960]

## Euler pole кинематика

Скорость поверхности: v = omega x r * R (стандартная кинематика жёсткой плиты).

```
omega = (omega_x, omega_y, omega_z) — вектор угловой скорости
r = (sx, sy, sz) — единичный вектор положения на сфере
```

Разложение на east/north через проекцию на касательную плоскость.
**Наука**: стандартная тектоника плит (DeMets et al. 2010).

## Назначение свойств плит

- `heat`: `random_range(mantle_heat * 0.5, mantle_heat * 1.5)`, min 1.0
- `buoyancy`: случайное в [0, 1] — определяет континентальность
- `speed`: из plate_speed_cm_per_year с вариацией

# 04. Классификация границ плит

## Детекция границ

Ячейка считается граничной, если хотя бы один из 8 соседей принадлежит другой плите.

## Декомпозиция скорости

Для каждой граничной ячейки вычисляется относительная скорость двух плит:

```
v_rel = v_plate_A - v_plate_B
```

Проецируется на нормаль к границе и касательную:
- **Normal** (conv/div): сжатие < 0, растяжение > 0
- **Tangential** (transform): сдвиг

## Нормализация

```
boundary_scale = plate_speed * 1.25, min 1.2    [эвристика]
shear_weight = 0.82   — трансформные границы имеют меньший рельеф (Bird 2003)
```

## Пороги классификации

| Тип | Условие |
|-----|---------|
| Конвергентная (1) | conv > 0.14*scale AND conv >= 0.95*div AND conv >= 0.75*shear |
| Дивергентная (2) | div > 0.14*scale AND div >= 0.95*conv AND div >= 0.75*shear |
| Трансформная (3) | shear > 0.18*scale |
| Fallback | conv >= div → 1, иначе → 2 |

Пороги подобраны под земные пропорции: ~50% conv, ~30% div, ~20% transform.
**Эвристика** (не из физики, но калибровка по Bird 2003).

## boundary_strength

Интенсивность: magnitude нормированной relative velocity.
Диапазон [0, 1] — используется далее для мощности коры и uplift.

# 05. Коровая структура

## Континентальная разметка

Плиты сортируются по buoyancy (убывание). Самые плавучие → континентальные.
Целевая площадь: `(land_frac + 0.12).min(0.85)` — буфер 12% на шельф (Cogley 1984).

Сглаживание `continental_frac`: **20 проходов** → sigma ~80 км → 3sigma ~240 км.
Соответствует ширине пассивной окраины (Bond et al. 1995; Watts 2001).

Noise на границе: 3 октавы (freq 4/8/16, amp 0.10/0.04/0.01) — ломает Вороной-контуры.

## Мощность коры

**Источник**: Christensen & Mooney 1995 Table 3.

```
base = cf * (40.0 + base_var) + (1 - cf) * 7.0
```

- Континентальная база: **40 км** (глобальное среднее 39.7 км)
- Океаническая: **7 км** (White et al. 1992)
- Noise: 2 октавы (freq 3/6), amp **7.0 + 3.5 = ±10.5 км** (~1.5 sigma)

### Деформационные вклады

| Процесс | Формула | Источник |
|----------|---------|----------|
| Конвергент (конт.) | +conv_def * **30 км** | Tibet 70 км (Owens & Zandt 1997) |
| Конвергент (океан.) | +conv_def * **15 км** | Andes 55 км (Beck et al. 1996) |
| Дивергент | -div_def * **20 км** | Corti 2009 |
| Трансформ | +trans_def * **2 км** | Rockwell et al. 2002 |

Clamp: **[6, 72] км** (White et al. 1992 → Owens & Zandt 1997).

## Флексуральное сглаживание

**12 проходов** 8-neighbor diffusion (sigma ~50 км).
Приближение Te = 20 км (Watts 2001: Te от 5 до 100 км, 20 — среднее для молодой коры).

## Типы пород → K_eff

**Источник**: Harel et al. 2016, Bucher & Grapes 2011.

| Порода | K_eff (m^0.5/yr) | Условие |
|--------|-------------------|---------|
| Granite | 0.5e-6 | conv > 0.5 (конт.) |
| Quartzite | 0.8e-6 | conv > 0.2 (конт.) |
| Schist | 1.2e-6 | conv > 0.25 (океан.) или trans > 0.15 |
| Basalt | 1.0e-6 | океаническая кора |
| Sandstone | 2.0e-6 | div > 0.25 (конт.) или interior noise > -0.3 |
| Limestone | 3.0e-6 | interior noise > 0.3 |

K_eff сглаживается: 3 прохода, center weight 4.0 (Hovius & Stark 2006).

# 06. Изостазия Эйри

## Физическая основа

**Источник**: Turcotte & Schubert 2002, Geodynamics, §2.2 eq. 2.4.

Изостатический баланс: равное давление на глубине компенсации для всех колонок.

```
h = C * (rho_m - rho_c_eff) / rho_m
```

## Плотности

| Параметр | Значение | Источник |
|----------|----------|----------|
| rho_c (конт.) | 2800 кг/м3 | Стандартное |
| rho_c (океан.) | 2900 кг/м3 | Стандартное |
| rho_m (мантия) | 3300 кг/м3 | Стандартное |

Интерполяция: `rho_c = 2900 - continental_frac * 100`.

## Термальная коррекция

```
rho_c_eff = rho_c * (1 - heat_anomaly * 0.02)
```

- alpha = 3e-5 /K (Turcotte & Schubert 2002 §4.3)
- deltaT ~300 K → deltaRho/rho = 0.9%
- Используется **2%** как верхняя граница (включает partial melt)

## Примеры

| Тип | C (км) | rho_c | h (м) |
|-----|--------|-------|-------|
| Континент (40 км) | 40 | 2800 | 6061 |
| Океан (7 км) | 7 | 2900 | 848 |
| Разница | — | — | 5213 |

При ocean_percent = 67% это даёт freeboard ~300 м выше уровня моря.

## Уровень моря

Определяется как N-й перцентиль рельефа, где N = ocean_percent.
```
sorted_relief[floor(size * ocean_frac)] → sea_level
```

Все высоты затем смещаются: `relief[i] -= sea_level`.

## Известные ограничения

Модель **не учитывает**:
1. Water loading (Δh ≈ −2.5 км, Turcotte & Schubert §2.6)
2. Термальную субсидию океанической литосферы (Parsons & Sclater 1977)
3. Седиментную нагрузку
4. Динамическую топографию (±0.5 км, Hager et al. 1985)

Эти эффекты компенсируются гипсометрической коррекцией (см. [08_ЭРОЗИЯ](08_ЭРОЗИЯ.md)).

# 07. Деформация

## Пропагация деформации

**Источник**: England & McKenzie 1982 (EPSL) — распределённая деформация.

Граничные ячейки используются как seed (по типу: conv/div/trans).
Деформация распространяется итеративной max-дилатацией с экспоненциальным затуханием.

```
decay_ns = exp(-1 / L_cells)
decay_ew = exp(-cos(lat) / L_cells)
decay_diag = exp(-sqrt(cos^2(lat) + 1) / L_cells)
```

### Длины затухания

| Тип границы | L_d (км) | Источник |
|-------------|----------|----------|
| Конвергентная | **250** | E&M82: 200-500 км, geometric mean |
| Дивергентная | **200** | Illies & Greiner 1978: 150-300 км |
| Трансформная | **150** | Bourne et al. 1998: 100-200 км |

Количество проходов: `ceil(3 * L_cells) + 5` [+5 — safety margin].

### Сглаживание

**8 проходов** 8-neighbor diffusion (sigma ~90 км).
Убирает угловые клинья Вороного (120/60 deg) от max-дилатации.

## Interior suppression

**Источник**: Artemieva & Mooney 2001 — термальный возраст литосферы.

Удалённые от границ континентальные области — старые, холодные, жёсткие.
Weakness поле затухает экспоненциально от границ:

```
weakness(d) = exp(-d_km / L_rheol)
L_rheol = 300 км
```

BFS от всех граничных ячеек (Manhattan distance), затем:
```
conv_def[i] *= weakness[i]   (только для continental_frac > 0.1)
```

### Характерные значения

| Расстояние | weakness | Интерпретация |
|------------|----------|---------------|
| 300 км | 0.37 | Подвижный пояс |
| 600 км | 0.13 | Край щита |
| 900 км | 0.05 | Глубокий кратон |

# 08. Эрозия и формирование рельефа

## Stream power эрозия

**Источник**: Braun & Willett 2013, O(n) implicit solver.

```
E = K * A^m * S^n
h_new = (h_old + U*dt + factor * h_recv) / (1 + factor)
factor = K * dt * A^m / dx^n
```

### Параметры

| Параметр | Значение | Источник |
|----------|----------|----------|
| m (area exp) | 0.5 | Howard 1994 (каноническое) |
| n (slope exp) | 1.0 | Howard 1994 (каноническое) |
| kappa (hillslope diffusivity) | **0.01** м2/год | Fernandes & Dietrich 1997 (медиана 0.001-0.05) |
| dt | 500,000 лет | — |
| steps | 3 (1 эпоха) | Лёгкая эрозия |
| mfd_p | 1.5 | Freeman 1991 (рекомендовано 1.1, используем 1.5) |

### MFD дренажная площадь

**Источник**: Freeman 1991; goSPL (Salles et al., Science 2023).

```
w_j = max(0, slope_j)^p
area[j] += area[i] * w_j / sum(w)
```

Порядок обработки: от высоких к низким (topological sort).

## Uplift rates

**Источник**: GPS-данные (Bevis et al. 2005; Calais et al. 2003; Meade & Hager 2005).

```
speed_factor = plate_speed / 5.0
```

| Тип | Формула | Целевой диапазон |
|-----|---------|-----------------|
| Конвергент | bstr * **0.008** * speed_factor | 5-10 мм/год |
| Дивергент | -bstr * **0.002** * speed_factor | -1-3 мм/год |
| Трансформ | bstr * **0.0005** * speed_factor | <1 мм/год |

Uplift сглаживается: **5 проходов**.

## Noise injection

**±5 м** (freq 48) после эрозии — ломает радиальную когерентность каналов.
[Хак: стандартная техника в landscape evolution моделях для грубых сеток.]

## Изостатическая релаксация

**Источник**: Watts 2001 §8.4.

```
tau_eff = 5 Myr
f_relax = 1 - exp(-1.5/5.0) ≈ 0.26
h = h_eroded * 0.74 + h_isostatic * 0.26
```

## Гипсометрическая коррекция

**Источник**: Harrison et al. 1983; Cogley 1984.

Степенное сжатие распределения высот суши до median ~400 м.

```
alpha = ln(target/max) / ln(median/max), clamp [1.0, 8.0]
t = (h - sea_level) / max_land
h_new = sea_level + t^alpha * max_land
```

Trigger: median_fb > target * 1.2.
Delta smoothing: **10 проходов** (сглаживает переход у подножий гор).
[Хак: компенсирует missing physics — см. 06_ИЗОСТАЗИЯ.]

## Detail noise

**Источник**: Huang & Turcotte 1989 (beta=2.0); Montgomery & Brandon 2002.

4 октавы, amplitude halves per octave (спектральный наклон beta=2.0):

| Freq | Amplitude | Lambda |
|------|-----------|--------|
| 16 | 80 м | ~400 км |
| 32 | 40 м | ~200 км |
| 64 | 20 м | ~100 км |
| 128 | 10 м | ~50 км |

Elevation scaling: `sqrt(h / 5000)`, clamp [0, 1].

# 09. Температурная модель

## Формула

```
T = T_sea(lat) - lapse*h + ocean_mod + noise*2 + atm - aerosol*15 - continentality
```

## Базовая зональная температура

**Источник**: Peixoto & Oort 1992 Table 7.3; Hartmann 1994 eq. 2.1.

4th-order полином:
```
x = |lat| / 90
T_sea = 28 - 70*x^2 + 14*x^4
```

| Широта | T_sea (°C) |
|--------|-----------|
| 0° | 28.0 |
| 30° | 20.4 |
| 45° | 11.4 |
| 60° | -0.3 |
| 90° | -28.0 |

## Lapse rate

**Источник**: Holton & Hakim 2013.

```
lapse = 6.0 K/км = 0.006 K/м (environmental lapse rate)
h_m = max(elevation, 0)
```

## Морская модерация

**Источник**: Terjung & Louie 1972.

```
ocean_mod = 2.0 * sin^2(lat)   (для ocean cells)
           0.0                  (для land cells)
```

Даёт 0°C на экваторе, +2°C на полюсах (SST теплее зонального среднего суши).

## Greenhouse эффект

**Источник**: Pierrehumbert 2010 §4.3, gray atmosphere model.

```
atm = 33 * (p^0.3 - 1)
```

При p = 1.0 bar: atm = 0 (базовая T_sea уже включает парниковый эффект Земли).
При p = 2.0 bar: atm ≈ +7°C.
При p = 0.006 bar (Mars): atm ≈ -23°C.

## Континентальность

**Источник**: Conrad continentality index; Terjung & Louie 1972.

```
cont_cooling = coast_dist_km * 0.008 * sin(|lat|)
```

Пример: Москва (55°N, ~600 км от берега): -4°C от зонального среднего.

## Noise

**Источник**: CRU TS4 (Harris et al. 2014): sigma ~2°C на 10 км.

```
T_noise = value_noise3(...) * 2.0
```

## Аэрозольное охлаждение

**Источник**: Toon et al. 1997 (Chicxulub → ~15°C cooling).

```
T_aerosol = -aerosol * 15.0
```

## Clamp

```
T = clamp(T, -70, 55)   — наблюдаемые экстремумы Земли [хак: safety valve]
```

# 09. Температурная модель

## Формула

```
T = T_sea(lat) - lapse*h + ocean_mod + noise*2 + atm - aerosol*15 - continentality
```

## Базовая зональная температура

**Источник**: Peixoto & Oort 1992 Table 7.3; Hartmann 1994 eq. 2.1.

4th-order полином:
```
x = |lat| / 90
T_sea = 28 - 70*x^2 + 14*x^4
```

| Широта | T_sea (°C) |
|--------|-----------|
| 0° | 28.0 |
| 30° | 20.4 |
| 45° | 11.4 |
| 60° | -0.3 |
| 90° | -28.0 |

## Lapse rate

**Источник**: Holton & Hakim 2013.

```
lapse = 6.0 K/км = 0.006 K/м (environmental lapse rate)
h_m = max(elevation, 0)
```

## Морская модерация

**Источник**: Terjung & Louie 1972.

```
ocean_mod = 2.0 * sin^2(lat)   (для ocean cells)
           0.0                  (для land cells)
```

Даёт 0°C на экваторе, +2°C на полюсах (SST теплее зонального среднего суши).

## Greenhouse эффект

**Источник**: Pierrehumbert 2010 §4.3, gray atmosphere model.

```
atm = 33 * (p^0.3 - 1)
```

При p = 1.0 bar: atm = 0 (базовая T_sea уже включает парниковый эффект Земли).
При p = 2.0 bar: atm ≈ +7°C.
При p = 0.006 bar (Mars): atm ≈ -23°C.

## Континентальность

**Источник**: Conrad continentality index; Terjung & Louie 1972.

```
cont_cooling = coast_dist_km * 0.008 * sin(|lat|)
```

Пример: Москва (55°N, ~600 км от берега): -4°C от зонального среднего.

## Noise

**Источник**: CRU TS4 (Harris et al. 2014): sigma ~2°C на 10 км.

```
T_noise = value_noise3(...) * 2.0
```

## Аэрозольное охлаждение

**Источник**: Toon et al. 1997 (Chicxulub → ~15°C cooling).

```
T_aerosol = -aerosol * 15.0
```

## Clamp

```
T = clamp(T, -70, 55)   — наблюдаемые экстремумы Земли [хак: safety valve]
```

# 10. Модель осадков

## Зональные осадки

**Источник**: GPCP v2.3 (Adler et al. 2003).

Два гауссиана + floor:

```
ITCZ  = 2000 * exp(-(lat/8)^2)         — глубокая конвекция
midlat = 700 * exp(-((lat-45)/12)^2)    — среднеширотный storm track
hadley = ITCZ + midlat + 150            — floor: полярный минимум (Arthern+ 2006)
```

| Широта | P (мм/год) |
|--------|-----------|
| 0° | 2150 |
| 15° | 1078 |
| 30° | 339 |
| 45° | 850 |
| 60° | 449 |
| 80° | 160 |

## Ветровая модель

**Источник**: Peixoto & Oort 1992, три ячейки циркуляции.

| Зона | |lat| | Направление |
|------|-------|-------------|
| Пассаты | < 25° | С востока |
| Переход | 25-35° | Плавный blend |
| Вестерлиз | 35-55° | С запада |
| Переход | 55-65° | Плавный blend |
| Полярные восточные | > 65° | С востока |

Переходные зоны: 10° ширина (Seidel et al. 2008).
Меридиональная компонента: `0.35 * sin(lat)` (20-40% от зональной).

## Windward moisture

**Источник**: van der Ent & Savenije 2011, Water Resources Research.

Трассировка upwind от каждой ячейки суши. Экспоненциальное затухание влаги:

```
L_moisture = 700 км (e-folding distance)
windward = 500 * exp(-eff_dist * km_per_cell / 700)
```

Прибрежный excess: **500 мм/год** (Trenberth et al. 2003: 400-600).

## Coastal exposure

Доля океана в радиусе ~60 км. Enhancement: **600 мм** (Daly et al. 1994, PRISM: 400-800).

## Орографический подъём

**Источник**: Roe 2005; Smith & Barstad 2004.

```
orographic = max(h_here - h_upwind, 0) * 0.8 мм/м
```

0.8 мм/м — середина диапазона 0.5-1.5 (Roe 2005). [Эвристика: конкретное значение.]

## Rain shadow

**Источник**: Smith 1979; Galewsky 2009.

```
shadow = max(h_barrier - h_here, 0) * 0.40 мм/м * decay
decay = exp(-d_km / 250)     — e-folding 250 км (Galewsky 2009: 200-400)
```

## Clausius-Clapeyron altitude factor

**Источник**: Held & Soden 2006.

```
alt_factor = exp(-h_m * 0.000544), min 0.1
```

6 K/км lapse * 7%/K = 42%/км → lambda = 0.000544/м.

## Континентальное высыхание

```
cont_dry = exp(-coast_dist_km / 700)    — то же L что windward
```

## Atmospheric factor

```
atm_factor = clamp(atmosphere_bar, 0.01, 3.0)   — линейное масштабирование
```

## Финальная формула (суша)

```
P = (hadley*cont_dry + windward + coastal + orographic - shadow)
    * alt_factor + noise*60
P = P * atm_factor * (1 - aerosol*0.3)
P = clamp(P, 20, 4500)
```

Ocean: `hadley * 1.2 * atm_factor` (Trenberth et al. 2007: +20% evaporation).

# 11. Биомная классификация

## Метод

Decision tree по температуре и осадкам.
**Источник**: Whittaker 1975; Ricklefs & Relyea 2014.

## Ecotone jitter

**Источник**: Risser 1995 (ecotone 10-50 км).

Перед классификацией T и P возмущаются:
```
T_jitter = T + noise * 1.5°C
P_jitter = P + noise * 75 мм
```

Создаёт ~30 км переходные зоны.

## Decision tree

### Тундра (ID 1)
```
seasonal_amp = 20 * sin(lat)     — Terjung 1970
T_warmest = T + seasonal_amp / 2
if T_warmest < 10°C → Tundra     — Köppen ET criterion
```

### Тропическая зона (T > 22°C)
| P > 2000 мм | Tropical Rainforest (6) |
|-------------|------------------------|
| P > 500 мм | Tropical Savanna (7) |
| P < 500 мм | Desert (8) |

### Субтропическая зона (T > 15°C)
| P > 1500 мм | Subtropical Forest (9) |
|-------------|----------------------|
| P > 600 мм | Mediterranean (5) |
| P > 250 мм | Steppe (11) |
| P < 250 мм | Desert (8) |

### Умеренная зона (T > 5°C)
| P > 1000 мм | Temperate Forest (3) |
|-------------|---------------------|
| P > 400 мм | Temperate Grassland (4) |
| P > 200 мм | Steppe (11) |
| P < 200 мм | Desert (8) |

### Холодная зона (T < 5°C)
| P > 400 мм | Boreal Forest (2) |
|-------------|-------------------|
| P < 400 мм | Tundra (1) |

## Альпийская зона (ID 10)

**Источник**: Körner 2003, "Alpine Plant Life" Fig. 7.1.

```
treeline_base = max(4000 - 55 * |lat|, 200)
threshold = treeline_base + noise * 300
if h > threshold AND h > 500 → Alpine
```

55 м/градус — термальный порог: сезон роста < 6.4°C.
500 м guard — предотвращает Alpine на низких плоских территориях.

## Riparian vegetation

Реки > 0.12 интенсивности улучшают сухие биомы:
- T > 20°C → Tropical Savanna
- T > 12°C → Subtropical Forest
- else → Temperate Forest

## Mode filter

2 прохода: если < 2 из 4 cardinal neighbors совпадают — замена на most common.
Убирает изолированные пиксели.

# 12. Гидрология

## D8 flow routing

Каждая ячейка суши (h > 0) направляет сток к соседу с максимальным градиентом.
Gradient = (h_i - h_j) / dist, где dist = 1.0 (cardinal) или sqrt(2) (diagonal).

## Flow accumulation

Топологическая сортировка (topo sort, Kahn): от истоков к устьям.
Каждая ячейка начинает с `flow_accumulation = 1.0`.
При обработке: `flow_acc[receiver] += flow_acc[current]`.

## Озёра

Ячейка без downslope receiver и не касающаяся океана → озеро.
Порог: `h > 20 м` (исключает tidal flats и прибрежные wetlands).
Также: все ячейки с indegree > 0 после topo sort (циклы) → озёра.

## Channel initiation threshold

**Источник**: Montgomery & Dietrich 1988 (A_crit пропорционально dx^2).

```
threshold = max(size * (0.00018 + fluvial_rounds * 0.00007), 22)
```

## Визуализация рек

**Источник**: Hack 1957; Leopold & Maddock 1953.

```
acc_term = ((flow_acc - threshold) / (max_acc - threshold))^0.45
slope_term = slope / (slope + 35)
river = acc_term * (0.34 + 0.86 * slope_term)
```

Озёра: `river = max(river, 0.15)` — гарантия видимости.

## Флювиальные долины

**Источник**: Leopold & Maddock 1953 (D = 0.2 * Q^0.36).

```
Q = flow_acc * cell_area_m2 * runoff_m_per_s
runoff = 0.4 м/год = 0.4 / (365.25 * 86400) м/с     — Fekete et al. 2002
```

### Valley incision

```
valley_depth = 16.0 * Q^0.36     (= 0.2 * 80 * Q^0.36)
```

80x — valley/bankfull ratio (Schumm 1977: 20-200x, geometric mean ≈ 63, округлено до 80).
[Эвристика: конкретное число подобрано.]

### Ограничения

- Max cut: **30%** от высоты (Parker 1979, bank stability)
- Bedrock transition: выше **1200 м** → incision на 40% медленнее (Whipple 2004)
- Tidal zone: `h <= 2 м` → skip [хак]

### Valley smoothing

Center weight 2.2, diagonal 0.7, blend 20% per round.
[Хак: подобрано визуально.]

# 13. Island scope

## Принцип

Island scope — это кроп из planet scope с повышенным разрешением.
Физическая ширина: `island_scale_km` (по умолчанию расчёт из planet).
Сетка: 1024 x 512, плоская (без сферической обёртки).

## Выбор региона (find_interesting_region)

Скользящее окно по planet scope. Скоринг:

```
mix_score = 1 - |land_frac - 0.4| * 2.5    — оптимум: 40% суши
coast_norm = min(coast_fraction, 1.0)
elev_norm = min(elevation_range / 5000, 1.0)
boundary_norm = 1 + min(boundary_frac, 0.5) * 2.0
score = mix * coast * elev * boundary + noise*0.3
```

[Хак: полностью ad-hoc формула.]
Исключение: верхние/нижние 10% по y (полярные зоны).

## Upsampling

Билинейная интерполяция planet → island grid.
Затем: 6-октавный fBm для sub-grid деталей.

### FBM параметры

```
starting_freq = 8.0
persistence = 0.62     — Hurst H = 0.7, decay = 2^(-H) (Huang & Turcotte 1989)
6 октав
amplitude = 15% от локальной высоты    — Montgomery & Brandon 2002: 10-25%
```

## Edge fade

Smoothstep fade к -500 м на краях:
```
margin = 6% + noise*4%, min 3%
depth = -500 м     — верхний континентальный склон (Kennett 1982)
t^2 * (3 - 2t) interpolation
```

[Хак: визуальная обработка краёв.]

## Island-scale эрозия

Braun-Willett stream power, **10 шагов**.

### K_eff для island

Базовые K_eff из типа породы, умноженные на **1.5**:
[Хак: fudge factor, компенсация разницы разрешения.]

| Тип границы | Порода | K_eff (после x1.5) |
|-------------|--------|---------------------|
| Конвергент (1) | Quartzite | 1.2e-6 |
| Дивергент (2) | Basalt | 1.5e-6 |
| Interior | Sandstone | 3.0e-6 |

kappa = 0.01, dt = 500,000 лет, m = 0.5, n = 1.0.

## Климат и биомы

Те же модели что и planet scope, но:
- CellCache вычисляется из center_lat/center_lon кропа
- T/P сглаживаются **5 проходов** перед биомной классификацией [хак: smooth boundaries]

## Settlement (island)

Miami model NPP + бонусы:
- River bonus: `river_map * 0.25` (Diamond 1997)
- Coastal bonus: `coastal_exposure * 0.15`

# 14. События

## Метеориты

### Crater scaling

**Источник**: Schmidt & Housen 1987; Melosh 1989.

```
energy = 0.5 * mass * v^2
D_final [км] = 0.0133 * E^0.22    — Pi-group, gravity regime
crater_radius = max(D/2, 8 км)     — 8 км: sub-grid minimum [хак]
```

### Глубина

**Источник**: Pike 1977; Melosh 1989.

| Тип | Формула |
|-----|---------|
| Простой (D < 4 км) | d = D/5 (Pike 1977: D/d ≈ 5:1) |
| Сложный (D >= 4 км) | d = D/20 (Melosh 1989: D/d ≈ 20:1) |

Cap: **9000 м** (Chicxulub ~2-3 км deep). [Хак: safety valve.]

### Профиль

Параболическая чаша (Pike 1977):
```
falloff = 1 - (d_km / crater_radius)^2
relief -= crater_depth * max(falloff, 0)
```

### Аэрозольный индекс

**Источник**: Toon et al. 1997.

```
aerosol_index += min(1.0, log10(E) / 24)
```

Chicxulub (E ≈ 4e23 J, log ≈ 23.6): aerosol ≈ 0.98 → ~15°C cooling.

## Ocean shift

```
relief[center] += magnitude * 0.5
relief[*] += magnitude * 0.15    — глобальный сдвиг
```

[Хак: произвольные множители.]

## Rifts и другие

```
magnitude_eff = event.magnitude * 40.0 * sign     — sign = -1 для rift
falloff = exp(-dist / radius)
relief += magnitude_eff * falloff
```

[Хак: 40.0 — произвольный усилитель; экспоненциальный профиль не из физики.]

# 15. Известные хаки и эвристики

## Что считается хаком

Техника без физического обоснования, добавленная для:
- Подавления визуальных артефактов
- Safety-ограничений на крайние значения
- Компенсации отсутствующей физики

## Artifact suppression

| Хак | Зачем | Значение |
|-----|-------|----------|
| Noise ±5 м после эрозии | Ломает радиальные каналы stream power | freq 48, amp 5 м |
| Coastline cleanup (2 прохода) | Убирает 1-cell полуострова/бухты | >=3 ocean/land neighbors |
| Biome mode filter (2 прохода) | Убирает изолированные пиксели | <2 matching neighbors |
| Deformation smoothing (8 проходов) | Убирает угловые клинья Вороного | sigma ~90 км |
| Edge fade на island crop | Плавный переход к океану | smoothstep, -500 м |

## Safety clamps

| Поле | Диапазон | Обоснование |
|------|----------|-------------|
| Температура | [-70, 55]°C | Наблюдаемые экстремумы Земли |
| Осадки | [20, 4500] мм/год | Предотвращает отрицательные/экстремальные |
| Мощность коры | [6, 72] км | White 1992 min → Tibet max |
| Ocean fraction | [0.3, 0.95] | Предотвращает degenerate cases |
| Hypsometric alpha | [1.0, 8.0] | alpha > 8 создаёт бимодальные артефакты |
| Atmosphere factor | [0.01, 3.0] | Предотвращает runaway precipitation |
| plate_count | [2, 20] | Минимум 2 для границ; >20 — фрагментация |

## Fudge factors

| Параметр | Значение | Зачем |
|----------|----------|-------|
| Island K_eff * 1.5 | x1.5 | Компенсация разницы разрешения |
| boundary_scale * 1.25, floor 1.2 | масштаб границ | Нормализация скоростей |
| heat / 100 | нормализация | Предполагает вход в диапазоне 0-100 |
| Event magnitude * 40 | усилитель | Произвольный масштаб для rift/generic events |
| Ocean shift * 0.5 / 0.15 | множители | Произвольные |
| Lake h > 20 м | порог | Исключает tidal flats |
| Tidal zone h <= 2 м | порог | Skip fluvial incision |
| Alpine h > 500 м guard | порог | Предотвращает Alpine на равнинах |

## Гипсометрическая коррекция

**Самый большой концептуальный хак**: степенное сжатие распределения высот
компенсирует 4 отсутствующих физических процесса:
1. Water loading
2. Thermal subsidence
3. Sediment loading
4. Dynamic topography

## Процедурные шумы

Все seed хеши (0xBEEF, 0xDEAD, 0xCAFE, 0xF00D и т.д.) — произвольные.
LCG-генератор (Numerical Recipes) — минимальное качество, достаточное для задачи.
Матрица доменной ротации в spherical_fbm — **не ортогональная** (det ≈ 0.89).

## Подобранные числа проходов сглаживания

| Операция | Проходы | Физическая мотивация |
|----------|---------|---------------------|
| Continental margin | 20 | sigma ~80 км → passive margin width |
| Deformation fields | 8 | sigma ~90 км → убирает Voronoi wedges |
| Flexural isostasy | 12 | sigma ~50 км → Te ≈ 20 км compromise |
| Hypsometric delta | 10 | Плавный переход у подножий |
| Uplift | 5 | Предотвращает точечные аномалии |
| Land post-erosion | 5 | Сглаживает sub-grid erosion artifacts |
| K_eff | 3 | Fault-zone weathering blending |
| Coastal noise follow-up | 3 | Blend noise at guard boundaries |
| Island T/P pre-biome | 5 | Smooth boundaries at island resolution |

# 16. Литература

## Тектоника и деформация

- **England & McKenzie 1982** — Distributed deformation in continental collision zones. *EPSL*.
- **Bird 2003** — An updated digital model of plate boundaries. *G-cubed*.
- **DeMets et al. 2010** — Geologically current plate motions. *GJI*.
- **Artemieva & Mooney 2001** — Thermal thickness of lithosphere. *JGR*.
- **Bourne et al. 1998** — Transcurrent fault zones. *JGR*.
- **Illies & Greiner 1978** — Continental rift zones.

## Коровая структура

- **Christensen & Mooney 1995** — Seismic velocity structure and composition of continental crust. *JGR*.
- **White et al. 1992** — Oceanic crustal thickness.
- **Owens & Zandt 1997** — Tibetan crustal thickness. *JGR*.
- **Beck et al. 1996** — Andes crustal thickness.
- **Corti 2009** — Continental rift thinning. *Tectonophysics*.
- **Rockwell et al. 2002** — Transpressional uplift.
- **Rudnick & Gao 2003** — Composition of the continental crust. *Treatise on Geochemistry*.

## Изостазия

- **Turcotte & Schubert 2002** — *Geodynamics*, 2nd ed. Cambridge.
- **Watts 2001** — *Isostasy and Flexure of the Lithosphere*. Cambridge.
- **Parsons & Sclater 1977** — Ocean floor bathymetry and heat flow.
- **Hager et al. 1985** — Dynamic topography.
- **Bond et al. 1995** — Continental margin structure.

## Эрозия

- **Braun & Willett 2013** — A very efficient O(n) implicit method for solving stream power. *Earth Surf. Dynamics*.
- **Howard 1994** — A detachment-limited model of drainage basin evolution. *WRR*.
- **Freeman 1991** — Calculating catchment area with MFD. *Comp. & Geosci.*
- **Salles et al. 2023** — goSPL: global landscape evolution. *Science*.
- **Fernandes & Dietrich 1997** — Hillslope evolution by diffusive processes.
- **Harel et al. 2016** — Global analysis of erosivity. *JGR*.
- **Montgomery & Brandon 2002** — Topographic controls on erosion rates. *EPSL*.
- **Huang & Turcotte 1989** — Fractal mapping of digitized images.

## Гидрология

- **Leopold & Maddock 1953** — The hydraulic geometry of stream channels. *USGS PP 252*.
- **Hack 1957** — Studies of longitudinal stream profiles. *USGS PP 294-B*.
- **Schumm 1977** — *The Fluvial System*. Wiley.
- **Parker 1979** — Hydraulic geometry of active gravel rivers.
- **Whipple & Tucker 2002** — Implications of sediment-flux-dependent erosion.
- **Whipple 2004** — Bedrock rivers and landscape. *Ann. Rev. Earth Planet. Sci.*
- **Montgomery & Dietrich 1988** — Channel initiation thresholds.
- **Fekete et al. 2002** — Global composite runoff fields.

## Климат

- **Peixoto & Oort 1992** — *Physics of Climate*. AIP.
- **Hartmann 1994** — *Global Physical Climatology*. Academic.
- **Holton & Hakim 2013** — *Introduction to Dynamic Meteorology*, 5th ed.
- **Held & Soden 2006** — Robust responses of the hydrological cycle. *J. Climate*.
- **Pierrehumbert 2010** — *Principles of Planetary Climate*. Cambridge.
- **Terjung & Louie 1972** — Continentality and temperature.
- **Seidel et al. 2008** — Widening of the tropical belt. *Nature Geosci.*
- **van der Ent & Savenije 2011** — Moisture recycling. *WRR*.
- **Roe 2005** — Orographic precipitation. *Ann. Rev. Earth Planet. Sci.*
- **Smith 1979** — Influence of mountains on the atmosphere. *Adv. in Geophys.*
- **Smith & Barstad 2004** — A linear theory of orographic precipitation.
- **Galewsky 2009** — Rain shadow recovery. *J. Climate*.
- **Trenberth et al. 2003** — The changing character of precipitation. *BAMS*.
- **Adler et al. 2003** — GPCP v2.3. *J. Hydromet.*
- **Daly et al. 1994** — PRISM precipitation analysis.
- **Harris et al. 2014** — CRU TS4 climate dataset.

## Биомы и экология

- **Whittaker 1975** — *Communities and Ecosystems*, 2nd ed. Macmillan.
- **Ricklefs & Relyea 2014** — *Ecology: The Economy of Nature*, 7th ed.
- **Körner 2003** — *Alpine Plant Life*, 2nd ed. Springer.
- **Risser 1995** — The status of the science examining ecotones.
- **Terjung 1970** — The annual march of temperature.
- **Lieth 1975** — Modeling the primary productivity of the world.
- **Diamond 1997** — *Guns, Germs, and Steel*.

## Импактные события

- **Schmidt & Housen 1987** — Crater scaling laws.
- **Melosh 1989** — *Impact Cratering: A Geologic Process*.
- **Pike 1977** — Crater morphometry on Mars and Earth.
- **Toon et al. 1997** — Environmental perturbations caused by impacts.

## Прочее

- **Cogley 1984** — Continental margins and the extent of the continental crust.
- **Harrison et al. 1983** — Hypsometric analysis of Earth.
- **Taylor & McLennan 1995** — Continental crust volume.
- **Musgrave et al. 1989** — The synthesis and rendering of eroded fractal terrains. *SIGGRAPH*.
- **Borgefors 1986** — Distance transformations in digital images.
- **Bucher & Grapes 2011** — *Petrogenesis of Metamorphic Rocks*, 8th ed.
- **Hovius & Stark 2006** — Landslide-driven erosion.
- **Sayles & Thomas 1978** — Surface topography as a nonstationary random process.
