# Tectonica — Blueprint (март 2026)

Процедурная генерация планет на основе геофизики.
Движок: `rust/planet_engine/src/lib.rs` (Rust -> WASM).

---

## Текущее состояние

### Что работает хорошо

- **Planet scope** (4096x2048): тектоника, изостазия, климат, биомы — научно обоснованы
- **Береговая линия на planet scope**: BFS-perturbation ломает Voronoi-контуры,
  coastline irregular на масштабе 200-2000 km
- **Continent scope — западный берег**: спектральная пертурбация (Huang & Turcotte 1989)
  дала бухты, острова, полуострова — визуально близко к реальным картам
- **SPACE erosion model** (Shobe et al. 2017): заменил SPL в crop pipeline,
  отслеживает bedrock + sediment раздельно
- **MFD** (Freeman 1991): заменил D8 в crop scope, убрал параллельные полосы

### Что НЕ работает

| Проблема | Scope | Корневая причина |
|----------|-------|-----------------|
| Остров = blob | Island | `fade_land_edges=true` топит ВСЮ сушу по краям crop |
| Континент = "тумбочка" | Continent | Суша заполняет 3-4 стороны crop окна |
| Voronoi-артефакты | Continent | Plate tessellation bleeding через elevation field |
| Нет речных долин | Island+Continent | SPACE не прорезает видимые каналы за 100 шагов |
| Однородный interior | Island+Continent | Uplift field = 4 константы по boundary type |
| Биомы однородны (island) | Island | Один широтный пояс на 400 km crop |

---

## Диагностика

### 1. Остров = blob

**Симптом**: гладкая капля/овал вместо сложной формы.

**Причина**: pipeline `run_crop_pipeline` с `fade_land_edges=true` применяет
smoothstep fade ко ВСЕМ ячейкам у краёв, заставляя h → -500m. Любая
естественная форма с планеты уничтожается.

**Почему так сделано**: D8/MFD drainage требует океанического border — иначе
реки не имеют outlet и flow accumulation не работает.

**Научный анализ**: на реальной Земле острова имеют сложную форму из-за:
- Тектонической структуры (разломы, субдукционные дуги, рифты)
- Волновой эрозии побережья (Trenhaile 2000)
- Речных устьев → эстуарии → rias
- Изменений уровня моря (затопление долин → fjords, bays)

Ни один из этих процессов не моделируется. Crop + edge fade — это zoom
в гладкую планетарную карту с наложением пластилиновой маски.

### 2. Континент = "тумбочка"

**Симптом**: суша упирается в 3 из 4 границ crop окна, прямоугольная обрезка.

**Причина**: `find_continent_region` ищет окно с 40-90% суши, но при
target_km = 3000-4000 km и крупных континентах на планете, почти все
окна имеют суши до краёв. Edge penalty штрафует только worst single side.

**Научный анализ**: это не проблема физики, а проблема framing. Реальные
карты континентов (Южная Америка, Австралия) показывают continent
ВНУТРИ ocean frame. Нужно либо:
- Уменьшить target_km чтобы окно было больше континента
- Найти окно где continent не касается краёв (min 2 стороны ocean)
- Перейти к подходу "continent mask" вместо rectangular crop

### 3. Voronoi-артефакты в interior

**Симптом**: шестиугольные пятна на height map, видимые как ступенчатые
переходы elevation в interior континента.

**Причина**: plate_field (Voronoi tessellation) → crustal thickness →
isostatic elevation. Границы плит дают резкие jumps в crustal thickness.
Flexural smoothing (34 прохода) сглаживает мелкие, но крупные (>200 km)
структуры остаются.

**Научный анализ**: на реальной Земле внутриплитные elevation variations
определяются:
- Литосферным возрастом → термальной субсидией (Parsons & Sclater 1977)
- Мантийными плюмами → dynamic topography (Hager et al. 1985)
- Осадочными бассейнами (flexural loading)

В нашей модели нет dynamic topography и литосферного возраста → нет
внутриплитных elevation gradients → Voronoi pattern dominates.

### 4. Нет видимых речных долин

**Симптом**: relief гладкий, нет V-образных каньонов или широких долин.

**Причина двухслойная**:

a) **SPACE параметры**: K_br = 1-5e-6, dt = 100 kyr, 100 steps = 10 Myr.
   Steady-state relief для SPL: h_ss = (U/K)^(1/n) * (A^(-m/n)) * dx.
   Для interior (U=0.05 mm/yr, K=5e-6): h_ss ≈ 10 m/km * distance.
   За 10 Myr при 0.05 mm/yr uplift, суммарное поднятие = 500m.
   Erosion incision depth ~ K*A^0.5*S * dt = десятки метров.
   Valleys не видны потому что erosion << initial relief from planet crop.

b) **MFD distributes flow**: вместо sharp channel (D8) flow spread across
   multiple neighbors → diffuse erosion → smooth surface. MFD корректен
   для area accumulation, но erosion incision в реальности идёт по
   single channel (D8-like).

### 5. Uplift поле однородно

**Симптом**: купол вместо хребтов + долин.

**Причина**: uplift назначается по boundary_type (4 константы).
Большинство ячеек в crop = interior (type 0) → uniform 0.05 mm/yr.
Нет spatial variation → нет differential erosion → нет topographic contrast.

**Научный анализ**: реальный uplift определяется:
- Мантийным потоком (dynamic topography, ±1 km, Hager et al. 1985)
- Историей субдукции (slab pull, corner flow)
- Flexural response to erosional unloading (Watts 2001)
- Rift shoulder uplift (Weissel & Karner 1989)

---

## План улучшений

### Phase 1: Фундамент (crop pipeline)

#### 1.1 Убрать "тумбочку" — adaptive framing

**Проблема**: rectangular crop с фиксированным aspect ratio.

**Решение**: вместо фиксированного окна, алгоритм flood-fill от выбранного
seed point по connected land mass. Crop window = bounding box land mass +
ocean padding (20% margin). Aspect ratio вычисляется из формы mass, не
навязывается.

**Результат**: окно подстраивается под форму континента/острова.
Небольшие континенты (Австралия) получат квадратное окно.
Длинные (Чили) — вытянутое.

**Научное основание**: нет — это геометрия framing, не физика.

#### 1.2 Объединить island и continent в единый scope

**Проблема**: два отдельных pipeline с разными параметрами для по сути
одной операции (crop + refine).

**Решение**: единый `crop_scope` с параметрами:
- `target_km`: физический размер (200-5000 km)
- `land_fraction_target`: желаемая доля суши (0.3 = остров, 0.7 = континент)
- `edge_policy`: "ocean" (force ocean at edges) | "natural" (preserve)

Для "ocean" policy — найти регион где суша уже окружена океаном
на планете (не навязывать edge fade).

Для "natural" — использовать planet coastline как есть.

**Научное основание**: нет — архитектурный рефакторинг.

#### 1.3 Uplift из деформационного поля (не boundary type)

**Проблема**: 4 константы uplift по boundary_type.

**Решение**: crop из planet deformation field (conv_def, div_def, trans_def)
→ вычислить uplift rate для каждой ячейки:

```
U_conv = conv_def * 3.0e-3 * speed_factor     (0-3 mm/yr, Whipple 2009)
U_div  = -div_def * 1.0e-3 * speed_factor     (subsidence in rift center)
U_trans = trans_def * 0.5e-3 * speed_factor    (transpressional uplift)
U_total = U_conv + U_div + U_trans + U_background
U_background = 0.02e-3                         (GIA, Peltier 2004)
```

**Требует**: передать deformation fields в run_crop_pipeline (сейчас
передаётся только boundary_types).

**Научное основание**: England & McKenzie 1982; Whipple 2009; Weissel & Karner 1989.

### Phase 2: Эрозия и рельеф

#### 2.1 Hybrid flow routing: MFD area + D8 incision

**Проблема**: MFD-only produces diffuse erosion without visible valleys.

**Решение**: использовать MFD для drainage area accumulation (smooth,
no artifacts), но D8 receivers для implicit bedrock incision (sharp channels).

В space_erosion_step:
- area = compute_mfd_area (как сейчас)
- receivers = compute_d8_receivers (для bedrock implicit step)
- Sediment routing по D8 receivers (single path)

Это стандартный подход в goSPL (Salles et al. 2023) и FastScape.

**Научное основание**: Salles et al. 2023; Braun & Willett 2013.

**Статус**: УЖЕ реализовано — SPACE step использует D8 receivers + MFD area.
Проверить что channels действительно формируются (возможно K_br слишком мал).

#### 2.2 Calibrate SPACE parameters для visible valleys

**Проблема**: 100 шагов SPACE не прорезают видимые каналы.

**Анализ**: при K_br = 1-5e-6 и dt = 100 kyr:
- factor = K * dt * A^0.5 / dx
- Для ячейки с A = 100 km^2 = 1e8 m^2, A^0.5 = 10000:
  factor = 5e-6 * 1e5 * 10000 / 400 = 12.5
  → (h_old + factor * h_recv) / (1 + factor) ≈ h_recv (converges)
- Это значит каналы ДОЛЖНЫ формироваться. Проверить:
  a) не перезаписывает ли sediment deposition каналы обратно
  b) не слишком ли большой V_s (deposition rate)
  c) начальный relief (planet crop) может быть слишком rough для
     stream power to develop organized drainage

**Действие**: тестовый запуск с V_s = 0 (no deposition) чтобы изолировать
проблему. Если каналы появятся → V_s слишком высок. Если нет → проблема
в flow routing или initial conditions.

**Научное основание**: Shobe et al. 2017 Table 1 (параметры).

#### 2.3 Climate-erosion coupling (Roe et al. 2003)

**Проблема**: климат вычисляется один раз после эрозии. Обратная связь
precipitation → erosion → relief change → precipitation change отсутствует.

**Решение**: каждые N erosion steps, пересчитывать orographic precipitation
и модулировать K_eff:

```
K_eff = K_br * (P_local / P_mean)^alpha
alpha = 1.0 (Whipple 2009: 1-2)
```

**Результат**: windward slopes erode faster → asymmetric valleys,
leeward slopes accumulate sediment → rain shadow plains.

**Научное основание**: Roe et al. 2003; Whipple 2009.

### Phase 3: Внутренняя структура

#### 3.1 Dynamic topography (simplified)

**Проблема**: interior elevation определяется только crustal thickness →
Voronoi pattern dominates.

**Решение**: добавить long-wavelength (>500 km) noise field, коррелированный
с heat anomaly из plate properties:

```
dyn_topo = sum_octaves(freq=2,4,8; amp from heat) * 500 m
```

Физически: мантийные плюмы и downwellings создают ±0.5-1 km topography
с длиной волны 1000-5000 km (Hager et al. 1985; Flament et al. 2013).

Это не полная dynamic topography model (нужна мантийная конвекция),
но правильный СПЕКТР вариаций на правильных масштабах.

**Научное основание**: Hager et al. 1985; Flament et al. 2013.

#### 3.2 Разрешение и производительность

**Текущее**:
- Planet: 4096x2048 (8M cells)
- Island: 1024x512 (0.5M cells)
- Continent: 1920x1080 (2M cells)

**Предложение**: сделать разрешения настраиваемыми:
- Crop scope: default 1024x1024 (1M cells) для любого target_km
- Planet: оставить 4096x2048

1M cells × 100 SPACE steps = manageable (30-60 sec).
Качество определяется физикой модели, не пикселями.

---

## Приоритеты

1. **Phase 2.2** — calibrate SPACE (V_s, K_br) → видимые долины (быстрый тест)
2. **Phase 1.3** — uplift из deformation field → differential erosion
3. **Phase 1.1** — adaptive framing → убрать "тумбочку"
4. **Phase 1.2** — unified crop scope → убрать blob
5. **Phase 3.1** — dynamic topography → убрать Voronoi
6. **Phase 2.3** — climate coupling → asymmetric erosion

---

## Файлы документации

| # | Файл | Тема |
|---|------|------|
| 01 | [01_ОБЗОР](01_ОБЗОР.md) | Архитектура, pipeline, входы/выходы |
| 02 | [02_СЕТКА](02_СЕТКА.md) | Сетка, координаты, обёртка полюсов |
| 03 | [03_ПЛИТЫ](03_ПЛИТЫ.md) | Генерация и эволюция тектонических плит |
| 04 | [04_ГРАНИЦЫ](04_ГРАНИЦЫ.md) | Классификация границ плит |
| 05 | [05_КОРА](05_КОРА.md) | Мощность коры, типы пород |
| 06 | [06_ИЗОСТАЗИЯ](06_ИЗОСТАЗИЯ.md) | Изостазия Эйри, флексуральное сглаживание |
| 07 | [07_ДЕФОРМАЦИЯ](07_ДЕФОРМАЦИЯ.md) | Пропагация деформации, interior suppression |
| 08 | [08_ЭРОЗИЯ](08_ЭРОЗИЯ.md) | Stream power (planet), SPACE (crop) |
| 09 | [09_ТЕМПЕРАТУРА](09_ТЕМПЕРАТУРА.md) | Температурная модель |
| 10 | [10_ОСАДКИ](10_ОСАДКИ.md) | Осадки, ветер, rain shadow |
| 11 | [11_БИОМЫ](11_БИОМЫ.md) | Классификация Уиттакера, альпийская зона |
| 12 | [12_РЕКИ](12_РЕКИ.md) | Гидрология, долины, озёра |
| 13 | [13_КРОП](13_ОСТРОВ.md) | Crop scope: island + continent pipeline |
| 14 | [14_СОБЫТИЯ](14_СОБЫТИЯ.md) | Метеориты, рифты, ocean shift |
| 15 | [15_ХАКИ](15_ХАКИ.md) | Известные хаки и эвристики |
| 16 | [16_ЛИТЕРАТУРА](16_ЛИТЕРАТУРА.md) | Список источников |

## Принцип

Каждый параметр указан с точным значением из кода и источником.
Где физика упрощена или подобрана вручную — указано явно (см. [15_ХАКИ](15_ХАКИ.md)).
