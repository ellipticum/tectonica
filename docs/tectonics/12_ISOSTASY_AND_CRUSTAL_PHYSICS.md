# 12. Изостатика И Физика Коры

## 1. Зачем

Текущая модель не связывает толщину коры с высотой рельефа физически. Uplift задается эмпирическими формулами от strain/collision. Без изостатики невозможны:

1. Корректные плато (Тибет: 75-85 км коры = 5 км высоты).
2. Корректная батиметрия (тонкая океаническая кора = глубокий океан).
3. Гравитационная релаксация (горы не могут расти бесконечно).

## 2. Модель Airy (толщина коры -> высота)

### 2.1 Базовое уравнение

Принцип: литостатическое давление на глубине компенсации одинаково под всеми столбами.

```
Для суши:
    h = (C - C_ref) * (rho_m - rho_c) / rho_c

Для океана:
    d = (C_ref - C) * (rho_m - rho_c) / (rho_m - rho_w)
```

Где:
- `h` — высота над уровнем моря (м)
- `d` — глубина океана (м)
- `C` — толщина коры (км)
- `C_ref` — референсная толщина коры (км)
- `rho_c` — плотность коры (кг/м3)
- `rho_m` — плотность мантии (кг/м3)
- `rho_w` — плотность воды (кг/м3)

### 2.2 Стандартные плотности

| Параметр | Значение | Единица |
|----------|----------|---------|
| `rho_c` (континентальная кора) | 2800 | кг/м3 |
| `rho_oc` (океаническая кора) | 2900 | кг/м3 |
| `rho_m` (верхняя мантия) | 3300 | кг/м3 |
| `rho_w` (морская вода) | 1030 | кг/м3 |
| `C_ref` (референсная толщина) | 35 | км |
| `C_ocean_ref` (молодая океаническая) | 7 | км |

### 2.3 Валидация

| Регион | C (км) | h_predicted (м) | h_observed (м) |
|--------|--------|-----------------|----------------|
| Тибет | 75-85 | 6070-7580 | 4500-5500 |
| Анды | 60-70 | 3790-5310 | 3000-4500 |
| Европейские равнины | 30-35 | -760-0 | 0-200 |
| Океаническая кора (7 км) | 7 | -4245 (глубина) | -4000-5000 |

Расхождение Тибет/Анды объясняется:
- Динамической топографией (мантийный flow).
- Эрозией, которая снижает высоту быстрее, чем утолщение компенсирует.
- Тепловой аномалией (горячая кора менее плотная → дополнительный подъем).

### 2.4 Формула для симулятора

```rust
fn isostatic_elevation(
    crust_thickness_km: f32,
    is_continental: bool,
    heat_anomaly: f32,  // 0.0 = cold, 1.0 = very hot
) -> f32 {
    let rho_c = if is_continental { 2800.0 } else { 2900.0 };
    let rho_m = 3300.0;
    let rho_w = 1030.0;
    let c_ref = if is_continental { 35.0 } else { 7.0 };

    // Thermal density correction: hot crust is less dense
    let thermal_correction = 1.0 - heat_anomaly * 0.02;  // up to 2% density reduction
    let rho_c_eff = rho_c * thermal_correction;

    let delta_c = crust_thickness_km - c_ref;

    if delta_c >= 0.0 {
        // Continental: positive elevation
        delta_c * 1000.0 * (rho_m - rho_c_eff) / rho_c_eff
    } else {
        // Oceanic: negative elevation (depth)
        delta_c * 1000.0 * (rho_m - rho_c_eff) / (rho_m - rho_w)
    }
}
```

## 3. Флексуральная изостатика

### 3.1 Зачем

Airy isostasy предполагает локальную компенсацию (каждый столб независим). В реальности литосфера имеет **жесткость** и распределяет нагрузку по площади. Это важно для:

1. Передовых бассейнов (foreland basins) — прогиб перед горами.
2. Формы островов — небольшой остров может быть "поддержан" жесткостью плиты, не проваливаясь по Airy.

### 3.2 Уравнение

```
D * nabla^4(w) + (rho_m - rho_fill) * g * w = q(x,y)
```

Где:
- `D` — флексуральная жесткость
- `w` — прогиб литосферы
- `q` — нагрузка (горный массив, ледник, осадки)
- `rho_fill` — плотность заполнителя (воздух=0, вода=1030, осадки=2200)

### 3.3 Флексуральная жесткость

```
D = E * Te^3 / (12 * (1 - nu^2))
```

| Параметр | Значение |
|----------|----------|
| `E` (модуль Юнга) | 100 GPa (стандарт, диапазон 70-200) |
| `nu` (коэффициент Пуассона) | 0.25 |
| `Te` (эффективная упругая толщина) | переменная, см. ниже |

### 3.4 Эффективная упругая толщина Te

| Тектоническая обстановка | Te (км) | Характерная длина волны (км) |
|--------------------------|---------|------------------------------|
| Срединно-океанический хребет | 2-5 | 20-50 |
| Молодая океаническая литосфера (<25 Myr) | 5-15 | 50-150 |
| Старая океаническая литосфера (>80 Myr) | 25-50 | 200-500 |
| Активный ороген | 5-25 | 50-200 |
| Стабильный кратон | 60-120 | 500-1000+ |
| Глобальное среднее | 34 ± 4 | ~300 |

### 3.5 Применение в симуляторе

Полное решение бигармонического уравнения дорого. Приближение:

```rust
// Simplified flexural isostasy via Gaussian smoothing
// The flexural wavelength lambda ≈ 2*pi * (D / (rho_m*g))^0.25
// For Te=30km: lambda ≈ 250 km, or about 2-3 cells on planet grid

fn apply_flexural_isostasy(
    elevation: &mut [f32],
    crust_thickness: &[f32],
    te_field: &[f32],      // Te per cell (km)
    grid: &GridConfig,
) {
    let airy_elevation = compute_airy_elevation(crust_thickness);

    // Smooth airy_elevation with kernel radius proportional to Te
    for i in 0..grid.size() {
        let te = te_field[i];
        let lambda_cells = flexural_wavelength_cells(te, grid.km_per_cell);
        let smoothed = gaussian_smooth_at(airy_elevation, i, lambda_cells, grid);
        elevation[i] = smoothed;
    }
}
```

## 4. Модель возраста океанической коры

### 4.1 GDH1 (Stein & Stein, 1992)

Стандартная модель зависимости глубины океана от возраста коры:

```
Для t <= 20 Myr:
    d(t) = 2600 + 365 * sqrt(t)   [метры]

Для t > 20 Myr:
    d(t) = 5651 - 2473 * exp(-0.0278 * t)   [метры]
```

| Параметр GDH1 | Значение |
|----------------|----------|
| Глубина хребта (t=0) | 2600 м |
| Асимптотическая глубина | ~5651 м |
| Толщина плиты | 95 км |
| Температура подошвы | 1450°C |

### 4.2 Эволюция ocean_age

```
dA/dt + v · grad(A) = 1
```

Где:
- `A` — возраст в Myr.
- `v` — скорость плиты.
- Сброс `A = 0` на дивергентных границах (spreading ridges).
- `A` растет на 1 Myr за каждый шаг dt=1 Myr.

### 4.3 Реализация в симуляторе

```rust
fn compute_ocean_depth_from_age(age_myr: f32) -> f32 {
    if age_myr <= 0.0 {
        return -2600.0;  // Ridge crest depth
    }
    if age_myr <= 20.0 {
        -(2600.0 + 365.0 * age_myr.sqrt())
    } else {
        -(5651.0 - 2473.0 * (-0.0278 * age_myr).exp())
    }
}
```

### 4.4 Инициализация ocean_age для island scope

На flat grid нет истории спрединга. Приближение:

```
ocean_age(x,y) = distance_to_nearest_divergent_boundary(x,y) / half_spreading_rate
```

Где `half_spreading_rate = |v_rel_normal| / 2` на дивергентной границе.

## 5. Движущие силы плит

### 5.1 Иерархия сил

| Сила | Величина (N/m) | Направление |
|------|----------------|-------------|
| Slab pull (полная) | 2-3 × 10^13 | Тянет плиту к зоне субдукции |
| Slab pull (нетто, переданная на поверхность) | 4-6 × 10^12 | ~90% рассеивается в мантии |
| Ridge push | 2-3 × 10^12 | Отталкивает от хребта |
| Basal drag | переменная | Может помогать или сопротивляться |

### 5.2 Следствия для скоростей плит

Корреляция между slab pull и скоростью плит — **ключевое наблюдение**:

| Плита | Прикреплённый slab? | Скорость (см/год) |
|-------|---------------------|-------------------|
| Pacific | Да (обширный) | ~7-10 |
| Nazca | Да | ~7 |
| Australia | Да (частичный) | ~6 |
| Cocos | Да | ~8-10 |
| Eurasia | Нет | ~2 |
| North America | Нет | ~2 |
| South America | Нет | ~1 |
| Africa/Nubia | Нет | ~2-3 |

### 5.3 Реализация в симуляторе

Вместо полного геодинамического решения, используем **empirical speed correlation**:

```rust
fn assign_plate_speed(
    has_subducting_slab: bool,
    slab_fraction: f32,      // 0..1, какая доля периметра субдуцирует
    base_speed: f32,          // из UI (plate_speed_cm_per_year)
) -> f32 {
    if has_subducting_slab {
        base_speed * (1.5 + 2.0 * slab_fraction)  // 1.5x - 3.5x faster
    } else {
        base_speed * 0.4  // slow plates without slabs
    }
}
```

### 5.4 NNR constraint (No-Net-Rotation)

Сумма угловых моментов всех плит должна быть нулевой:

```
Sum_i (integral over plate_i of (Omega_i x r) dA) = 0
```

Приближение: после генерации всех Omega_i, вычислить средний вектор и вычесть его:

```rust
fn enforce_nnr(plate_vectors: &mut [PlateVector], plate_field: &[i16]) {
    // Compute area-weighted mean angular velocity
    let mut mean_omega = (0.0, 0.0, 0.0);
    let mut total_area = 0.0;
    for (pid, pv) in plate_vectors.iter().enumerate() {
        let area = plate_field.iter().filter(|&&p| p == pid as i16).count() as f32;
        mean_omega.0 += pv.omega_x * area;
        mean_omega.1 += pv.omega_y * area;
        mean_omega.2 += pv.omega_z * area;
        total_area += area;
    }
    mean_omega.0 /= total_area;
    mean_omega.1 /= total_area;
    mean_omega.2 /= total_area;

    // Subtract mean to enforce NNR
    for pv in plate_vectors.iter_mut() {
        pv.omega_x -= mean_omega.0;
        pv.omega_y -= mean_omega.1;
        pv.omega_z -= mean_omega.2;
    }
}
```

## 6. Реология и ширина деформации

### 6.1 Текущая проблема

Ширина деформационных коридоров контролируется `buoyancy_smooth_passes` — числовым параметром без физического смысла. Реальная ширина зависит от:

1. `Te` (упругая толщина) — жесткая кора деформируется узко, мягкая — широко.
2. `heat_flux` — горячая кора слабее.
3. Наследованная слабость (old rifts, suture zones) — предопределяет где сконцентрируется деформация.
4. Тип столкновения — C-C деформируется шире (1000+ км) чем O-C (100-300 км).

### 6.2 Деформационная ширина

```
W_def = W0 * (1 + b1*(1 - strength) + b2*heat_norm + b3*weakness - b4*Te_norm)
```

Где:
- `W0` = 3 ячейки (базовая ширина, ~330 км на planet grid).
- `strength` = нормализованная прочность коры (0-1).
- `heat_norm` = нормализованный тепловой поток (0-1).
- `weakness` = наследованная слабость (0-1).
- `Te_norm` = Te / 100 (нормализованная упругая толщина).

### 6.3 Реальные примеры

| Система | Ширина (км) | Причина |
|---------|-------------|---------|
| Гималаи-Тибет | 1000-1200 | Слабая азиатская кора, горячая мантия, C-C коллизия |
| Анды | 300-900 | Крутой slab на севере (узко), flat slab на юге (широко) |
| Basin and Range | ~800 | Распределённая экстензия, горячая литосфера |
| Субдукция O-C | 100-300 | Жесткая океаническая литосфера |
| Transform | 10-50 | Узкая зона сдвига |

## 7. Геометрия субдукции

### 7.1 Угол погружения slab

```
slab_dip = f(slab_age, convergence_rate, overriding_plate_type)
```

Эмпирические диапазоны (из Slab2):

| Параметр | Крутой slab (Mariana) | Пологий slab (Peru) |
|----------|-----------------------|---------------------|
| Dip angle | 70-80° | 25-30° → flat |
| Trench depth | 10-11 км | 7-8 км |
| Arc-trench distance | 100-200 км | 250-400 км |
| Back-arc | Да (active spreading) | Нет |
| Volcanism | Active arc | Suppressed above flat segment |

### 7.2 Формула для симулятора

```rust
fn slab_dip_deg(
    slab_age_myr: f32,      // возраст субдуцирующей коры
    convergence_cm_yr: f32,  // скорость сближения
    is_overriding_continental: bool,
) -> f32 {
    // Base dip: older slabs -> steeper (more negative buoyancy)
    let age_factor = (slab_age_myr / 150.0).clamp(0.0, 1.0);
    let base_dip = 30.0 + 40.0 * age_factor;

    // Continental overriding plate -> shallower dip
    let continent_correction = if is_overriding_continental { -10.0 } else { 0.0 };

    // Fast convergence -> slightly steeper
    let speed_correction = (convergence_cm_yr - 5.0) * 1.5;

    (base_dip + continent_correction + speed_correction).clamp(20.0, 85.0)
}
```

### 7.3 Расстояние дуга-желоб

```
arc_trench_distance_km = slab_depth_to_volcanism / tan(slab_dip)
```

Где `slab_depth_to_volcanism ≈ 100-120 км` (начало плавления над slab).

### 7.4 Spacing вулканов в дуге

```
volcano_spacing_km = 30 + 40 * (mantle_wedge_viscosity / 1e20)
```

Типичный диапазон: 30-70 км.

## 8. Связь с текущим кодом

### 8.1 Что уже реализовано

1. `crust_thickness` поле — есть, эволюционирует во времени (thickening/thinning/relaxation).
2. `heat_norm` — есть, используется в relief.
3. `collision modes` (CC/OC/OO) — есть.
4. `strain propagation` с conductivity — есть.

### 8.2 Что нужно добавить

1. **Isostatic elevation**: `h = f(crust_thickness)` вместо эмпирических формул. Заменить блок base relief (lines 4141-4306) на isostatic calculation.
2. **ocean_age** поле: advection + age-depth law. Заменить `apply_ocean_profile` на GDH1-based depth.
3. **Te field**: вычислять из crust_type + heat + age. Использовать для flexural smoothing вместо буйansancy_smooth_passes.
4. **NNR constraint**: добавить в `compute_plates` после генерации omega.
5. **Speed-slab correlation**: назначать скорости плит на основе наличия субдуцирующего slab.
6. **Slab dip**: вычислять из возраста коры и скорости, использовать для trench depth и arc-trench distance.
