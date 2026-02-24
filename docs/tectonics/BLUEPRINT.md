# Tectonica: Physics-Based Procedural Planet Generation

**Technical Blueprint — From Physical Foundations to Numerical Implementation**

> This document describes the complete generation pipeline as implemented in
> `rust/planet_engine/src/lib.rs` (compiled to WASM). Every formula, parameter,
> and approximation is stated explicitly with its source. Where physics is
> simplified or a heuristic is used, it is flagged.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Plate Tectonics](#2-plate-tectonics)
3. [Crustal Structure and Isostasy](#3-crustal-structure-and-isostasy)
4. [Surface Evolution: Erosion](#4-surface-evolution-erosion)
5. [Climate](#5-climate)
6. [Hydrology](#6-hydrology)
7. [Biome Classification](#7-biome-classification)
8. [Settlement Suitability](#8-settlement-suitability)
9. [Island-as-Crop](#9-island-as-crop)
10. [Complete Parameter Table](#10-complete-parameter-table)
11. [Pipeline Execution Order](#11-pipeline-execution-order)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)
13. [References](#13-references)

---

## 1. System Overview

### 1.1 What This System Does

Tectonica generates procedural Earth-like planets. Given a seed number and
physical parameters (radius, gravity, atmosphere, ocean fraction), it produces:

- A tectonic plate field with classified boundaries
- A physically-motivated relief (elevation) map
- Temperature and precipitation fields
- River networks and lakes
- Biome classification
- Settlement suitability

All output is deterministic: the same seed and inputs always produce the same
planet.

### 1.2 Design Philosophy

**One pipeline, two scales.** There is a single generation pipeline. The
"planet" scope runs it on a 2048×1024 spherical grid (~20 km/cell). The
"island" scope first generates the full planet, then crops an interesting region
and refines it on a 1024×512 flat grid (~0.4 km/cell).

**Physics-first, not art-directed.** Relief emerges from isostasy and stream
power erosion, not from hand-tuned noise blending. Climate follows from
latitude, altitude, and atmospheric circulation, not from painting. Biomes
follow from temperature and precipitation via Whittaker classification.

### 1.3 Grid Abstraction

All functions operate on a generic `GridConfig`:

| Field | Planet | Island |
|-------|--------|--------|
| `width` | 2048 | 1024 |
| `height` | 1024 | 512 |
| `is_spherical` | true | false |
| `km_per_cell_x` | ~19.6 | ~0.39 |
| `km_per_cell_y` | ~19.6 | ~0.39 |

**Planet cell size derivation:**
```
km_per_cell_y = π × R_earth / height = π × 6371 / 1024 ≈ 19.55 km
km_per_cell_x = 2π × R_earth / width  = 2π × 6371 / 2048 ≈ 19.55 km (at equator)
```

**Neighbor lookup.** On the spherical grid, x wraps (left edge → right edge);
y clamps (no polar wraparound). On the flat grid, both axes clamp. 8
neighbors (N, NE, E, SE, S, SW, W, NW) are supported.

### 1.4 Coordinate Systems

Each cell `i` has pre-computed:
- **Latitude** `lat_deg[i]` in [−90, +90]
- **Longitude** `lon_deg[i]` in [−180, +180]
- **Unit sphere** `(sx, sy, sz)` for 3D noise sampling
- For island crops: lat/lon are real planetary coordinates of the source region

---

## 2. Plate Tectonics

### 2.1 Physical Background

Earth's lithosphere is divided into ~15 major tectonic plates that move relative
to each other. Plate motion is described by **Euler's theorem**: any rigid-body
rotation on a sphere can be expressed as rotation about a fixed pole (the Euler
pole) with angular velocity ω.

> **Cox & Hart (1986)** *Plate Tectonics: How It Works*

The velocity of a point **r** on a plate with rotation vector **Ω** is:

```
v(r) = Ω × r                                                           (2.1)
```

where **Ω** = (ω_x, ω_y, ω_z) is in radians/year and **r** is on the unit
sphere scaled by radius R (in cm for cm/yr velocities).

### 2.2 Plate Field Generation

**Input:** `plate_count` ∈ [2, 20], `plate_speed_cm_per_year`, `mantle_heat`.

**Step 1: Plate specification.** For each plate k:
- Random position (lat_k, lon_k) on the sphere
- Random pole direction (uniform on sphere via acos(2u−1) for latitude)
- Speed: `speed_k = plate_speed × uniform(0.5, 1.5)` cm/yr
- Angular velocity magnitude: `|Ω_k| = speed_k / R_cm × uniform(0.82, 1.24)`
- Buoyancy: `uniform(−1, +1)` (>0 → continental, <0 → oceanic)
- Heat: `0.5 + mantle_heat × uniform(0, 1) × 2.5`

**Step 2: Irregular plate field growth** (`build_irregular_plate_field`).

This is a priority-queue (Dijkstra-like) flood fill from plate nuclei. Each
plate has:
- 1 primary nucleus at its center
- 2–7 secondary nuclei at random offsets
- 6–18 historical path nuclei along a curved trajectory

The growth cost from cell `a` to neighbor `b` is:

```
cost(a→b) = w × spread × f_rough × f_drift × f_structure × f_polar      (2.2)

where:
  w           = 1.0 (orthogonal) or 1.414 (diagonal)
  spread      = random(0.82, 1.22) / size_bias^0.36
  f_rough     = 1 + roughness × (0.22 sin(ξ_a) + 0.18 cos(ξ_b))
  f_drift     = 1.03 − 0.12 × dot(direction_to_b, plate_drift)
  f_structure = clamp(0.62 + 0.98 × noise + 0.36 − roughness × 0.17, 0.45, 1.9)
  f_polar     = 1 + |lat|/90 × 0.1
```

**⚠ Approximation:** This is a geometric heuristic, not geophysics. Real plate
shapes emerge from rheology, driving forces, and ~200 Myr of history. The
priority-queue growth produces irregular, non-Voronoi shapes that *look*
plausible, but the cost function parameters (0.22, 0.18, 0.12, 0.62, etc.) are
tuned by eye, not derived from physics.

**Step 3: Fragment cleanup.** Plate regions smaller than
`clamp(WORLD_SIZE / (plate_count × 420), 240, 1800)` cells are merged into
their largest neighbor.

### 2.3 Plate Evolution

The plate field evolves over geological time via `evolve_plate_field`. Each
step:

1. Time span: `random(0.9, 3.4) Myr × (0.92 + 0.32 × age_norm)`
2. For each cell: compute plate velocity → displacement in cells
3. Bilinear vote interpolation (which plate "arrives" at each cell)
4. Relaxation: 3×3 majority filter
5. Fragment cleanup

Number of steps = `detail.plate_evolution_steps` (2–10 depending on preset).

**Velocity at a cell** (`plate_velocity_xy_from_omega`):

Given unit-sphere position (sx, sy, sz) and plate rotation vector Ω:

```
v_3D = Ω × r̂ × R_cm                                                    (2.3)

Transform to east-north:
  cos_lat = sqrt(sx² + sy²)
  v_east  = v_3D · ê_east   where ê_east = (−sy, sx, 0) / cos_lat
  v_north = v_3D · ê_north  where ê_north = (−sz·sx, −sz·sy, cos_lat) / cos_lat
```

### 2.4 Boundary Classification

At each cell, the relative velocity between adjacent plates is decomposed into
normal (convergence/divergence) and tangential (shear) components.

**Classification rules** (implemented at lines 1638–1649):

| Type | Condition |
|------|-----------|
| Convergent (1) | conv > 0.14 × scale AND conv ≥ 0.95 × div AND conv ≥ 0.75 × shear |
| Divergent (2) | div > 0.14 × scale AND div ≥ 0.95 × conv AND div ≥ 0.75 × shear |
| Transform (3) | shear > 0.18 × scale |
| Interior (0) | otherwise |

where `scale = max(1.2, plate_speed × 1.25)`.

**Boundary strength** `bstr ∈ [0, 1]`: the magnitude of the dominant velocity
component, normalized against the maximum observed.

---

## 3. Crustal Structure and Isostasy

### 3.1 Physical Background

The elevation of a point on Earth's surface is primarily determined by the
thickness and density of the crust beneath it. Thick, light continental crust
"floats" high on the denser mantle (mountains). Thin, dense oceanic crust sits
low (ocean floor). This is **Airy isostasy**.

> **Turcotte & Schubert (2002)** *Geodynamics*, Chapter 3

### 3.2 Crustal Thickness from Plates

For each cell, crustal thickness `C` is computed from the plate field:

```
C_base = 35 km  (continental, buoyancy > 0)                             (3.1)
       =  7 km  (oceanic, buoyancy ≤ 0)

ΔC = { +bstr × 30 km   if convergent AND continental (CC collision)
      { +bstr × 18 km   if convergent AND oceanic (subduction arc)
      { −bstr × 20 km   if divergent (rifting)
      { +bstr ×  5 km   if transform

C = clamp(C_base + ΔC, 5, 75) km
```

**Physical basis:**
- Tibet has crust ~70 km from India-Asia collision (Dewey & Burke, 1973)
- Mid-ocean ridges thin to ~5 km
- Subduction arcs: ~25 km (Christensen & Mooney, 1995)

**⚠ Approximation:** Thickening is instantaneous and proportional to boundary
strength. In reality, collision builds crust over tens of millions of years, and
the relationship is nonlinear. Transform faults produce only minor transpression
(+2 km max), consistent with compressive bends on strike-slip faults (e.g.,
Alpine Fault, NZ).

### 3.3 Flexural Smoothing

Raw crustal thickness is step-function-like at plate boundaries. Real
lithosphere has **flexural rigidity** — it bends over ~200 km wavelengths.

> **Watts (2001)** *Isostasy and Flexure of the Lithosphere*

**Implementation:** 8 passes of 4-neighbor diffusion averaging.

Each pass computes the mean of the 4 orthogonal neighbors and blends:
```
C_new = 0.5 × C_old + 0.5 × C_neighbors_mean                          (3.2)
```

8 passes at cell size ~20 km → effective smoothing radius ≈ 8 × 20 / √2 ≈ 113
km. This approximates a flexural wavelength of ~200 km.

**⚠ Approximation:** True flexure solves a 4th-order PDE (D∇⁴w = q). The
diffusion averaging is a low-pass filter that produces similar spatial
smoothing but does not capture foreland basins or moat-and-bulge geometry.

### 3.4 Rock Type Classification

Each cell is assigned a `RockType` based on its tectonic setting:

| Setting | Rock Type | K_eff (m^0.5/yr) |
|---------|-----------|-------------------|
| Oceanic, no subduction | Basalt | 1.0 × 10⁻⁶ |
| Oceanic, subduction (convergent) | Schist | 1.2 × 10⁻⁶ |
| Continental, strong convergent (bstr > 0.5) | Granite | 0.5 × 10⁻⁶ |
| Continental, weak convergent (bstr ≤ 0.5) | Quartzite | 0.8 × 10⁻⁶ |
| Continental, divergent (bstr > 0.3) | Sandstone | 2.0 × 10⁻⁶ |
| Continental, transform | Schist | 1.2 × 10⁻⁶ |
| Continental, interior | Noise → Limestone/Sandstone/Granite | 0.5–3.0 × 10⁻⁶ |

> **Harel et al. (2016)** reported erodibility varies by ~2 orders of magnitude
> across lithologies. Our 6× range (0.5–3.0 × 10⁻⁶) is conservative.

**K_eff smoothing:** 3 passes of neighbor averaging. Fault zones (boundary
cells) get K_eff × 1.5 (Hovius & Stark 2006: fracturing increases
erodibility).

### 3.5 Airy Isostatic Elevation

Given crustal thickness C, the elevation relative to sea level is:

**For thickened crust (C > C_ref, land/mountains):**
```
h = (C − C_ref) × 1000 × (ρ_m − ρ_c_eff) / ρ_c_eff     [meters]      (3.3)
```

**For thinned crust (C < C_ref, ocean basins):**
```
h = (C − C_ref) × 1000 × (ρ_m − ρ_c_eff) / (ρ_m − ρ_w)  [meters]     (3.4)
```

**Densities:**

| Symbol | Value | Source |
|--------|-------|--------|
| ρ_c (continental) | 2800 kg/m³ | Turcotte & Schubert (2002) |
| ρ_c (oceanic) | 2900 kg/m³ | Turcotte & Schubert (2002) |
| ρ_m (mantle) | 3300 kg/m³ | Turcotte & Schubert (2002) |
| ρ_w (water) | 1030 kg/m³ | Standard |
| C_ref (continental) | 35 km | Christensen & Mooney (1995) |
| C_ref (oceanic) | 7 km | Christensen & Mooney (1995) |

**Thermal correction:**
```
ρ_c_eff = ρ_c × (1 − heat_anomaly × 0.02)                             (3.5)
```
Hot mantle reduces crustal density by ~2% per unit heat anomaly, causing
thermal uplift. This is a linearized approximation of thermal expansion
(α ≈ 3 × 10⁻⁵ K⁻¹, ΔT ≈ 200 K → Δρ/ρ ≈ 0.6%). The 2% coefficient is
~3× larger than thermal expansion alone, implicitly including dynamic
topography from mantle convection.

**Noise overlay** (topographic roughness not captured by isostasy):
```
h += 60 × noise(3f) + 35 × noise(6f) + 18 × noise(12f) + 7 × noise(24f)  (3.6)
```
where `f` is the base frequency and noise is spherical FBM. This adds ~120 m
of structure at scales from ~700 km down to ~80 km.

**⚠ Approximation:** Equations (3.3–3.4) are Airy (local) isostasy, not
flexural. This means each cell's elevation depends only on the crust directly
beneath it, ignoring the rigidity of the lithosphere. The flexural smoothing in
§3.3 partially compensates but cannot reproduce foreland basins or flexural
bulges.

---

## 4. Surface Evolution: Erosion

### 4.1 Physical Background

Once the initial isostatic relief is established, it is modified by fluvial
erosion and hillslope diffusion over geological time. The governing equation is:

```
∂h/∂t = U(x) − K_eff × A^m × S^n − κ∇²h                              (4.1)
```

| Term | Physical meaning |
|------|-----------------|
| U(x) | Tectonic uplift rate (m/yr) |
| K_eff × A^m × S^n | Fluvial erosion (stream power law) |
| κ∇²h | Hillslope diffusion (soil creep) |

> **Howard (1994)**, **Whipple & Tucker (1999)**: The stream power law.
> **Braun & Willett (2013)**: O(n) implicit solver.

### 4.2 Stream Power Law

The fluvial erosion rate at a point is proportional to the power dissipated by
flowing water at the bed:

```
E = K_eff × A^m × S^n                                                   (4.2)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| m (area exponent) | 0.5 | Whipple & Tucker (1999) |
| n (slope exponent) | 1.0 | Whipple & Tucker (1999) |
| K_eff (erodibility) | 0.5–3.0 × 10⁻⁶ m^0.5/yr | Harel et al. (2016), §3.4 |

A is drainage area (km²), S is local slope (dimensionless).

These m, n values correspond to the "detachment-limited" end-member where
erosion is controlled by the river's ability to detach bedrock, not by its
capacity to transport sediment.

### 4.3 Braun-Willett O(n) Implicit Solver

The implicit Euler discretization of Eq. (4.1) for a single time step Δt:

```
h_i^{new} = h_i^{old} + U_i × Δt + factor_i × h_{recv(i)}^{new}       (4.3)
            ─────────────────────────────────────────────────────
                           1 + factor_i

where factor_i = K_eff_i × Δt × A_i^m / Δx^n
```

`recv(i)` is the downstream receiver of cell i (D8 steepest descent).

**Algorithm:**

1. **D8 flow routing.** Each cell drains to its steepest-descent neighbor
   among 8 directions. Cells with no lower neighbor are local sinks.

2. **Topological sort.** Cells ordered from highest to lowest elevation.

3. **Drainage area.** Traverse high→low, accumulating: `A[recv(i)] += A[i]`.
   Each cell starts with `A = Δx²`.

4. **Implicit update.** Traverse low→high (downstream first). For each land
   cell (h > 0):
   ```
   factor = K × Δt × A^m / Δx^n
   h_new = (h_old + U×Δt + factor × h_recv_new) / (1 + factor)
   ```
   This is unconditionally stable for any Δt (implicit Euler).

5. **Hillslope diffusion** (explicit):
   ```
   Δh = κ × Δt × ∇²h                                                  (4.4)
   ```
   CFL stability requires `κ × Δt / Δx² < 0.25`.

**Boundary conditions:** Ocean cells (h ≤ 0) are held fixed. Coastal cells
drain to the ocean (receiver elevation = 0).

### 4.4 Multi-Epoch Evolution

The planet runs 3 geological epochs × 5 erosion steps = 15 total stream power
passes.

**Per epoch:**
```
Uplift rates:                                                            (4.5)
  U_convergent = bstr × 0.005 × speed_factor × epoch_scale   m/yr (up to 5 mm/yr)
  U_divergent  = −bstr × 0.002 × speed_factor                m/yr (up to −2 mm/yr)
  U_transform  = bstr × 0.001 × speed_factor                 m/yr

  speed_factor = plate_speed / 5.0
  epoch_scale  = 1 − epoch_fraction × 0.3
```

Each epoch runs `stream_power_step` 5 times with:
- Δt = 500,000 years
- Δx = km_per_cell × 1000 meters
- Total simulated time: 3 × 5 × 0.5 Myr = 7.5 Myr

**Isostatic relaxation** after each epoch:
```
h = 0.85 × h_eroded + 0.15 × h_isostatic_target                       (4.6)
```
This represents the mantle's viscous response: as erosion removes mass,
isostatic rebound raises the surface. The 85/15 blend corresponds to partial
relaxation over ~2.5 Myr, consistent with a Maxwell time of ~1 Myr for the
upper mantle.

**Note:** Convergent uplift of up to 5 mm/yr is consistent with GPS
observations of 0.5–10 mm/yr for active orogens (Bevis et al.). Combined with
7.5 Myr of simulation, this produces realistic mountain heights (6–8 km max)
balanced by Braun-Willett erosion.

### 4.5 Sea Level Determination

After erosion, sea level is set by the `ocean_percent` parameter:

1. Sort all elevation values
2. Sea level = elevation at the `ocean_percent` percentile
3. Subtract sea level from all cells

This guarantees that exactly `ocean_percent` of the surface is below zero
(underwater).

---

## 5. Climate

### 5.1 Physical Background

Earth's climate is governed by:
1. **Solar insolation** — decreases from equator to poles
2. **Atmospheric circulation** — Hadley, Ferrel, and Polar cells
3. **Orographic effects** — mountains force air upward, causing rain
4. **Lapse rate** — temperature decreases with altitude

> **Held & Hou (1980)** Hadley circulation theory.
> **Peixoto & Oort (1992)** Physics of Climate.
> **Holton & Hakim (2013)** Dynamic Meteorology.

### 5.2 Temperature Model

**Sea-level temperature** (quadratic fit to zonal-mean Earth data):
```
T_sea(lat) = 28.0 − 0.007 × lat²     [°C]                             (5.1)
```
This gives 28°C at the equator, ~0°C at 63°, −31°C at 90°.

**Fit quality:** This is a good approximation of the annual zonal mean.
Observations: ~27°C equator, ~−20°C at 80°N (annual mean). The quadratic
slightly overestimates polar cold.

**Environmental lapse rate:**
```
T(lat, h) = T_sea(lat) − Γ × h                                        (5.2)

Γ = 6.0 K/km = 0.006 K/m
```

> Source: ICAO Standard Atmosphere, Holton & Hakim (2013). The environmental
> lapse rate (6.0 K/km) is less than the dry adiabatic rate (9.8 K/km) due to
> latent heat release.

**Modifiers:**
```
T_final = T_sea − Γ×h + T_ocean + T_atmosphere + T_noise − T_aerosol   (5.3)

T_ocean     = +2°C if cell is underwater (ocean thermal inertia)
T_atmosphere = 5 × ln(1 + P_atm)  [°C, greenhouse effect]
T_noise     = ±3°C (spatial noise for local variability)
T_aerosol   = event-driven cooling (meteorite impacts)
```

**Final:** clamp to [−70, +55] °C.

**⚠ Approximation:** The greenhouse term `5 × ln(1 + P_atm)` is a rough fit
to Earth (1 bar → +3.5°C). Real greenhouse forcing depends on atmospheric
composition (CO₂, H₂O, CH₄), not just total pressure. For Earth-like planets
this is adequate; for Venus-like or Mars-like it would be wrong.

### 5.3 Atmospheric Circulation and Wind

Wind direction depends on latitude via the three-cell circulation model:

| Latitude band | Wind direction | Name |
|---------------|---------------|------|
| 0°–30° | From east (→ +x) | Trade winds |
| 30°–60° | From west (→ −x) | Westerlies |
| 60°–90° | From east (→ +x) | Polar easterlies |

Transitions are smooth (linear blend over 25°–35° and 55°–65°).

A meridional component `0.35 × sin(lat)` represents Coriolis deflection: zero
at the equator (no deflection), maximum at the poles.

> **Peixoto & Oort (1992)** — zonal wind pattern.

### 5.4 Precipitation Model

Precipitation is built from five components:

**Component 1: Hadley cell zonal base** (Held & Hou 1980)

```
P_hadley(lat):                                                           (5.4)
  |lat| < 10°:   2000 mm/yr   (ITCZ, deep convection)
  10°–20°:        2000 → 600   (subsidence begins)
  20°–35°:        600 → 250    (subtropical high, desert belt)
  35°–50°:        250 → 900    (westerly storm track)
  50°–70°:        900 → 400    (diminishing moisture)
  > 70°:          400 → 150    (polar desert)
```

**Component 2: Windward moisture** (exponential decay from coast)

```
P_windward = 600 × exp(−d_land × λ)    [mm/yr]                          (5.5)

λ = 0.02 × (km_per_cell / 20)    [per cell]
```

`d_land` is the distance (in cells) from the upwind coast, traced in the wind
direction. Moisture decays exponentially as air moves inland.

The trace continues over ocean cells (straits, bays) without breaking,
counting only land cells for distance. Maximum cumulative land distance: 50
cells (~1000 km).

**Component 3: Coastal exposure**

```
P_coastal = f_coast × 600    [mm/yr]                                     (5.6)
```

`f_coast` is the fraction of cells within 3 cells (~60 km) that are ocean.
Measures how "coastal" a location is.

Coastal exposure uses a multi-cell radius kernel counting ocean fraction
(omnidirectional).

**Component 4: Orographic lift** (Roe 2005, Smith 1979)

```
P_orographic = max(0, h_local − h_upwind) × 0.8    [mm/yr]             (5.7)
```

When terrain rises in the wind direction, air is forced upward, cools
adiabatically, and precipitates. The 0.8 factor converts elevation gain
(meters) to additional precipitation (mm/yr).

**Component 5: Rain shadow with distance decay** (Smith 1979)

```
P_shadow = max over upwind cells of:                                    (5.8)
    max(0, h_upwind − h_local) × 0.40 × exp(−d × Δx / 250 km)
```

Behind mountains, descending air warms and dries. A 2000 m range at the foot
creates ~800 mm/yr shadow. Shadow decays exponentially with distance from the
barrier (e-folding distance 250 km). At 500 km: 13% remaining. Maximum trace
distance: 500 km.

**Altitude factor** (Clausius-Clapeyron, cold air holds less moisture):

```
f_altitude = max(0.1, exp(−h × 0.0004))                                 (5.9)
```

At sea level: 1.0. At 1000 m: 0.67. At 3000 m: 0.30. At 5000 m: 0.14.
Floor at 0.1.

The coefficient 0.0004 /m derives from: lapse rate 6 K/km × Clausius-Clapeyron
~7%/K = 42%/km = 0.00042/m, rounded to 0.0004.

**Final precipitation (land cells):**

```
P = (P_hadley + P_windward + P_coastal + P_orographic − P_shadow)       (5.10)
    × f_altitude × f_atmosphere + noise × 60

f_atmosphere = √(P_atm)

P_final = clamp(P, 20, 4500) mm/yr
```

Ocean cells receive `P_hadley × 1.2`.

---

## 6. Hydrology

### 6.1 Physical Background

Water flows downhill. The **D8 algorithm** (O'Callaghan & Mark 1984) routes
each cell's water to its steepest-descent neighbor among 8 directions.
Accumulating flow gives drainage area, which determines river size.

### 6.2 Slope Calculation

For each cell, slope is the maximum elevation drop to any of the 8 neighbors,
normalized by distance:

```
slope = max over 8 neighbors of: (h_center − h_neighbor) / d            (6.1)
```

where d = 1 for orthogonal neighbors, √2 for diagonal.

### 6.3 Flow Routing (D8)

Each cell receives a flow direction pointing to its lowest neighbor:
```
receiver[i] = argmin_{j ∈ neighbors(i)} h[j]                           (6.2)
```

If all neighbors are higher (local pit), `receiver[i] = −1` (sink/lake).

### 6.4 Flow Accumulation

Cells sorted topologically (high → low). Each cell passes its area to its
receiver:

```
A[receiver[i]] += A[i]                                                   (6.3)
```

Cells with large `A` are rivers. The threshold for "visible river":

```
threshold = max(22, size × (0.00018 + fluvial_rounds × 0.00007))       (6.4)
```

### 6.5 River Intensity

```
river[i] = ((A − threshold) / (A_max − threshold))^0.45                (6.5)
           × (0.34 + 0.86 × slope / (slope + 35))
```

Rivers are stronger where drainage area is large and slope is moderate.

### 6.6 Fluvial Valley Carving

After initial hydrology, river channels are deepened:

```
Per round (1–3 rounds depending on detail preset):
  raw_cut = A^0.52 × (0.28 + river × 1.55) × (24 + round × 11)        (6.6)
  cut     = min(raw_cut, h × 0.23)

  For high elevations (h > 1600 m):
    h_new = lerp(h − cut, h, 0.42)     (diminishing carving)
```

Followed by 3×3 smoothing (20% blend toward neighborhood mean).

**⚠ Approximation:** This is a heuristic carving, not physics. Real valley
formation is handled by the stream power erosion in §4. This additional
carving is a post-processing step to enhance visual detail at coarse resolution
where stream power alone doesn't produce visible valleys.

---

## 7. Biome Classification

### 7.1 Method: Whittaker Diagram

> **Whittaker (1975)** *Communities and Ecosystems*

Biomes are classified by nearest-centroid matching in normalized
(temperature, precipitation) space:

```
biome[i] = argmin_k  ((T_i/15 − T_k/15)² + (P_i/500 − P_k/500)²)     (7.1)
```

### 7.2 Biome Centroids

| ID | Biome | T (°C) | P (mm/yr) |
|----|-------|--------|-----------|
| 0 | Ocean | — | — |
| 1 | Tundra / Ice Sheet | −8 | 250 |
| 2 | Boreal Forest / Taiga | 2 | 600 |
| 3 | Temperate Forest | 12 | 1200 |
| 4 | Temperate Grassland | 10 | 500 |
| 5 | Mediterranean Scrub | 16 | 550 |
| 6 | Tropical Rainforest | 25 | 2500 |
| 7 | Tropical Savanna | 25 | 1000 |
| 8 | Hot Desert | 25 | 150 |
| 9 | Subtropical Forest | 20 | 1400 |
| 10 | Alpine | (derived) | (derived) |
| 11 | Steppe | 8 | 350 |

The normalization (T/15, P/500) makes temperature and precipitation contribute
roughly equally to the distance metric. Without normalization, precipitation
(range ~4000 mm) would dominate over temperature (range ~100°C).

### 7.3 Alpine Override

```
treeline = 2000 + noise × 300    [meters]                               (7.2)

Soft transition from 1700 m to treeline.
If h > treeline and biome is forest/grassland → Alpine
```

### 7.4 Riparian Corridors

```
If river_map[i] > 0.12 and biome ∈ {Desert, Steppe, Mediterranean,      (7.3)
   Temperate Grassland} → upgrade to Savanna or Forest
```

River corridors support vegetation even in dry climates (gallery forests,
riparian woodlands).

---

## 8. Settlement Suitability

A simple habitability score:

```
settlement = 0.45 × comfort_T + 0.45 × comfort_P                       (8.1)
           + 0.42 × river + 0.30 × coast − h/3000

comfort_T = max(0, 1 − |T − 17| / 32)
comfort_P = max(0, 1 − |P − 1200| / 1800)

settlement_final = clamp(settlement, 0, 1)
```

**Interpretation:** Peak suitability at T = 17°C, P = 1200 mm/yr (temperate
forest climate). Rivers and coasts are bonuses. High elevation is a penalty.

**⚠ Approximation:** This is entirely heuristic. Real settlement depends on
soil fertility, navigable waterways, mineral resources, defensibility, trade
routes, and historical path dependence — none of which are modeled.

---

## 9. Island-as-Crop

### 9.1 Concept

The island scope does not run a separate pipeline. Instead:

1. Generate the full planet (§2–§8)
2. Select an interesting rectangular region
3. Crop and upsample to fine resolution
4. Refine with additional erosion, hydrology, climate, and biomes

### 9.2 Region Selection

**Scoring function** (`find_interesting_region`):

The planet grid is scanned with a window of size
`(ISLAND_WIDTH × planet_km / island_km) × (ISLAND_HEIGHT × planet_km / island_km)`.

For each window position:

```
land_fraction = count(h > 0) / window_size                              (9.1)
mix_score     = 1 − |land_fraction − 0.4| × 2.5            ← peak at 40%
coast_norm    = count(coast cells) / window_size            ∈ [0, 1]
elev_norm     = min(elev_range / 5000, 1)                   ∈ [0, 1]
boundary_norm = 1 + min(boundary_count / window_size, 0.5) × 2   ∈ [1, 2]

score = mix_score × coast_norm × elev_norm × boundary_norm + noise × 0.3
```

All components normalized to [0, 1] or [1, 2]. Noise amplitude 0.3 provides
seed-dependent variety without dominating the signal. Coast detection checks
4 cardinal neighbors. Polar 10% excluded.

### 9.3 Bicubic Upsampling

The coarse planet relief (e.g., 20 × 10 cells in the window) is upsampled to
1024 × 512 using **Catmull-Rom bicubic interpolation**:

```
For each fine cell (fx, fy):
  (cx, cy) = map to coarse coordinates
  h = bicubic_sample(coarse_relief, cx, cy)                             (9.2)
```

Catmull-Rom is a C¹-continuous interpolant that passes through all data points
(no overshooting like cubic B-spline, no ringing like Lanczos).

### 9.4 Fractal Detail

Sub-grid topographic detail is added via Fractal Brownian Motion (FBM):

```
detail = FBM(x, y, octaves=6, freq₀=8, decay=0.5)                     (9.3)
amplitude = |h| × 0.15      (15% of local elevation)

h_fine = h_upsampled + detail × amplitude
```

This adds realistic roughness at scales below the planet grid resolution. The
Hurst exponent (implicit in the 0.5 decay) is ~0.7, consistent with observed
terrain power spectra.

### 9.5 Edge Fade

To avoid sharp edges at the island boundary:

```
margin = 0.06 × grid_dimension ± noise                                  (9.4)
fade   = smoothstep(distance_from_edge / margin)
h      = h × fade + (−500) × (1 − fade)
```

Terrain smoothly transitions to −500 m ocean at edges.

### 9.6 Fine-Scale Erosion

10 stream power steps at island resolution:

```
Δt = 100,000 years                                                      (9.5)
Δx = km_per_cell × 1000   (≈ 400 m)
U  = 0                     (no uplift — planet already has large-scale relief)

K_eff from boundary type:
  convergent → 2.0 × 10⁻⁶
  divergent  → 5.0 × 10⁻⁶
  interior   → 8.0 × 10⁻⁶
```

These 10 steps simulate 1 Myr of erosion at fine scale, creating realistic
valleys, ridges, and drainage networks from the upsampled + fractalized terrain.

### 9.7 Island Climate

`compute_climate_unified` is called with a `CellCache` that maps island cells
to their real planetary coordinates. This means:

- Temperature uses the correct latitude
- Precipitation uses the correct Hadley cell zone
- Wind direction matches the latitude band

The climate is self-consistent with the planet — not independently generated.

### 9.8 Island Biomes and Settlement

Same functions as planet scope (§7, §8), applied to the fine-resolution
temperature and precipitation fields. Temperature and precipitation are
smoothed (5-pass neighbor averaging) before biome classification to avoid
noisy boundaries.

---

## 10. Complete Parameter Table

### 10.1 Physical Constants

| Symbol | Value | Units | Source |
|--------|-------|-------|--------|
| R_earth | 6371 | km | Standard |
| ρ_c (continental) | 2800 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_c (oceanic) | 2900 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_m | 3300 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_w | 1030 | kg/m³ | Standard |
| Γ (lapse rate) | 6.0 | K/km | ICAO, Holton & Hakim (2013) |
| g | 9.81 | m/s² | (user configurable) |

### 10.2 Tectonic Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| C_ref continental | 35 | km | Christensen & Mooney (1995) |
| C_ref oceanic | 7 | km | Christensen & Mooney (1995) |
| CC collision thickening | bstr × 30 | km | ~ Dewey & Burke (1973) |
| OC subduction thickening | bstr × 18 | km | ~ Christensen & Mooney (1995) |
| Rift thinning | bstr × 20 | km | General |
| Transform thickening | bstr × 2 | km | Minor transpression (Alpine Fault) |
| Crust clamp | [5, 75] | km | Physical bounds |
| Flexural smoothing | 8 passes | — | ~200 km wavelength |
| Fault K_eff boost | ×1.5 | — | Hovius & Stark (2006) |

### 10.3 Erosion Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| m (area exponent) | 0.5 | — | Whipple & Tucker (1999) |
| n (slope exponent) | 1.0 | — | Whipple & Tucker (1999) |
| κ (hillslope diffusivity) | 0.01 | m²/yr | CFL-stable for Δx=20km |
| K_eff Granite | 0.5 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Quartzite | 0.8 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Basalt | 1.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Schist | 1.2 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Sandstone | 2.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Limestone | 3.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| Convergent uplift | bstr × 0.005 × speed_factor | m/yr | GPS: 0.5–10 mm/yr (Bevis et al.) |
| Divergent subsidence | bstr × 0.002 × speed_factor | m/yr | — |
| Epochs × steps | 3 × 5 = 15 | — | Performance budget |
| Δt per step | 500,000 | yr | — |
| Total sim time | 7.5 | Myr | — |
| Isostatic relaxation | 85/15 | — | ~1 Myr Maxwell time |

### 10.4 Climate Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| T_equator | 28 | °C | Zonal mean obs. |
| T_lat coefficient | 0.007 | °C/deg² | Quadratic fit |
| Greenhouse factor | 5 × ln(1+P) | °C | **⚠ Rough** |
| Ocean T modifier | +2 | °C | Thermal inertia |
| ITCZ precipitation | 2000 | mm/yr | Held & Hou (1980) |
| Subtropical minimum | 250 | mm/yr | Held & Hou (1980) |
| Westerly belt max | 900 | mm/yr | — |
| Polar minimum | 150 | mm/yr | — |
| Windward decay rate | 0.02 | per cell | Normalized to ~20 km cells |
| Windward amplitude | 600 | mm/yr | — |
| Coast detection range | 3 cells | ~60 km | — |
| Coastal P amplitude | 600 | mm/yr | — |
| Orographic factor | 0.8 | mm/m | Roe (2005) |
| Rain shadow factor | 0.40 × exp(−d/250km) | mm/m | Smith (1979), distance decay |
| CC altitude factor | exp(−h × 4e-4), floor 0.1 | — | Clausius-Clapeyron exponential |
| Coriolis deflection | 0.35 × sin(lat) | — | f = 2Ω sin(φ) |
| Noise amplitude | 60 | mm/yr | — |

### 10.5 Biome Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| T normalization | /15 °C | Whittaker (1975) |
| P normalization | /500 mm | Whittaker (1975) |
| Treeline base | 2000 m | General alpine ecology |
| Treeline noise | ±300 m | — |
| Riparian threshold | river > 0.12 | — |

### 10.6 Island Crop Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| FBM octaves | 6 | — |
| FBM base frequency | 8.0 | — |
| FBM amplitude decay | 0.5 | /octave |
| Elevation amplitude | 15% | of local |h| |
| Edge margin | 6% | of grid dimension |
| Fine erosion steps | 10 | — |
| Fine Δt | 100,000 | yr |
| Fine total time | 1 | Myr |
| K_eff convergent | 2.0 × 10⁻⁶ | m^0.5/yr |
| K_eff divergent | 5.0 × 10⁻⁶ | m^0.5/yr |
| K_eff interior | 8.0 × 10⁻⁶ | m^0.5/yr |
| Smoothing passes (T/P) | 5 | before biome classification |

---

## 11. Pipeline Execution Order

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: seed, PlanetInputs, TectonicInputs, detail, scope      │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. PLATE TECTONICS  (§2)                          [0% → 24%]   │
│     build_irregular_plate_field → evolve_plate_field             │
│     → classify boundaries → plate_field, boundary_types, bstr   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. RELIEF PHYSICS  (§3, §4)                      [24% → 74%]   │
│     crustal_thickness → flexural_smooth → rock_type → K_eff     │
│     → isostatic_elevation → noise overlay                        │
│     → 3 epochs × 5 stream_power_step → isostatic relaxation     │
│     → sea_level from ocean_percent                               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. EVENTS  (§4.5*)                                              │
│     apply meteorite impacts, rifts, ocean shifts                 │
└──────────────────┬───────────────────────────────────────────────┘
                   │
            ┌──────┴──────┐
            │             │
     [Planet scope]  [Island scope]
            │             │
            ▼             ▼
┌──────────────┐  ┌──────────────────────────────────────────────┐
│ 4. SLOPE     │  │ 4'. ISLAND CROP  (§9)           [70% → 100%] │
│ 5. CLIMATE   │  │     find_interesting_region                   │
│ 6. HYDROLOGY │  │     → bicubic upsample + FBM fractal         │
│ 7. BIOMES    │  │     → edge fade                               │
│ 8. SETTLEMENT│  │     → 10× stream_power (fine)                 │
│              │  │     → slope → hydrology → fluvial carving     │
│ [74% → 100%]│  │     → climate (real lat/lon)                   │
│              │  │     → smooth T/P → biomes → settlement        │
└──────────────┘  └──────────────────────────────────────────────┘
            │             │
            ▼             ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: WasmSimulationResult                                    │
│    plates, boundary_types, height_map, slope_map, river_map,     │
│    lake_map, flow_direction, flow_accumulation, temperature_map, │
│    precipitation_map, biome_map, settlement_map                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Known Limitations and Future Work

### 12.1 Resolved Issues (fixed 2026-02-25)

All critical issues from the initial audit have been resolved:

| # | Issue | Was | Now |
|---|-------|-----|-----|
| 1 | Uplift rates | 0.003 m/yr (3 mm/yr max) | 0.005 m/yr (5 mm/yr max) |
| 2 | Rain shadow | No distance decay, 1200 km trace | Exponential decay (250 km e-fold), 500 km max |
| 3 | Coriolis | Constant 0.2 | 0.35 × sin(lat) |
| 4 | Clausius-Clapeyron | Linear: 1 − h×2.5e-4 | Exponential: exp(−h×4e-4) |
| 5 | Coast detection (scoring) | East neighbor only | 4 cardinal neighbors |
| 6 | Region scoring | Unnormalized (cells × meters) | All components in [0, 1] |
| 7 | Transform thickening | 5 km (heuristic) | 2 km (transpression only) |
| 8 | Windward moisture | Breaks on first ocean | Continues over water, counts land only |
| 9 | Divergent subsidence | 1 mm/yr | 2 mm/yr |

### 12.2 Missing Physics

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| Ocean age–depth | GDH1: d(t) = 2600 + 365√t | Ocean bathymetry is uniform; should deepen away from ridges |
| Flexural isostasy | D∇⁴w = q(x) − (ρ_m−ρ_c)gw | No foreland basins or flexural bulges |
| Slab pull / ridge push | Driving forces → plate speeds | Plate speeds are random, not physics-driven |
| No-Net-Rotation | ΣΩ_k × A_k = 0 | Plates can have net angular momentum (unphysical) |
| Sediment transport | Rivers deposit, not just erode | No deltas, alluvial plains, or sedimentary basins |
| Ocean currents | Sverdrup balance, thermohaline | No Gulf Stream warming or Humboldt cooling |
| Glaciation | Ice sheets at high lat / altitude | Tundra is classified but no ice dynamics |
| Seasonality | Axial tilt → seasons | Single annual mean, no monsoons |
| Subduction dip | Slab angle from age + speed | Uniform subduction geometry |

### 12.3 Remaining Heuristic Components

These parts of the code are not derived from physics:

1. **Plate field growth** (§2.2) — Dijkstra flood fill with tuned cost function
2. **Noise overlay** (§3.5, Eq. 3.6) — 120 m of arbitrary topographic noise
3. **Fluvial valley carving** (§6.6) — post-hoc deepening, not stream power
4. **Settlement model** (§8) — entirely heuristic comfort function
5. **FBM fractal detail** (§9.4) — sub-grid roughness is not geology

### 12.4 Remaining Improvement Priority

**Tier 1 (high impact, moderate effort):**
- Implement ocean age field + GDH1 bathymetry
- Add sediment deposition in lowlands

**Tier 2 (significant effort):**
- Flexural isostasy (4th-order PDE solver)
- Ocean currents (energy balance model)
- Seasonal climate cycle

---

## 13. References

1. **Braun, J. & Willett, S.D. (2013).** A very efficient O(n), implicit and parallel method to solve the stream power equation governing fluvial incision and landscape evolution. *Geomorphology*, 180-181, 170-179.

2. **Christensen, N.I. & Mooney, W.D. (1995).** Seismic velocity structure and composition of the continental crust: A global view. *J. Geophys. Res.*, 100(B6), 9761-9788.

3. **Cox, A. & Hart, R.B. (1986).** *Plate Tectonics: How It Works*. Blackwell Scientific.

4. **Harel, M.-A., Mudd, S.M. & Attal, M. (2016).** Global analysis of the stream power law parameters based on worldwide 10Be denudation rates. *Geomorphology*, 268, 184-196.

5. **Held, I.M. & Hou, A.Y. (1980).** Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. *J. Atmos. Sci.*, 37, 515-533.

6. **Holton, J.R. & Hakim, G.J. (2013).** *An Introduction to Dynamic Meteorology*, 5th ed. Academic Press.

7. **Hovius, N. & Stark, C.P. (2006).** Landslide-driven erosion and topographic evolution of active mountain belts. In *Landslides from Massive Rock Slope Failure*, S.G. Evans et al. (eds), NATO Science Series, 573-590.

8. **Howard, A.D. (1994).** A detachment-limited model of drainage basin evolution. *Water Resources Research*, 30(7), 2261-2285.

9. **O'Callaghan, J.F. & Mark, D.M. (1984).** The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28(3), 323-344.

10. **Peixoto, J.P. & Oort, A.H. (1992).** *Physics of Climate*. American Institute of Physics.

11. **Roe, G.H. (2005).** Orographic precipitation. *Annual Review of Earth and Planetary Sciences*, 33, 645-671.

12. **Smith, R.B. (1979).** The influence of mountains on the atmosphere. *Advances in Geophysics*, 21, 87-230.

13. **Turcotte, D.L. & Schubert, G. (2002).** *Geodynamics*, 2nd ed. Cambridge University Press.

14. **Watts, A.B. (2001).** *Isostasy and Flexure of the Lithosphere*. Cambridge University Press.

15. **Whipple, K.X. & Tucker, G.E. (1999).** Dynamics of the stream-power river incision model. *J. Geophys. Res.*, 104(B8), 17661-17674.

16. **Whittaker, R.H. (1975).** *Communities and Ecosystems*, 2nd ed. Macmillan.

---

*Document updated 2026-02-25. Reflects `rust/planet_engine/src/lib.rs` after
Phase A–C unification + physics fixes (9 issues resolved). All equation numbers
and parameter values verified against the implementation.*
