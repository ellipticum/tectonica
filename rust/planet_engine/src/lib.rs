use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

const WORLD_WIDTH: usize = 2048;
const WORLD_HEIGHT: usize = 1024;
const WORLD_SIZE: usize = WORLD_WIDTH * WORLD_HEIGHT;
const RADIANS: f32 = std::f32::consts::PI / 180.0;
const KILOMETERS_PER_DEGREE: f32 = 111.319_490_793;

#[derive(Clone)]
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state as f32 / 4_294_967_296.0
    }
}

#[inline]
fn clampf(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
fn index(x: usize, y: usize) -> usize {
    y * WORLD_WIDTH + x
}

#[inline]
fn random_range(rng: &mut Rng, min: f32, max: f32) -> f32 {
    min + rng.next_f32() * (max - min)
}

#[inline]
fn spherical_wrap(mut x: i32, mut y: i32) -> (usize, usize) {
    while y < 0 || y >= WORLD_HEIGHT as i32 {
        if y < 0 {
            y = -y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        } else {
            y = 2 * WORLD_HEIGHT as i32 - y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        }
    }

    let mut wrapped_x = x % WORLD_WIDTH as i32;
    if wrapped_x < 0 {
        wrapped_x += WORLD_WIDTH as i32;
    }
    let wrapped_y = y.clamp(0, WORLD_HEIGHT as i32 - 1);
    (wrapped_x as usize, wrapped_y as usize)
}

#[inline]
fn index_spherical(x: i32, y: i32) -> usize {
    let (sx, sy) = spherical_wrap(x, y);
    index(sx, sy)
}

fn quantile_sorted(sorted_values: &[f32], q: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let t = clampf(q, 0.0, 1.0) * (sorted_values.len() as f32 - 1.0);
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let k = t - lo as f32;
    sorted_values[lo] * (1.0 - k) + sorted_values[hi] * k
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = -f32::INFINITY;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

struct WorldCache {
    lat_by_y: Vec<f32>,
    lon_by_x: Vec<f32>,
    x_by_cell: Vec<f32>,
    y_by_cell: Vec<f32>,
    z_by_cell: Vec<f32>,
}

static WORLD_CACHE: OnceLock<WorldCache> = OnceLock::new();

fn world_cache() -> &'static WorldCache {
    WORLD_CACHE.get_or_init(|| {
        let mut lat_by_y = vec![0.0_f32; WORLD_SIZE];
        let mut lon_by_x = vec![0.0_f32; WORLD_SIZE];
        let mut x_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut y_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut z_by_cell = vec![0.0_f32; WORLD_SIZE];

        for y in 0..WORLD_HEIGHT {
            let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32);
            let lat_rad = lat_deg * RADIANS;
            let cos_lat = lat_rad.cos();
            for x in 0..WORLD_WIDTH {
                let lon_deg = (x as f32 + 0.5) * (360.0 / WORLD_WIDTH as f32) - 180.0;
                let lon_rad = lon_deg * RADIANS;
                let i = index(x, y);
                lat_by_y[i] = lat_deg;
                lon_by_x[i] = lon_deg;
                x_by_cell[i] = cos_lat * lon_rad.cos();
                y_by_cell[i] = cos_lat * lon_rad.sin();
                z_by_cell[i] = lat_rad.sin();
            }
        }

        WorldCache {
            lat_by_y,
            lon_by_x,
            x_by_cell,
            y_by_cell,
            z_by_cell,
        }
    })
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlanetInputs {
    pub radius_km: f32,
    pub gravity: f32,
    pub density: f32,
    pub rotation_hours: f32,
    pub axial_tilt_deg: f32,
    pub eccentricity: f32,
    pub atmosphere_bar: f32,
    pub ocean_percent: f32,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TectonicInputs {
    pub plate_count: i32,
    pub plate_speed_cm_per_year: f32,
    pub mantle_heat: f32,
}

#[derive(Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorldEventRecord {
    pub kind: String,
    pub latitude: f32,
    pub longitude: f32,
    #[serde(default)]
    pub diameter_km: f32,
    #[serde(default)]
    pub speed_kms: f32,
    #[serde(default)]
    pub angle_deg: f32,
    #[serde(default)]
    pub density_kg_m3: f32,
    #[serde(default)]
    pub radius_km: f32,
    #[serde(default)]
    pub magnitude: f32,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub energy_joule: Option<f64>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub seed: u32,
    pub planet: PlanetInputs,
    pub tectonics: TectonicInputs,
    pub events: Vec<WorldEventRecord>,
    #[serde(default)]
    pub generation_preset: Option<String>,
}

#[derive(Clone, Copy)]
enum RecomputeReason {
    Global,
    Tectonics,
    Events,
}

#[derive(Clone, Copy)]
enum GenerationPreset {
    Ultra,
    Fast,
    Balanced,
    Detailed,
}

#[derive(Clone, Copy)]
struct DetailProfile {
    buoyancy_smooth_passes: usize,
    erosion_rounds: usize,
    ocean_smooth_passes: usize,
    max_kernel_radius: i32,
}

fn parse_generation_preset(value: Option<&str>) -> GenerationPreset {
    match value.unwrap_or("balanced") {
        "ultra" => GenerationPreset::Ultra,
        "fast" => GenerationPreset::Fast,
        "detailed" => GenerationPreset::Detailed,
        _ => GenerationPreset::Balanced,
    }
}

fn detail_profile(preset: GenerationPreset) -> DetailProfile {
    match preset {
        GenerationPreset::Ultra => DetailProfile {
            buoyancy_smooth_passes: 1,
            erosion_rounds: 0,
            ocean_smooth_passes: 0,
            max_kernel_radius: 3,
        },
        GenerationPreset::Fast => DetailProfile {
            buoyancy_smooth_passes: 4,
            erosion_rounds: 1,
            ocean_smooth_passes: 1,
            max_kernel_radius: 4,
        },
        GenerationPreset::Detailed => DetailProfile {
            buoyancy_smooth_passes: 14,
            erosion_rounds: 3,
            ocean_smooth_passes: 3,
            max_kernel_radius: 7,
        },
        GenerationPreset::Balanced => DetailProfile {
            buoyancy_smooth_passes: 10,
            erosion_rounds: 2,
            ocean_smooth_passes: 2,
            max_kernel_radius: 6,
        },
    }
}

fn parse_reason(reason: &str) -> RecomputeReason {
    match reason {
        "tectonics" => RecomputeReason::Tectonics,
        "events" => RecomputeReason::Events,
        _ => RecomputeReason::Global,
    }
}

fn evaluate_recompute(reason: RecomputeReason) -> Vec<String> {
    let all = [
        "planet",
        "plates",
        "relief",
        "hydrology",
        "climate",
        "biomes",
        "settlement",
    ];

    let active: &[&str] = match reason {
        RecomputeReason::Global => &all,
        RecomputeReason::Tectonics => &all,
        RecomputeReason::Events => &["relief", "hydrology", "climate", "biomes", "settlement"],
    };

    all.iter()
        .filter(|id| active.contains(id))
        .map(|id| (*id).to_string())
        .collect()
}

fn ensure_event_energy(mut event: WorldEventRecord) -> WorldEventRecord {
    if event.kind == "meteorite" && event.energy_joule.is_none() {
        let r = (event.diameter_km * 1000.0) as f64 / 2.0;
        let mass = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) * event.density_kg_m3 as f64;
        event.energy_joule = Some(0.5 * mass * (event.speed_kms as f64 * 1000.0).powi(2));
    }
    event
}

#[derive(Clone)]
struct PlateSpec {
    lat: f32,
    lon: f32,
    speed: f32,
    dir_x: f32,
    dir_y: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct PlateVector {
    speed: f32,
    dir_x: f32,
    dir_y: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct ComputePlatesResult {
    plate_field: Vec<i16>,
    boundary_types: Vec<i8>,
    boundary_scores: Vec<f32>,
    boundary_normal_x: Vec<f32>,
    boundary_normal_y: Vec<f32>,
    boundary_strength: Vec<f32>,
    plate_vectors: Vec<PlateVector>,
}

#[derive(Clone)]
struct ReliefResult {
    relief: Vec<f32>,
    sea_level: f32,
}

#[derive(Clone, Copy, Debug)]
struct FrontierNode {
    cost: f32,
    index: usize,
    plate: i16,
}

impl PartialEq for FrontierNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
            && self.index == other.index
            && self.plate == other.plate
    }
}

impl Eq for FrontierNode {}

impl PartialOrd for FrontierNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.index.cmp(&other.index))
            .then_with(|| self.plate.cmp(&other.plate))
    }
}

struct MinCostQueue {
    data: BinaryHeap<Reverse<FrontierNode>>,
}

impl MinCostQueue {
    fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, item: FrontierNode) {
        self.data.push(Reverse(item));
    }

    fn pop(&mut self) -> Option<FrontierNode> {
        self.data.pop().map(|Reverse(node)| node)
    }
}

#[derive(Clone, Copy)]
struct GrowthParam {
    drift_x: f32,
    drift_y: f32,
    spread: f32,
    roughness: f32,
    freq_a: f32,
    freq_b: f32,
    freq_c: f32,
    freq_d: f32,
    phase_a: f32,
    phase_b: f32,
}

fn lat_lon_to_index(lat: f32, lon: f32) -> usize {
    let y = clampf(
        ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
        0.0,
        (WORLD_HEIGHT - 1) as f32,
    ) as usize;
    let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32 % WORLD_WIDTH as i32)
        + WORLD_WIDTH as i32)
        % WORLD_WIDTH as i32;
    index(x as usize, y)
}

fn nearest_free_index(start: usize, occupied: &[u8]) -> usize {
    if occupied[start] == 0 {
        return start;
    }

    let sx = start % WORLD_WIDTH;
    let sy = start / WORLD_WIDTH;
    let max_radius = WORLD_WIDTH.max(WORLD_HEIGHT) as i32;

    for radius in 1..max_radius {
        for dy in -radius..=radius {
            let y = sy as i32 + dy;
            if y < 0 || y >= WORLD_HEIGHT as i32 {
                continue;
            }
            let span = radius - dy.abs();
            let candidates = [sx as i32 - span, sx as i32 + span];
            for cx in candidates {
                let mut x = cx % WORLD_WIDTH as i32;
                if x < 0 {
                    x += WORLD_WIDTH as i32;
                }
                let idx = index(x as usize, y as usize);
                if occupied[idx] == 0 {
                    return idx;
                }
            }
        }
    }

    start
}

fn build_irregular_plate_field(plates: &[PlateSpec], seed: u32, cache: &WorldCache) -> Vec<i16> {
    let mut plate_field = vec![-1_i16; WORLD_SIZE];
    let mut open_cost = vec![f32::INFINITY; WORLD_SIZE];
    let mut occupied_seeds = vec![0_u8; WORLD_SIZE];
    let mut growth_rng = Rng::new(seed ^ 0x9e37_79b9);
    let mut queue = MinCostQueue::new();

    let growth_params: Vec<GrowthParam> = plates
        .iter()
        .map(|plate| {
            let len = (plate.dir_x.hypot(plate.dir_y)).max(1.0);
            GrowthParam {
                drift_x: plate.dir_x / len,
                drift_y: plate.dir_y / len,
                spread: random_range(&mut growth_rng, 0.85, 1.25),
                roughness: random_range(&mut growth_rng, 0.28, 1.05),
                freq_a: random_range(&mut growth_rng, 0.045, 0.145),
                freq_b: random_range(&mut growth_rng, 0.055, 0.16),
                freq_c: random_range(&mut growth_rng, 0.08, 0.22),
                freq_d: random_range(&mut growth_rng, 0.07, 0.2),
                phase_a: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
                phase_b: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
            }
        })
        .collect();

    for (plate_id, plate) in plates.iter().enumerate() {
        let seed_index =
            nearest_free_index(lat_lon_to_index(plate.lat, plate.lon), &occupied_seeds);
        occupied_seeds[seed_index] = 1;
        open_cost[seed_index] = 0.0;
        queue.push(FrontierNode {
            cost: 0.0,
            index: seed_index,
            plate: plate_id as i16,
        });
    }

    const STEPS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, std::f32::consts::SQRT_2),
        (-1, 1, std::f32::consts::SQRT_2),
        (1, -1, std::f32::consts::SQRT_2),
        (-1, -1, std::f32::consts::SQRT_2),
    ];

    let mut assigned = 0usize;
    while queue.size() > 0 && assigned < WORLD_SIZE {
        let Some(node) = queue.pop() else {
            break;
        };

        if node.cost > open_cost[node.index] + 1e-6 {
            continue;
        }
        if plate_field[node.index] != -1 {
            continue;
        }

        plate_field[node.index] = node.plate;
        assigned += 1;

        let x = node.index % WORLD_WIDTH;
        let y = node.index / WORLD_WIDTH;
        let gp = growth_params[node.plate as usize];

        for (dx, dy, w) in STEPS {
            let j = index_spherical(x as i32 + dx, y as i32 + dy);
            if plate_field[j] != -1 {
                continue;
            }

            let lat = cache.lat_by_y[j];
            let sx = cache.x_by_cell[j];
            let sy = cache.y_by_cell[j];
            let sz = cache.z_by_cell[j];
            let scale_a = 0.8 + gp.freq_a * 24.0;
            let scale_b = 0.8 + gp.freq_b * 24.0;
            let scale_c = 0.8 + gp.freq_c * 24.0;
            let scale_d = 0.8 + gp.freq_d * 24.0;
            let wave_a = (sx * scale_a + sy * scale_b + gp.phase_a).sin();
            let wave_b = (sy * scale_c - sz * scale_d + gp.phase_b).cos();
            let rough_factor = 1.0 + gp.roughness * (0.22 * wave_a + 0.18 * wave_b);
            let drift_align = dx as f32 * gp.drift_x + dy as f32 * gp.drift_y;
            let drift_factor = 1.03 - 0.12 * drift_align;
            let polar_factor = 1.0 + (lat.abs() / 90.0) * 0.1;
            let step_cost = (w * gp.spread * rough_factor * drift_factor * polar_factor).max(0.08);
            let next_cost = node.cost + step_cost;

            if next_cost + 1e-6 < open_cost[j] {
                open_cost[j] = next_cost;
                queue.push(FrontierNode {
                    cost: next_cost,
                    index: j,
                    plate: node.plate,
                });
            }
        }
    }

    if assigned < WORLD_SIZE {
        for i in 0..WORLD_SIZE {
            if plate_field[i] != -1 {
                continue;
            }
            let mut best_plate = 0usize;
            let mut best_cost = f32::INFINITY;
            let cell_x = cache.x_by_cell[i];
            let cell_y = cache.y_by_cell[i];
            let cell_z = cache.z_by_cell[i];

            for (p, plate) in plates.iter().enumerate() {
                let lat_r = plate.lat * RADIANS;
                let lon_r = plate.lon * RADIANS;
                let c = lat_r.cos();
                let px = c * lon_r.cos();
                let py = c * lon_r.sin();
                let pz = lat_r.sin();
                let d = 1.0 - (cell_x * px + cell_y * py + cell_z * pz);
                if d < best_cost {
                    best_cost = d;
                    best_plate = p;
                }
            }

            plate_field[i] = best_plate as i16;
        }
    }

    plate_field
}

fn compute_plates(
    _planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    seed: u32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ComputePlatesResult {
    let plate_count = tectonics.plate_count.clamp(2, 20) as usize;
    let mut rng = Rng::new(seed.wrapping_add((plate_count as u32).wrapping_mul(7_919)));
    let mut plates: Vec<PlateSpec> = Vec::with_capacity(plate_count);

    for _ in 0..plate_count {
        let lat = random_range(&mut rng, -90.0, 90.0);
        let lon = random_range(&mut rng, -180.0, 180.0);
        let speed =
            (random_range(&mut rng, 0.5, 1.5) * tectonics.plate_speed_cm_per_year).max(0.001);
        let dir = random_range(&mut rng, 0.0, std::f32::consts::PI * 2.0);
        plates.push(PlateSpec {
            lat,
            lon,
            speed,
            dir_x: dir.cos() * speed,
            dir_y: dir.sin() * speed,
            heat: random_range(
                &mut rng,
                (tectonics.mantle_heat * 0.5).max(1.0),
                tectonics.mantle_heat * 1.5,
            ),
            buoyancy: random_range(&mut rng, -1.0, 1.0),
        });
    }

    let plate_field = build_irregular_plate_field(&plates, seed, cache);

    let plate_vectors: Vec<PlateVector> = plates
        .iter()
        .map(|plate| PlateVector {
            speed: plate.speed,
            dir_x: plate.dir_x,
            dir_y: plate.dir_y,
            heat: plate.heat,
            buoyancy: plate.buoyancy,
        })
        .collect();

    let mut boundary_types = vec![0_i8; WORLD_SIZE];
    let mut boundary_scores = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_normal_x = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_normal_y = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_strength = vec![0.0_f32; WORLD_SIZE];
    let boundary_scale = (tectonics.plate_speed_cm_per_year * 1.25).max(1.2);

    const NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (1, 1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, 1, std::f32::consts::FRAC_1_SQRT_2),
    ];

    for y in 0..WORLD_HEIGHT {
        progress.phase(progress_base, progress_span, y as f32 / WORLD_HEIGHT as f32);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let plate_a = plate_field[i] as usize;
            let a = &plate_vectors[plate_a];
            let mut score = 0.0_f32;
            let mut normal_x = 0.0_f32;
            let mut normal_y = 0.0_f32;
            let mut has_different_neighbor = false;

            for (dx, dy, w) in NEIGHBORS {
                let j = index_spherical(x as i32 + dx, y as i32 + dy);
                let plate_b = plate_field[j] as usize;
                if plate_b == plate_a {
                    continue;
                }
                has_different_neighbor = true;
                let b = &plate_vectors[plate_b];
                let rel_x = b.dir_x - a.dir_x;
                let rel_y = b.dir_y - a.dir_y;
                let edge_score = (rel_x * dx as f32 + rel_y * dy as f32) * w;
                if edge_score.abs() > score.abs() {
                    score = edge_score;
                    normal_x = dx as f32;
                    normal_y = dy as f32;
                }
            }

            boundary_scores[i] = score;
            boundary_normal_x[i] = normal_x;
            boundary_normal_y[i] = normal_y;
            boundary_strength[i] = clampf(score.abs() / boundary_scale, 0.0, 1.0);

            if score > 0.2 * boundary_scale {
                boundary_types[i] = 1;
            } else if score < -0.2 * boundary_scale {
                boundary_types[i] = 2;
            } else if has_different_neighbor {
                boundary_types[i] = 3;
            } else {
                boundary_types[i] = 0;
            }
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    ComputePlatesResult {
        plate_field,
        boundary_types,
        boundary_scores,
        boundary_normal_x,
        boundary_normal_y,
        boundary_strength,
        plate_vectors,
    }
}

fn normalize_height_range(relief: &mut [f32], planet: &PlanetInputs, tectonics: &TectonicInputs) {
    let mut land_heights: Vec<f32> = Vec::new();
    let mut ocean_depths: Vec<f32> = Vec::new();

    for &h in relief.iter() {
        if h > 0.0 {
            land_heights.push(h);
        } else if h < 0.0 {
            ocean_depths.push(-h);
        }
    }

    if land_heights.len() < 8 || ocean_depths.len() < 8 {
        return;
    }

    land_heights.sort_by(|a, b| a.total_cmp(b));
    ocean_depths.sort_by(|a, b| a.total_cmp(b));

    let land_ref = quantile_sorted(&land_heights, 0.995).max(1.0);
    let ocean_ref = quantile_sorted(&ocean_depths, 0.995).max(1.0);

    let speed_factor = clampf(tectonics.plate_speed_cm_per_year / 5.0, 0.35, 2.8);
    let heat_factor = clampf(tectonics.mantle_heat / 55.0, 0.45, 2.2);
    let gravity_factor = clampf(9.81 / planet.gravity.max(1.0), 0.45, 2.4);
    let ocean_factor = clampf(planet.ocean_percent / 67.0, 0.45, 1.9);

    let target_land_max = clampf(
        6000.0 * speed_factor.powf(0.55) * heat_factor.powf(0.45) * gravity_factor.powf(0.7),
        1800.0,
        9000.0,
    );
    let target_ocean_depth = clampf(
        7500.0 * speed_factor.powf(0.5) * heat_factor.powf(0.35) * ocean_factor.powf(0.55),
        1400.0,
        12000.0,
    );

    let land_scale = target_land_max / land_ref;
    let ocean_scale = target_ocean_depth / ocean_ref;

    for h in relief.iter_mut() {
        if *h > 0.0 {
            *h = clampf(*h * land_scale, -12000.0, 9000.0);
        } else if *h < 0.0 {
            *h = clampf(-(-*h * ocean_scale), -12000.0, 9000.0);
        }
    }
}

fn reshape_ocean_boundaries(relief: &mut [f32], boundary_types: &[i8], boundary_strength: &[f32]) {
    let mut deepest = 0.0_f32;
    for &h in relief.iter() {
        if h < 0.0 {
            deepest = deepest.max(-h);
        }
    }
    let depth_scale = deepest.max(1.0);

    for i in 0..WORLD_SIZE {
        if relief[i] >= 0.0 {
            continue;
        }
        let t = boundary_types[i];
        if t == 0 {
            continue;
        }

        let depth_t = clampf(-relief[i] / depth_scale, 0.0, 1.0);
        let strength = clampf(boundary_strength[i], 0.0, 1.0);

        if t == 2 {
            let uplift = (180.0 + 900.0 * strength) * (0.7 + 0.3 * (1.0 - depth_t));
            relief[i] = (relief[i] + uplift).min(-15.0);
        } else if t == 3 {
            let uplift = (40.0 + 180.0 * strength) * (0.65 + 0.35 * (1.0 - depth_t));
            relief[i] += uplift;
        }
    }

    let mut ocean_smoothed = relief.to_vec();
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            if relief[i] >= 0.0 {
                continue;
            }
            let mut sum = relief[i] * 3.4;
            let mut wsum = 3.4;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        continue;
                    }
                    let w = if ox == 0 || oy == 0 { 0.6 } else { 0.45 };
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            ocean_smoothed[i] = sum / wsum.max(1e-6);
        }
    }
    relief.copy_from_slice(&ocean_smoothed);
}

fn apply_coastal_detail(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    for i in 0..WORLD_SIZE {
        let h = relief[i];
        let near_sea = (-h.abs() / 900.0).exp();
        if near_sea < 0.035 {
            continue;
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let seed_phase = seed as f32 * 0.00091;
        let warp_x =
            sx + 0.22 * (sy * 6.1 + seed_phase).sin() + 0.14 * (sz * 7.4 - seed_phase * 1.7).cos();
        let warp_y = sy
            + 0.2 * (sz * 5.8 - seed_phase * 1.2).sin()
            + 0.12 * (sx * 6.9 + seed_phase * 0.8).cos();
        let warp_z = sz
            + 0.18 * (sx * 5.2 + seed_phase * 1.5).sin()
            + 0.1 * (sy * 7.2 - seed_phase * 0.6).cos();
        let n = (warp_x * 7.6 + warp_y * 6.4 + warp_z * 5.1 + seed_phase * 1.9).sin() * 0.62
            + (warp_y * 8.1 - warp_z * 5.7 + warp_x * 4.6 - seed_phase * 1.3).cos() * 0.38;
        let coast_weight = near_sea.powf(1.18);
        relief[i] += coast_weight * n * 92.0;
    }
}

fn cleanup_coastal_speckles(relief: &mut [f32]) {
    let mut next = relief.to_vec();

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 260.0 {
                continue;
            }

            let mut land = 0;
            let mut water = 0;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        land += 1;
                    } else {
                        water += 1;
                    }
                }
            }

            if h >= 0.0 && water >= 7 {
                next[i] = next[i].min(-18.0 - h.abs() * 0.15);
            } else if h < 0.0 && land >= 7 {
                next[i] = next[i].max(18.0 + h.abs() * 0.15);
            }
        }
    }

    relief.copy_from_slice(&next);
}

fn apply_ocean_profile(
    relief: &mut [f32],
    boundary_types: &[i8],
    boundary_strength: &[f32],
    planet: &PlanetInputs,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let mut distance_to_coast = vec![-1_i32; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.1 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            if relief[i] >= 0.0 {
                continue;
            }

            let mut is_coast_adjacent = false;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        is_coast_adjacent = true;
                        break;
                    }
                }
                if is_coast_adjacent {
                    break;
                }
            }

            if is_coast_adjacent {
                distance_to_coast[i] = 0;
                queue.push_back(i);
            }
        }
    }

    const FLOW_STEPS: [(i32, i32); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ];

    while let Some(i) = queue.pop_front() {
        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let next_dist = distance_to_coast[i] + 1;

        for (dx, dy) in FLOW_STEPS {
            let j = index_spherical(x as i32 + dx, y as i32 + dy);
            if relief[j] >= 0.0 || distance_to_coast[j] >= 0 {
                continue;
            }
            distance_to_coast[j] = next_dist;
            queue.push_back(j);
        }
    }
    progress.phase(progress_base, progress_span, 0.2);

    let mut max_distance = 1_i32;
    for i in 0..WORLD_SIZE {
        if relief[i] < 0.0 && distance_to_coast[i] > max_distance {
            max_distance = distance_to_coast[i];
        }
    }

    let shelf_cells = clampf(4.8 + (planet.radius_km / 6371.0 - 1.0) * 1.2, 3.8, 8.4);
    let slope_cells = clampf(15.0 + (planet.ocean_percent - 67.0) * 0.08, 11.0, 23.0);
    let abyssal_base = clampf(
        4200.0 + (planet.ocean_percent - 67.0) * 26.0,
        3600.0,
        5600.0,
    );

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.2 + 0.55 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        if relief[i] >= 0.0 {
            continue;
        }

        let d = distance_to_coast[i].max(0) as f32;
        let mut target_depth = if d <= shelf_cells {
            let t = clampf(d / shelf_cells, 0.0, 1.0);
            -25.0 - 220.0 * t.powf(1.8)
        } else if d <= shelf_cells + slope_cells {
            let t = clampf((d - shelf_cells) / slope_cells, 0.0, 1.0);
            -245.0 - (abyssal_base - 245.0) * t.powf(0.78)
        } else {
            let t = clampf(
                (d - shelf_cells - slope_cells) / (max_distance as f32 + 1.0),
                0.0,
                1.0,
            );
            -abyssal_base - 1850.0 * t.powf(0.62)
        };

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let basin_weight = clampf(
            (d - shelf_cells) / (shelf_cells + slope_cells + 1.0),
            0.0,
            1.0,
        );
        let seed_phase = seed as f32 * 0.0012;
        let undulation = (sx * 7.2 + sy * 5.3 + sz * 4.1 + seed_phase).sin() * 0.6
            + (sy * 8.6 - sz * 4.9 + sx * 3.7 - seed_phase * 1.35).cos() * 0.4;
        target_depth += undulation * (120.0 + 320.0 * basin_weight);

        let strength = clampf(boundary_strength[i], 0.0, 1.0);
        target_depth += match boundary_types[i] {
            1 => -920.0 * strength,
            2 => 360.0 * strength,
            3 => -140.0 * strength,
            _ => 0.0,
        };

        let blend = 0.74 + 0.21 * basin_weight;
        relief[i] = clampf(
            relief[i] * (1.0 - blend) + target_depth * blend,
            -12000.0,
            -5.0,
        );
    }

    let mut smoothed = relief.to_vec();
    for pass in 0..detail.ocean_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let pass_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32)
                / detail.ocean_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.75 + 0.25 * pass_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                if relief[i] >= 0.0 {
                    continue;
                }
                let mut sum = relief[i] * 2.8;
                let mut wsum = 2.8;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.5 } else { 0.35 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                smoothed[i] = sum / wsum.max(1e-6);
            }
        }
        for i in 0..WORLD_SIZE {
            if relief[i] < 0.0 {
                relief[i] = smoothed[i];
            }
        }
    }
    progress.phase(progress_base, progress_span, 1.0);
}

fn compute_relief(
    planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ReliefResult {
    let mut relief = vec![0.0_f32; WORLD_SIZE];

    let seed_mix = (((planet.radius_km * 1000.0) as i32)
        ^ ((tectonics.plate_speed_cm_per_year * 1000.0) as i32)
        ^ ((tectonics.mantle_heat * 1000.0) as i32)
        ^ (seed.wrapping_mul(2_654_435_761) as i32)) as u32;
    let mut random_seed = Rng::new(seed_mix);

    let macro_a = random_range(&mut random_seed, 0.012, 0.03);
    let macro_b = random_range(&mut random_seed, 0.008, 0.02);
    let macro_c = random_range(&mut random_seed, 0.006, 0.016);
    let phase_a = random_range(
        &mut random_seed,
        -std::f32::consts::PI,
        std::f32::consts::PI,
    );
    let phase_b = random_range(
        &mut random_seed,
        -std::f32::consts::PI,
        std::f32::consts::PI,
    );
    let phase_c = random_range(
        &mut random_seed,
        -std::f32::consts::PI,
        std::f32::consts::PI,
    );
    let k_relief = 560.0_f32;
    let max_boundary_score = (tectonics.plate_speed_cm_per_year * 2.5).max(1.0);
    let base_kernel_radius = 3_i32;

    // Smooth per-plate buoyancy so continent/ocean macro-shape is not cut by hard plate polygons.
    let mut buoyancy_field = vec![0.0_f32; WORLD_SIZE];
    for (i, b) in buoyancy_field.iter_mut().enumerate() {
        let pid = plates.plate_field[i] as usize;
        *b = plates.plate_vectors[pid].buoyancy;
    }
    let mut buoyancy_scratch = vec![0.0_f32; WORLD_SIZE];
    for pass in 0..detail.buoyancy_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let pass_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32)
                / detail.buoyancy_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.16 * pass_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = buoyancy_field[i] * 0.34;
                let mut wsum = 0.34;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.125 } else { 0.07 };
                        sum += buoyancy_field[j] * w;
                        wsum += w;
                    }
                }
                buoyancy_scratch[i] = sum / wsum.max(1e-6);
            }
        }
        buoyancy_field.copy_from_slice(&buoyancy_scratch);
    }

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.16 + 0.42 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        let plate_id = plates.plate_field[i] as usize;
        let plate_speed = plates.plate_vectors[plate_id].speed;
        let heat = plates.plate_vectors[plate_id].heat;
        let plate_buoyancy = buoyancy_field[i];
        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;

        let mut conv_influence = 0.0_f32;
        let mut div_influence = 0.0_f32;
        let mut transform_influence = 0.0_f32;

        let local_kernel = (base_kernel_radius
            + (plate_speed / 1.8).floor() as i32
            + ((heat / 30.0) * 2.0).floor() as i32)
            .min(detail.max_kernel_radius);

        for oy in -local_kernel..=local_kernel {
            for ox in -local_kernel..=local_kernel {
                let j = index_spherical(x as i32 + ox, y as i32 + oy);
                let t = plates.boundary_types[j];
                if t == 0 {
                    continue;
                }

                let source_plate = plates.plate_field[j] as usize;
                let source_heat = plates.plate_vectors[source_plate].heat;
                let bx = cache.x_by_cell[j];
                let by = cache.y_by_cell[j];
                let bz = cache.z_by_cell[j];
                let ridge_noise =
                    0.65 + 0.35 * (bx * 3.9 + by * 5.2 + bz * 4.3 + seed as f32 * 0.0027).sin();
                let source_heat_norm = source_heat / tectonics.mantle_heat.max(1.0);
                let heat_width = 0.75 + source_heat_norm * 1.6 + (source_heat_norm - 0.55) * 0.9;
                let local_width = clampf(0.65 + heat_width * ridge_noise, 0.5, 4.8);
                let s = clampf(
                    plates.boundary_scores[j].abs() / max_boundary_score,
                    0.0,
                    1.0,
                );
                let n_normal_x = plates.boundary_normal_x[j];
                let n_normal_y = plates.boundary_normal_y[j];
                let normal_len = n_normal_x.hypot(n_normal_y).max(1.0);
                let normal_x = n_normal_x / normal_len;
                let normal_y = n_normal_y / normal_len;
                let tangent_x = -normal_y;
                let tangent_y = normal_x;
                let across = ox as f32 * normal_x + oy as f32 * normal_y;
                let along = ox as f32 * tangent_x + oy as f32 * tangent_y;
                let boundary_bias = clampf(plates.boundary_strength[j] * 1.25, 0.0, 1.4);
                let sigma_across = (local_width * (0.85 + 1.1 * s)
                    + 0.05 * ((ox as f32) * 3.1 + (oy as f32) * 2.3).sin())
                .max(0.55);
                let sigma_along = (local_width * (3.0 + 3.2 * boundary_bias + 1.6 * s)
                    + 1.0 * ((oy as f32) * 2.1 - (ox as f32) * 1.9).cos())
                .max(1.6);
                let anisotropy = (across * across) / (sigma_across * sigma_across)
                    + (along * along) / (sigma_along * sigma_along);
                let w = (-anisotropy).exp();
                let width_tuning = 0.32 + 0.95 * s + 0.28 * boundary_bias;
                let chain_segment =
                    0.35 + 0.65 * (0.5 + 0.5 * (bx * 9.4 + by * 7.6 + bz * 5.8 + phase_a).sin());
                let chain_cluster =
                    0.45 + 0.55 * (0.5 + 0.5 * (by * 8.8 - bz * 6.3 + bx * 4.2 + phase_b).cos());
                let mut intensity = w * width_tuning;

                if t == 1 {
                    intensity *= chain_segment * chain_cluster;
                    conv_influence += intensity;
                } else if t == 2 {
                    intensity *= 0.55 + 0.45 * chain_segment;
                    div_influence += intensity;
                } else {
                    transform_influence += intensity;
                }
            }
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let warp_x = sx + 0.34 * (sy * 4.6 + phase_a).sin() + 0.21 * (sz * 5.1 + phase_b).cos();
        let warp_y = sy + 0.31 * (sz * 4.2 - phase_b).sin() + 0.18 * (sx * 5.4 + phase_c).cos();
        let warp_z = sz + 0.27 * (sx * 4.9 + phase_c).sin() + 0.16 * (sy * 5.8 - phase_a).cos();

        let continental_signal = (warp_x * 2.6 + warp_y * 2.1 + phase_a).sin() * 0.92
            + (warp_y * 3.4 - warp_z * 1.8 + phase_b).cos() * 0.78
            + ((warp_x + warp_z * 0.7) * 3.9 + phase_c).sin() * 0.58
            + ((warp_y - warp_x * 0.45) * 4.3 + phase_a * 0.65).cos() * 0.38;
        let regional_signal = (warp_x * 4.2 + warp_y * 3.5 + warp_z * 2.8 + phase_b).sin()
            * (warp_y * 2.4 - warp_z * 4.1 + warp_x * 1.7 + phase_c).cos();
        let crust_mask_raw =
            continental_signal * 0.95 + regional_signal * 0.45 + plate_buoyancy * 0.18;
        let crust_mask = 1.0 / (1.0 + (-crust_mask_raw).exp());

        let mut base = 0.0_f32;
        if conv_influence > 0.03 {
            let conv_shape = conv_influence.powf(0.78);
            base += ((k_relief * plate_speed) / planet.gravity.max(1.0))
                * (0.04 + 0.12 * conv_shape.atan());
            base += 105.0 * conv_shape.ln_1p();
        }
        if div_influence > 0.03 {
            let div_shape = div_influence.powf(0.84);
            base -= (95.0 + 6.0 * plate_speed) * (0.06 + 0.18 * div_shape.atan());
        }
        if transform_influence > 0.03 {
            base += 18.0 * (plate_speed - 1.0) * transform_influence.atan();
        }

        let tectonic_weight = 0.16 + 0.84 * crust_mask;
        base *= tectonic_weight;

        let intraplate_signal = (warp_x * 7.1 + warp_y * 6.6 + phase_a * 0.25).sin()
            * (warp_y * 6.9 - warp_z * 5.5 + phase_b * 0.3).cos();

        let macro_noise = (sx * (4.8 + macro_a * 38.0) + sy * (3.6 + macro_b * 34.0) + phase_a)
            .sin()
            * 140.0
            + (sy * (4.1 + macro_c * 30.0) - sz * (3.2 + macro_a * 26.0) + phase_b).sin() * 90.0
            + (sz * (4.7 + macro_b * 22.0) + sx * (2.9 + macro_c * 18.0) + phase_c).cos() * 70.0;
        let continental_base = (crust_mask - 0.5) * 3600.0;
        let macro_base = continental_base
            + intraplate_signal * 260.0
            + plate_buoyancy * 70.0
            + regional_signal * 110.0;
        let noise = (random_seed.next_f32() - 0.5) * 78.0 + macro_noise * 0.45;
        let heat_term = heat * 1.8;

        relief[i] = base + macro_base + noise + heat_term;
    }

    let mut macro_blend = vec![0.0_f32; WORLD_SIZE];
    let macro_radius = 1_i32;

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.58 + 0.1 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let mut sum = 0.0_f32;
            let mut wsum = 0.0_f32;
            for oy in -macro_radius..=macro_radius {
                for ox in -macro_radius..=macro_radius {
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let dist2 = (ox * ox + oy * oy) as f32;
                    let w = (-dist2 / 1.9).exp();
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            macro_blend[i] = sum / wsum.max(1e-6);
        }
    }

    for i in 0..WORLD_SIZE {
        relief[i] = relief[i] * 0.82 + macro_blend[i] * 0.18;
    }

    let erosion_rounds = detail.erosion_rounds;
    let mut smoothed = relief.clone();
    let mut scratch = vec![0.0_f32; WORLD_SIZE];

    for round in 0..erosion_rounds {
        for y in 0..WORLD_HEIGHT {
            let round_t =
                (round as f32 + y as f32 / WORLD_HEIGHT as f32) / erosion_rounds.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.68 + 0.16 * round_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = 0.0_f32;
                let mut count = 0.0_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        sum += smoothed[j];
                        count += 1.0;
                    }
                }
                let avg = sum / count;
                scratch[i] = smoothed[i] * 0.92 + avg * 0.08;
            }
        }

        for i in 0..WORLD_SIZE {
            let drop = (smoothed[i] - scratch[i]).max(0.0);
            let height_loss = (drop * 0.18).min(22.0);
            smoothed[i] = scratch[i] - height_loss;
        }
    }

    relief.copy_from_slice(&smoothed);

    let ocean_cut = (((planet.ocean_percent / 100.0) * WORLD_SIZE as f32).floor() as isize)
        .clamp(0, (WORLD_SIZE - 1) as isize) as usize;

    let mut sorted = relief.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sea_level = *sorted.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= sea_level;
    }

    apply_coastal_detail(&mut relief, seed, cache);
    cleanup_coastal_speckles(&mut relief);

    let mut sorted_after_coast = relief.clone();
    sorted_after_coast.sort_by(|a, b| a.total_cmp(b));
    let coast_recenter = *sorted_after_coast.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= coast_recenter;
    }

    normalize_height_range(&mut relief, planet, tectonics);
    progress.phase(progress_base, progress_span, 0.86);
    apply_ocean_profile(
        &mut relief,
        &plates.boundary_types,
        &plates.boundary_strength,
        planet,
        seed,
        cache,
        detail,
        progress,
        progress_base + progress_span * 0.86,
        progress_span * 0.1,
    );
    reshape_ocean_boundaries(
        &mut relief,
        &plates.boundary_types,
        &plates.boundary_strength,
    );
    progress.phase(progress_base, progress_span, 0.98);

    let mut sorted_final = relief.clone();
    sorted_final.sort_by(|a, b| a.total_cmp(b));
    let final_recenter = *sorted_final.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= final_recenter;
    }
    progress.phase(progress_base, progress_span, 1.0);

    ReliefResult { relief, sea_level }
}

fn apply_events(
    planet: &PlanetInputs,
    relief: &[f32],
    events: &[WorldEventRecord],
    cache: &WorldCache,
) -> (Vec<f32>, f32) {
    let mut updated = relief.to_vec();
    let mut aerosol_index = 0.0_f32;

    let nearest_cell = |lat: f32, lon: f32| -> usize {
        let y = clampf(
            ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
            0.0,
            (WORLD_HEIGHT - 1) as f32,
        ) as usize;
        let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32
            % WORLD_WIDTH as i32)
            + WORLD_WIDTH as i32)
            % WORLD_WIDTH as i32;
        index(x as usize, y)
    };

    let km_per_cell_lat = KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0;
    let km_per_cell_lon = |lat: f32, lon: f32| -> f32 {
        (KILOMETERS_PER_DEGREE * (lat * RADIANS).cos() * (lon * RADIANS).cos() * planet.radius_km
            / 6371.0)
            .max(1.0)
    };

    for event in events {
        if event.kind == "meteorite" {
            let radius_m = ((event.diameter_km * 1000.0) / 2.0).max(1.0);
            let mass = (4.0 / 3.0) * std::f32::consts::PI * radius_m.powi(3) * event.density_kg_m3;
            let energy = 0.5 * mass * (event.speed_kms * 1000.0).powi(2);
            let crater_radius_km = 8.0_f32.max(energy.powf(1.0 / 5.0) / 2500.0);
            let crater_depth = 9000.0_f32.min(800.0 + ((energy + 1.0).log10() - 10.0) * 250.0);
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;

            let lat_span = crater_radius_km / km_per_cell_lat;
            let lon_span = crater_radius_km / km_per_cell_lon(event.latitude, event.longitude);

            let min_x = (cx as f32 - lon_span - 1.0).floor() as i32;
            let max_x = (cx as f32 + lon_span + 1.0).ceil() as i32;
            let min_y = clampf(
                (cy as f32 - lat_span - 1.0).floor(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;
            let max_y = clampf(
                (cy as f32 + lat_span + 1.0).ceil(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let d_lat = cache.lat_by_y[target] - event.latitude;
                    let d_lon = (((cache.lon_by_x[target] - event.longitude + 180.0) % 360.0
                        + 360.0)
                        % 360.0)
                        - 180.0;
                    let d_km = ((d_lat * km_per_cell_lat).powi(2)
                        + (d_lon
                            * KILOMETERS_PER_DEGREE
                            * (cache.lat_by_y[target] * RADIANS).cos()
                            * planet.radius_km
                            / 6371.0)
                            .powi(2))
                    .sqrt();

                    if d_km <= crater_radius_km {
                        let falloff = 1.0 - (d_km / crater_radius_km).powi(2);
                        updated[target] -= crater_depth * falloff.max(0.0) * 0.5;
                        updated[target] = updated[target].max(-planet.radius_km * 10.0);
                    }
                }
            }

            aerosol_index += 0.45_f32.min((energy + 1.0).log10() / 18.0);
        } else if event.kind == "oceanShift" {
            let idx = nearest_cell(event.latitude, event.longitude);
            updated[idx] += event.magnitude * 0.5;
            for v in updated.iter_mut() {
                *v += event.magnitude * 0.15;
            }
        } else {
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;
            let sign = if event.kind == "rift" { -1.0 } else { 1.0 };
            let magnitude = event.magnitude * 40.0 * sign;
            let radius_cells = ((event.radius_km
                / (KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0))
                .round() as i32)
                .max(1);

            for y in (cy as i32 - radius_cells)..=(cy as i32 + radius_cells) {
                if y < 0 || y >= WORLD_HEIGHT as i32 {
                    continue;
                }
                for x in (cx as i32 - radius_cells)..=(cx as i32 + radius_cells) {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let dx = x - cx as i32;
                    let dy = y - cy as i32;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist <= radius_cells as f32 {
                        let falloff = (-dist / (radius_cells as f32).max(1.0)).exp();
                        updated[target] += magnitude * falloff;
                    }
                }
            }
        }
    }

    (updated, clampf(aerosol_index, 0.0, 1.0))
}

fn compute_slope(
    heights: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, f32, f32) {
    let mut slope = vec![0.0_f32; WORLD_SIZE];
    let mut min_slope = f32::INFINITY;
    let mut max_slope = 0.0_f32;

    for y in 0..WORLD_HEIGHT {
        progress.phase(progress_base, progress_span, y as f32 / WORLD_HEIGHT as f32);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut max_drop = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = (h - heights[j]).max(0.0);
                    if drop > max_drop {
                        max_drop = drop;
                    }
                }
            }
            slope[i] = max_drop;
            min_slope = min_slope.min(max_drop);
            max_slope = max_slope.max(max_drop);
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (slope, min_slope, max_slope)
}

fn compute_hydrology(
    heights: &[f32],
    slope: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut flow_direction = vec![-1_i32; WORLD_SIZE];
    let mut flow_accumulation = vec![1.0_f32; WORLD_SIZE];

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.45 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut best_drop = 0.0_f32;
            let mut best = -1_i32;

            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = h - heights[j];
                    if drop > best_drop {
                        best_drop = drop;
                        best = j as i32;
                    }
                }
            }

            flow_direction[i] = best;
        }
    }

    let mut order: Vec<usize> = (0..WORLD_SIZE).collect();
    order.sort_by(|a, b| heights[*b].total_cmp(&heights[*a]));
    progress.phase(progress_base, progress_span, 0.48);

    for (rank, i) in order.iter().copied().enumerate() {
        if rank % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.48 + 0.35 * (rank as f32 / WORLD_SIZE as f32),
            );
        }
        let to = flow_direction[i];
        if to >= 0 {
            flow_accumulation[to as usize] += flow_accumulation[i] * (1.0 + slope[i] / 1000.0);
        }
    }
    progress.phase(progress_base, progress_span, 0.83);

    let mut sorted = flow_accumulation.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold_index = ((sorted.len() as f32 * 0.985).floor() as usize).min(sorted.len() - 1);
    let threshold = sorted[threshold_index].max(1.0);

    let mut rivers = vec![0.0_f32; WORLD_SIZE];
    let mut lakes = vec![0_u8; WORLD_SIZE];

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.83 + 0.17 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        rivers[i] = flow_accumulation[i] / threshold;
        if flow_direction[i] < 0 && heights[i] < 0.0 && slope[i] < 40.0 {
            lakes[i] = 1;
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (flow_direction, flow_accumulation, rivers, lakes)
}

fn compute_climate(
    planet: &PlanetInputs,
    heights: &[f32],
    slope: &[f32],
    rivers: &[f32],
    flow: &[f32],
    aerosol: f32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    let mut temperature = vec![0.0_f32; WORLD_SIZE];
    let mut precipitation = vec![0.0_f32; WORLD_SIZE];
    let tilt_rad = planet.axial_tilt_deg * RADIANS;
    let rotation_factor = 20.0 / planet.rotation_hours.max(1.0);

    let mut min_temp = f32::INFINITY;
    let mut max_temp = -f32::INFINITY;
    let mut min_prec = f32::INFINITY;
    let mut max_prec = -f32::INFINITY;

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(progress_base, progress_span, i as f32 / WORLD_SIZE as f32);
        }
        let lat = cache.lat_by_y[i] * RADIANS;
        let base_insolation = (lat - tilt_rad).cos().max(0.0) * (1.0 + planet.eccentricity * 0.35);
        let elevation_for_temperature = heights[i].max(0.0);
        let base_temp =
            55.0 * base_insolation - 0.0065 * elevation_for_temperature - lat.abs() * 0.4;
        let pressure_term = 12.0 * planet.atmosphere_bar.max(0.0).ln_1p();
        let ocean_term = if heights[i] < 0.0 { 6.0 } else { 0.0 };
        let seasonal = 1.0 + 0.08 * ((lat + 0.2 * std::f32::consts::PI * tilt_rad) * 2.0).sin();
        let temp =
            base_temp + pressure_term + ocean_term + 0.12 * rotation_factor * 100.0 * seasonal
                - aerosol * 8.0;
        let slope_orographic = 1.0 + (slope[i].abs() / 400.0).min(1.4);
        let precip_raw = 600.0
            + 1800.0 * base_insolation * seasonal
            + 1200.0 * rivers[i].min(1.0) * slope_orographic
            + flow[i] * 0.1;
        let precip = (precip_raw - aerosol * 250.0).max(0.0);

        temperature[i] = clampf(temp, -90.0, 70.0);
        precipitation[i] = precip;

        min_temp = min_temp.min(temperature[i]);
        max_temp = max_temp.max(temperature[i]);
        min_prec = min_prec.min(precip);
        max_prec = max_prec.max(precip);
    }
    progress.phase(progress_base, progress_span, 1.0);

    (
        temperature,
        precipitation,
        min_temp,
        max_temp,
        min_prec,
        max_prec,
    )
}

#[inline]
fn classify_biome(temp: f32, precip: f32, height: f32) -> u8 {
    if height < 0.0 {
        return 0;
    }
    if temp < -5.0 {
        return 1;
    }
    if precip > 1600.0 && temp >= 5.0 && temp < 22.0 {
        return 2;
    }
    if temp > 12.0 && precip > 900.0 {
        return 3;
    }
    if temp > 24.0 && precip > 500.0 && precip <= 1400.0 {
        return 4;
    }
    if precip < 400.0 && temp > 18.0 {
        return 5;
    }
    if precip > 700.0 && temp > 5.0 && temp < 22.0 {
        return 6;
    }
    if height > 1800.0 {
        return 7;
    }
    8
}

fn compute_biomes(temperature: &[f32], precipitation: &[f32], heights: &[f32]) -> Vec<u8> {
    let mut biomes = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        biomes[i] = classify_biome(temperature[i], precipitation[i], heights[i]);
    }
    biomes
}

fn compute_settlement(
    biomes: &[u8],
    heights: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let mut settlement = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if biomes[i] == 0 {
            settlement[i] = 0.0;
            continue;
        }
        let comfort_t = 1.0 - (temperature[i] - 18.0).abs() / 45.0;
        let comfort_p = 1.0 - (precipitation[i] - 1400.0).abs() / 2200.0;
        let elevation_penalty = (heights[i] - 1200.0).max(0.0) / 2600.0;
        settlement[i] = ((comfort_t + comfort_p) / 2.0 - elevation_penalty).clamp(0.0, 1.0);
    }
    settlement
}

#[wasm_bindgen]
pub struct WasmSimulationResult {
    width: u32,
    height: u32,
    seed: u32,
    sea_level: f32,
    radius_km: f32,
    ocean_percent: f32,
    recomputed_layers: Vec<String>,
    plates: Vec<i16>,
    boundary_types: Vec<i8>,
    height_map: Vec<f32>,
    slope_map: Vec<f32>,
    river_map: Vec<f32>,
    lake_map: Vec<u8>,
    flow_direction: Vec<i32>,
    flow_accumulation: Vec<f32>,
    temperature_map: Vec<f32>,
    precipitation_map: Vec<f32>,
    biome_map: Vec<u8>,
    settlement_map: Vec<f32>,
    min_height: f32,
    max_height: f32,
    min_temperature: f32,
    max_temperature: f32,
    min_precipitation: f32,
    max_precipitation: f32,
    min_slope: f32,
    max_slope: f32,
}

#[wasm_bindgen]
impl WasmSimulationResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    #[wasm_bindgen(js_name = seaLevel)]
    pub fn sea_level(&self) -> f32 {
        self.sea_level
    }

    #[wasm_bindgen(js_name = radiusKm)]
    pub fn radius_km(&self) -> f32 {
        self.radius_km
    }

    #[wasm_bindgen(js_name = oceanPercent)]
    pub fn ocean_percent(&self) -> f32 {
        self.ocean_percent
    }

    #[wasm_bindgen(js_name = recomputedLayers)]
    pub fn recomputed_layers(&self) -> Array {
        self.recomputed_layers.iter().map(JsValue::from).collect()
    }

    pub fn plates(&self) -> Vec<i16> {
        self.plates.clone()
    }

    #[wasm_bindgen(js_name = boundaryTypes)]
    pub fn boundary_types(&self) -> Vec<i8> {
        self.boundary_types.clone()
    }

    #[wasm_bindgen(js_name = heightMap)]
    pub fn height_map(&self) -> Vec<f32> {
        self.height_map.clone()
    }

    #[wasm_bindgen(js_name = slopeMap)]
    pub fn slope_map(&self) -> Vec<f32> {
        self.slope_map.clone()
    }

    #[wasm_bindgen(js_name = riverMap)]
    pub fn river_map(&self) -> Vec<f32> {
        self.river_map.clone()
    }

    #[wasm_bindgen(js_name = lakeMap)]
    pub fn lake_map(&self) -> Vec<u8> {
        self.lake_map.clone()
    }

    #[wasm_bindgen(js_name = flowDirection)]
    pub fn flow_direction(&self) -> Vec<i32> {
        self.flow_direction.clone()
    }

    #[wasm_bindgen(js_name = flowAccumulation)]
    pub fn flow_accumulation(&self) -> Vec<f32> {
        self.flow_accumulation.clone()
    }

    #[wasm_bindgen(js_name = temperatureMap)]
    pub fn temperature_map(&self) -> Vec<f32> {
        self.temperature_map.clone()
    }

    #[wasm_bindgen(js_name = precipitationMap)]
    pub fn precipitation_map(&self) -> Vec<f32> {
        self.precipitation_map.clone()
    }

    #[wasm_bindgen(js_name = biomeMap)]
    pub fn biome_map(&self) -> Vec<u8> {
        self.biome_map.clone()
    }

    #[wasm_bindgen(js_name = settlementMap)]
    pub fn settlement_map(&self) -> Vec<f32> {
        self.settlement_map.clone()
    }

    #[wasm_bindgen(js_name = minHeight)]
    pub fn min_height(&self) -> f32 {
        self.min_height
    }

    #[wasm_bindgen(js_name = maxHeight)]
    pub fn max_height(&self) -> f32 {
        self.max_height
    }

    #[wasm_bindgen(js_name = minTemperature)]
    pub fn min_temperature(&self) -> f32 {
        self.min_temperature
    }

    #[wasm_bindgen(js_name = maxTemperature)]
    pub fn max_temperature(&self) -> f32 {
        self.max_temperature
    }

    #[wasm_bindgen(js_name = minPrecipitation)]
    pub fn min_precipitation(&self) -> f32 {
        self.min_precipitation
    }

    #[wasm_bindgen(js_name = maxPrecipitation)]
    pub fn max_precipitation(&self) -> f32 {
        self.max_precipitation
    }

    #[wasm_bindgen(js_name = minSlope)]
    pub fn min_slope(&self) -> f32 {
        self.min_slope
    }

    #[wasm_bindgen(js_name = maxSlope)]
    pub fn max_slope(&self) -> f32 {
        self.max_slope
    }
}

fn report_progress(progress_callback: Option<&js_sys::Function>, value: f32) {
    if let Some(cb) = progress_callback {
        let _ = cb.call1(
            &JsValue::NULL,
            &JsValue::from_f64(clampf(value, 0.0, 100.0) as f64),
        );
    }
}

struct ProgressTap<'a> {
    callback: Option<&'a js_sys::Function>,
    last: f32,
}

impl<'a> ProgressTap<'a> {
    fn new(callback: Option<&'a js_sys::Function>) -> Self {
        Self {
            callback,
            last: -1.0,
        }
    }

    fn emit(&mut self, value: f32) {
        let v = clampf(value, 0.0, 100.0);
        if v >= 100.0 || self.last < 0.0 || v - self.last >= 0.08 {
            report_progress(self.callback, v);
            self.last = v;
        }
    }

    fn phase(&mut self, base: f32, span: f32, t: f32) {
        self.emit(base + span * clampf(t, 0.0, 1.0));
    }
}

fn run_simulation_internal(
    config: JsValue,
    reason: String,
    progress_callback: Option<&js_sys::Function>,
) -> Result<WasmSimulationResult, JsValue> {
    let mut progress = ProgressTap::new(progress_callback);
    progress.emit(0.0);
    let mut cfg: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("config deserialize error: {e}")))?;
    progress.emit(1.0);

    cfg.events = cfg.events.into_iter().map(ensure_event_energy).collect();

    let cache = world_cache();
    let recomputed_layers = evaluate_recompute(parse_reason(&reason));
    let preset = parse_generation_preset(cfg.generation_preset.as_deref());
    let detail = detail_profile(preset);

    let plates_layer = compute_plates(
        &cfg.planet,
        &cfg.tectonics,
        cfg.seed,
        cache,
        &mut progress,
        2.0,
        22.0,
    );
    let relief_raw = compute_relief(
        &cfg.planet,
        &cfg.tectonics,
        &plates_layer,
        cfg.seed,
        cache,
        detail,
        &mut progress,
        24.0,
        50.0,
    );
    progress.emit(74.0);
    let (event_relief, aerosol) = apply_events(&cfg.planet, &relief_raw.relief, &cfg.events, cache);
    progress.emit(78.0);

    let (slope_map, min_slope, max_slope) = compute_slope(&event_relief, &mut progress, 78.0, 8.0);
    let (flow_direction, flow_accumulation, river_map, lake_map) =
        compute_hydrology(&event_relief, &slope_map, &mut progress, 86.0, 7.0);

    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate(
        &cfg.planet,
        &event_relief,
        &slope_map,
        &river_map,
        &flow_accumulation,
        aerosol,
        cache,
        &mut progress,
        93.0,
        5.0,
    );
    progress.emit(98.0);

    let biome_map = compute_biomes(&temperature_map, &precipitation_map, &event_relief);
    progress.emit(99.0);
    let settlement_map = compute_settlement(
        &biome_map,
        &event_relief,
        &temperature_map,
        &precipitation_map,
    );
    progress.emit(99.7);
    let (min_height, max_height) = min_max(&event_relief);
    let result = WasmSimulationResult {
        width: WORLD_WIDTH as u32,
        height: WORLD_HEIGHT as u32,
        seed: cfg.seed,
        sea_level: relief_raw.sea_level,
        radius_km: cfg.planet.radius_km,
        ocean_percent: cfg.planet.ocean_percent,
        recomputed_layers,
        plates: plates_layer.plate_field,
        boundary_types: plates_layer.boundary_types,
        height_map: event_relief,
        slope_map,
        river_map,
        lake_map,
        flow_direction,
        flow_accumulation,
        temperature_map,
        precipitation_map,
        biome_map,
        settlement_map,
        min_height,
        max_height,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
        min_slope,
        max_slope,
    };

    // Keep progress <100 until JS wrapper and worker finish marshalling + posting result.
    progress.emit(99.9);
    Ok(result)
}

#[wasm_bindgen]
pub fn run_simulation(config: JsValue, reason: String) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, None)
}

#[wasm_bindgen(js_name = run_simulation_with_progress)]
pub fn run_simulation_with_progress(
    config: JsValue,
    reason: String,
    progress_callback: js_sys::Function,
) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, Some(&progress_callback))
}
