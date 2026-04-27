use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const APPLE_COST: f64 = 2.0;
pub const BARRACUDA_COST: f64 = 5.0;
pub const CASH_COST: f64 = 1.0;
pub const LEARNING_RATE: f64 = 0.015;
pub const EPS: f64 = 1.0e-8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Good {
    Apple,
    Barracuda,
}

impl Good {
    pub const ALL: [Good; 2] = [Good::Apple, Good::Barracuda];

    pub fn name(self) -> &'static str {
        match self {
            Good::Apple => "apple",
            Good::Barracuda => "barracuda",
        }
    }

    pub fn inventory_index(self) -> usize {
        match self {
            Good::Apple => 0,
            Good::Barracuda => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    M,
    F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraitKey {
    Intelligence,
    Wisdom,
    Strength,
    Dexterity,
    Charisma,
    Comeliness,
    Constitution,
    MetabolicRate,
}

impl TraitKey {
    pub const ALL: [TraitKey; 8] = [
        TraitKey::Intelligence,
        TraitKey::Wisdom,
        TraitKey::Strength,
        TraitKey::Dexterity,
        TraitKey::Charisma,
        TraitKey::Comeliness,
        TraitKey::Constitution,
        TraitKey::MetabolicRate,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TraitKey::Intelligence => "intelligence",
            TraitKey::Wisdom => "wisdom",
            TraitKey::Strength => "strength",
            TraitKey::Dexterity => "dexterity",
            TraitKey::Charisma => "charisma",
            TraitKey::Comeliness => "comeliness",
            TraitKey::Constitution => "constitution",
            TraitKey::MetabolicRate => "metabolic_rate",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn random_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            x: rng.gen::<f64>(),
            y: rng.gen::<f64>(),
        }
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n <= EPS {
            Self::new(0.0, 0.0)
        } else {
            self.scale(1.0 / n)
        }
    }

    pub fn orthogonal(self) -> Self {
        Self::new(-self.y, self.x)
    }

    pub fn clamp01(self) -> Self {
        Self::new(self.x.clamp(0.0, 1.0), self.y.clamp(0.0, 1.0))
    }
}

#[derive(Clone, Debug)]
pub struct Genome {
    pub gender: Gender,
    pub intelligence: f64,
    pub wisdom: f64,
    pub strength: f64,
    pub dexterity: f64,
    pub charisma: f64,
    pub comeliness: f64,
    pub constitution: f64,
    pub metabolic_rate: f64,
}

impl Genome {
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let normal = Normal::new(100.0, 15.0).unwrap();
        Self {
            gender: if rng.gen_bool(0.5) { Gender::M } else { Gender::F },
            intelligence: normal.sample(rng),
            wisdom: 200.0 * rng.gen::<f64>(),
            strength: 200.0 * rng.gen::<f64>(),
            dexterity: 200.0 * rng.gen::<f64>(),
            charisma: 200.0 * rng.gen::<f64>(),
            comeliness: 200.0 * rng.gen::<f64>(),
            constitution: 200.0 * rng.gen::<f64>(),
            metabolic_rate: 200.0 * rng.gen::<f64>(),
        }
    }

    pub fn get(&self, key: TraitKey) -> f64 {
        match key {
            TraitKey::Intelligence => self.intelligence,
            TraitKey::Wisdom => self.wisdom,
            TraitKey::Strength => self.strength,
            TraitKey::Dexterity => self.dexterity,
            TraitKey::Charisma => self.charisma,
            TraitKey::Comeliness => self.comeliness,
            TraitKey::Constitution => self.constitution,
            TraitKey::MetabolicRate => self.metabolic_rate,
        }
    }

    pub fn set(&mut self, key: TraitKey, value: f64) {
        match key {
            TraitKey::Intelligence => self.intelligence = value,
            TraitKey::Wisdom => self.wisdom = value,
            TraitKey::Strength => self.strength = value,
            TraitKey::Dexterity => self.dexterity = value,
            TraitKey::Charisma => self.charisma = value,
            TraitKey::Comeliness => self.comeliness = value,
            TraitKey::Constitution => self.constitution = value,
            TraitKey::MetabolicRate => self.metabolic_rate = value,
        }
    }
}

pub fn dictionary_to_vector(genome: &Genome) -> [f64; 8] {
    [
        genome.intelligence,
        genome.wisdom,
        genome.strength,
        genome.dexterity,
        genome.charisma,
        genome.comeliness,
        genome.constitution,
        genome.metabolic_rate,
    ]
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

pub fn softmax(mut w: [f64; 3]) -> [f64; 3] {
    let max_w = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for val in &mut w {
        *val = (*val - max_w).exp();
    }
    let sum = w.iter().sum::<f64>() + EPS;
    [w[0] / sum, w[1] / sum, w[2] / sum]
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub birth_round: f64,
    pub position: Vec2,
    pub genome: Genome,
    pub radius: f64,
    pub relationships: HashMap<(usize, usize), f64>,
    pub genome_vector: [f64; 8],
    pub genome_vector_normalized: [f64; 8],
    pub energy: f64,
    pub income: f64,
    pub w: [f64; 3],
    pub true_cd_exponents: [f64; 3],
    pub expected_cd_exponents: [f64; 3],
    pub inventory: [f64; 3],
    pub saving_rate: f64,
    pub fitness: f64,
    pub x: usize,
    pub y: usize,
    pub married: bool,
    pub partner_id: Option<usize>,
}

impl Agent {
    pub fn new<R: Rng + ?Sized>(id: usize, birth_round: f64, rng: &mut R, grid_size: usize) -> Self {
        let position = Vec2::random_uniform(rng);
        let genome = Genome::random(rng);
        let radius = 0.02 + 0.08 * (genome.dexterity / 200.0);
        let genome_vector = dictionary_to_vector(&genome);
        let vector_norm = norm(&genome_vector) + EPS;
        let mut genome_vector_normalized = [0.0; 8];
        for i in 0..8 {
            genome_vector_normalized[i] = (genome_vector[i] - 1.0) / vector_norm;
        }
        let energy = rng.gen_range(0..3) as f64;
        let income = genome.intelligence * rng.gen_range(1..5) as f64;
        let w = [1.0, 1.0, 1.0];
        let true_cd_exponents = softmax([rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()]);
        let expected_cd_exponents = softmax(w);

        let x = ((position.x * grid_size as f64).floor() as usize).min(grid_size.saturating_sub(1));
        let y = ((position.y * grid_size as f64).floor() as usize).min(grid_size.saturating_sub(1));

        let mut agent = Self {
            id,
            birth_round,
            position,
            genome,
            radius,
            relationships: HashMap::new(),
            genome_vector,
            genome_vector_normalized,
            energy,
            income,
            w,
            true_cd_exponents,
            expected_cd_exponents,
            inventory: [0.0, 0.0, 0.0],
            saving_rate: rng.gen::<f64>(),
            fitness: 0.0,
            x,
            y,
            married: false,
            partner_id: None,
        };
        agent.fitness = agent.compute_fitness();
        agent.inventory = agent.random_allocation(rng);
        agent
    }

    pub fn get_cell(&self, grid_size: usize) -> (usize, usize) {
        let x = ((self.position.x * grid_size as f64).floor() as usize).min(grid_size.saturating_sub(1));
        let y = ((self.position.y * grid_size as f64).floor() as usize).min(grid_size.saturating_sub(1));
        (x, y)
    }

    pub fn compute_fitness(&self) -> f64 {
        let survival = self.genome.constitution * 0.4 + self.genome.strength * 0.3;
        let attractiveness = self.genome.charisma * 0.5 + self.genome.comeliness * 0.5;
        let efficiency = self.genome.strength / (self.genome.metabolic_rate + EPS);
        0.4 * survival + 0.4 * attractiveness + 0.2 * efficiency
    }

    pub fn log_true_utility(&self, bundle: [f64; 3]) -> f64 {
        let a = bundle[0].max(EPS);
        let b = bundle[1].max(EPS);
        let c = bundle[2].max(EPS);
        let alpha = self.true_cd_exponents[0];
        let beta = self.true_cd_exponents[1];
        let gamma = self.true_cd_exponents[2];
        alpha * a.ln() + beta * b.ln() + gamma * c.ln()
    }

    pub fn log_expected_utility(&self, bundle: [f64; 3]) -> f64 {
        let a = bundle[0].max(EPS);
        let b = bundle[1].max(EPS);
        let c = bundle[2].max(EPS);
        let alpha = self.expected_cd_exponents[0];
        let beta = self.expected_cd_exponents[1];
        let gamma = self.expected_cd_exponents[2];
        alpha * a.ln() + beta * b.ln() + gamma * c.ln()
    }

    pub fn update_beliefs(&mut self, bundle: [f64; 3]) {
        let logs = [bundle[0].ln(), bundle[1].ln(), bundle[2].ln()];
        let true_log = self.log_true_utility(bundle);
        let pred_log = self.log_expected_utility(bundle);
        let error = true_log - pred_log;
        let alpha_hat = self.expected_cd_exponents;
        for i in 0..3 {
            let gradient = error * alpha_hat[i] * (logs[i] - pred_log);
            self.w[i] += LEARNING_RATE * gradient;
        }
        self.expected_cd_exponents = softmax(self.w);
    }

    pub fn learning<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; 3] {
        let bundle_owned = self.random_allocation(rng);
        self.update_beliefs(bundle_owned);
        self.expected_cd_exponents
    }

    pub fn random_allocation<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        let apples_spread = 0.1 + 3.9 * rng.gen::<f64>();
        let apples_allocation = self.income / apples_spread;
        let barracudas_allocation = self.income - apples_allocation;
        let apples_owned = (apples_allocation / APPLE_COST).floor().max(1.0e-6);
        let barracudas_owned = (barracudas_allocation / BARRACUDA_COST).floor().max(1.0e-6);
        let cash_owned = (self.income - (apples_owned * APPLE_COST + barracudas_owned * BARRACUDA_COST)).max(1.0e-6);
        [apples_owned, barracudas_owned, cash_owned]
    }

    pub fn mrs(&self) -> (f64, f64) {
        let alpha = self.expected_cd_exponents[0];
        let beta = self.expected_cd_exponents[1];
        let gamma = self.expected_cd_exponents[2];
        let apples = self.inventory[0].max(EPS);
        let barracudas = self.inventory[1].max(EPS);
        let cash = self.inventory[2].max(EPS);
        let mrs_ac = (alpha / gamma) * (cash / apples);
        let mrs_bc = (beta / gamma) * (cash / barracudas);
        (mrs_ac, mrs_bc)
    }
}

#[derive(Clone, Debug)]
pub struct Relationship {
    pub a_id: usize,
    pub b_id: usize,
    pub key: (usize, usize),
    pub distance: f64,
    pub married: bool,
    pub children: usize,
    pub score: f64,
    pub friends: bool,
    pub transaction_cost: f64,
    pub food_surplus: f64,
}

impl Relationship {
    pub fn new<R: Rng + ?Sized>(a: &Agent, b: &Agent, rng: &mut R) -> Self {
        let key = if a.id < b.id { (a.id, b.id) } else { (b.id, a.id) };
        let distance = a.position.distance(b.position);
        let score = Self::rel_score(a, b, distance, 2.0, rng);
        let friends = score >= 0.7;
        Self {
            a_id: a.id,
            b_id: b.id,
            key,
            distance,
            married: false,
            children: 0,
            score,
            friends,
            transaction_cost: distance / 100.0,
            food_surplus: 0.0,
        }
    }

    pub fn rel_score<R: Rng + ?Sized>(a: &Agent, b: &Agent, distance: f64, alpha: f64, rng: &mut R) -> f64 {
        let similarity = dot(&a.genome_vector_normalized, &b.genome_vector_normalized);
        let fitness_score = 0.5 * (a.fitness + b.fitness);
        let noise = Normal::new(0.0, 0.15).unwrap().sample(rng);
        let distance_penalty = (-distance / (alpha * (a.radius + b.radius) + EPS)).exp();
        let raw = distance_penalty * (0.5 * similarity + 0.5 * fitness_score) + noise;
        raw.clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct History {
    pub population: Vec<usize>,
    pub food: Vec<f64>,
    pub traits: BTreeMap<String, Vec<f64>>,
}

#[derive(Clone, Debug, Default)]
pub struct MarketSeries {
    pub prices: Vec<Vec<f64>>,
    pub demand_curves: Vec<Vec<f64>>,
    pub supply_curves: Vec<Vec<f64>>,
    pub executed_prices: Vec<f64>,
    pub executed_qty: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct MarketHistory {
    pub apple: MarketSeries,
    pub barracuda: MarketSeries,
}

impl MarketHistory {
    pub fn series(&self, good: Good) -> &MarketSeries {
        match good {
            Good::Apple => &self.apple,
            Good::Barracuda => &self.barracuda,
        }
    }

    pub fn series_mut(&mut self, good: Good) -> &mut MarketSeries {
        match good {
            Good::Apple => &mut self.apple,
            Good::Barracuda => &mut self.barracuda,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PriceHistory {
    pub apple: Vec<f64>,
    pub barracuda: Vec<f64>,
}

impl PriceHistory {
    pub fn get(&self, good: Good) -> &Vec<f64> {
        match good {
            Good::Apple => &self.apple,
            Good::Barracuda => &self.barracuda,
        }
    }

    pub fn get_mut(&mut self, good: Good) -> &mut Vec<f64> {
        match good {
            Good::Apple => &mut self.apple,
            Good::Barracuda => &mut self.barracuda,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrderRecord {
    pub good: Good,
    pub buyer_id: usize,
    pub seller_id: usize,
    pub mid_price: f64,
    pub buyer_price: f64,
    pub seller_price: f64,
    pub quantity: f64,
    pub t_cost: f64,
    pub buyer_cash: f64,
    pub seller_inventory: f64,
    pub buyer_mrs: f64,
    pub seller_mrs: f64,
    pub success: bool,
    pub reason: String,
    pub round: Option<usize>,
}

impl OrderRecord {
    pub fn invalid(good: Good, reason: &str) -> Self {
        Self {
            good,
            buyer_id: 0,
            seller_id: 0,
            mid_price: 0.0,
            buyer_price: 0.0,
            seller_price: 0.0,
            quantity: 0.0,
            t_cost: 0.0,
            buyer_cash: 0.0,
            seller_inventory: 0.0,
            buyer_mrs: 0.0,
            seller_mrs: 0.0,
            success: false,
            reason: reason.to_string(),
            round: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OrderHistory {
    pub apple: Vec<OrderRecord>,
    pub barracuda: Vec<OrderRecord>,
}

impl OrderHistory {
    pub fn get(&self, good: Good) -> &Vec<OrderRecord> {
        match good {
            Good::Apple => &self.apple,
            Good::Barracuda => &self.barracuda,
        }
    }

    pub fn get_mut(&mut self, good: Good) -> &mut Vec<OrderRecord> {
        match good {
            Good::Apple => &mut self.apple,
            Good::Barracuda => &mut self.barracuda,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub positions: Vec<Vec2>,
    pub birth_rounds: Vec<f64>,
    pub energies: Vec<f64>,
    pub ids: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ResourceGrid {
    pub apple: Vec<Vec<f64>>,
    pub barracuda: Vec<Vec<f64>>,
}

impl ResourceGrid {
    pub fn size(&self) -> usize {
        self.apple.len()
    }

    pub fn get(&self, good: Good, x: usize, y: usize) -> f64 {
        match good {
            Good::Apple => self.apple[x][y],
            Good::Barracuda => self.barracuda[x][y],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub num_agents: usize,
    pub total_rounds: usize,
    pub grid_size: usize,
    pub gaussian_sigma: f64,
    pub delta: f64,
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            num_agents: 1000,
            total_rounds: 20,
            grid_size: 50,
            gaussian_sigma: 5.0,
            delta: 0.1,
            seed: 67,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub agents: Vec<Agent>,
    pub history: History,
    pub market_history: MarketHistory,
    pub order_history: OrderHistory,
    pub snapshot_history: Vec<Snapshot>,
    pub resource_grid: ResourceGrid,
}

pub fn initialize_tracker() -> History {
    let mut traits = BTreeMap::new();
    for key in TraitKey::ALL {
        traits.insert(key.name().to_string(), Vec::new());
    }
    History {
        population: Vec::new(),
        food: Vec::new(),
        traits,
    }
}

pub fn normalize_shifted(grid: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for row in grid {
        for &v in row {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    let denom = (max_v - min_v).max(EPS);
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&v| 0.5 + (v - min_v) / denom)
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as isize;
    let mut kernel = Vec::new();
    let mut sum = 0.0;
    for i in -radius..=radius {
        let x = i as f64;
        let value = (-(x * x) / (2.0 * sigma * sigma + EPS)).exp();
        kernel.push(value);
        sum += value;
    }
    for v in &mut kernel {
        *v /= sum.max(EPS);
    }
    kernel
}

pub fn gaussian_filter(grid: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
    let n = grid.len();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = (kernel.len() as isize - 1) / 2;

    let mut tmp = vec![vec![0.0; n]; n];
    for x in 0..n {
        for y in 0..n {
            let mut acc = 0.0;
            for (k_idx, &k_val) in kernel.iter().enumerate() {
                let dy = k_idx as isize - radius;
                let yy = ((y as isize + dy).clamp(0, (n - 1) as isize)) as usize;
                acc += grid[x][yy] * k_val;
            }
            tmp[x][y] = acc;
        }
    }

    let mut out = vec![vec![0.0; n]; n];
    for x in 0..n {
        for y in 0..n {
            let mut acc = 0.0;
            for (k_idx, &k_val) in kernel.iter().enumerate() {
                let dx = k_idx as isize - radius;
                let xx = ((x as isize + dx).clamp(0, (n - 1) as isize)) as usize;
                acc += tmp[xx][y] * k_val;
            }
            out[x][y] = acc;
        }
    }
    out
}

pub fn make_resource_grid<R: Rng + ?Sized>(grid_size: usize, sigma: f64, rng: &mut R) -> ResourceGrid {
    let mut raw_apple = vec![vec![0.0; grid_size]; grid_size];
    for x in 0..grid_size {
        for y in 0..grid_size {
            raw_apple[x][y] = rng.gen::<f64>();
        }
    }
    let raw_barracuda = raw_apple
        .iter()
        .map(|row| row.iter().map(|&v| 1.0 - v).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    ResourceGrid {
        apple: normalize_shifted(&gaussian_filter(&raw_apple, sigma)),
        barracuda: normalize_shifted(&gaussian_filter(&raw_barracuda, sigma)),
    }
}

pub fn first_generation<R: Rng + ?Sized>(num_agents: usize, grid_size: usize, rng: &mut R) -> Vec<Agent> {
    let mut agents = Vec::with_capacity(num_agents);
    for i in 0..num_agents {
        let mut agent = Agent::new(i, 0.0, rng, grid_size);
        agent.birth_round = 0.0;
        agents.push(agent);
    }
    agents
}

pub fn agent_learning<R: Rng + ?Sized>(agents: &mut [Agent], rng: &mut R) {
    for agent in agents {
        let _ = agent.learning(rng);
    }
}

pub fn choose_trade_good(a: &Agent, b: &Agent) -> Good {
    let (a_apple, a_barracuda) = a.mrs();
    let (b_apple, b_barracuda) = b.mrs();
    let gap_apple = (a_apple - b_apple).abs();
    let gap_barracuda = (a_barracuda - b_barracuda).abs();
    if gap_apple >= gap_barracuda {
        Good::Apple
    } else {
        Good::Barracuda
    }
}

pub fn compute_supply_demand_curves(agents: &[Agent], good: Good, price_grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let i = good.inventory_index();
    let mut demand = Vec::with_capacity(price_grid.len());
    let mut supply = Vec::with_capacity(price_grid.len());

    for &p in price_grid {
        let mut total_demand = 0.0;
        let mut total_supply = 0.0;
        for agent in agents {
            let mrs = match good {
                Good::Apple => agent.mrs().0,
                Good::Barracuda => agent.mrs().1,
            };
            let inventory_good = agent.inventory[i].max(0.0);
            let cash = agent.inventory[2].max(0.0);

            if mrs > p {
                let desired_qty = 0.25 * (mrs - p);
                let affordable_qty = cash / (p + EPS);
                total_demand += desired_qty.min(affordable_qty).max(0.0);
            } else if mrs < p {
                let desired_qty = 0.25 * (p - mrs);
                total_supply += desired_qty.min(inventory_good).max(0.0);
            }
        }
        demand.push(total_demand);
        supply.push(total_supply);
    }
    (demand, supply)
}

pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (q / 100.0).clamp(0.0, 1.0) * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

pub fn record_market_state(agents: &[Agent], market_history: &mut MarketHistory) {
    for good in Good::ALL {
        let mut mrs_vals = Vec::new();
        for agent in agents {
            let mrs = match good {
                Good::Apple => agent.mrs().0,
                Good::Barracuda => agent.mrs().1,
            };
            if mrs.is_finite() {
                mrs_vals.push(mrs);
            }
        }

        let price_grid = if mrs_vals.is_empty() {
            linspace(0.01, 10.0, 50)
        } else {
            let low = percentile(&mrs_vals, 5.0).max(0.01);
            let high = percentile(&mrs_vals, 95.0).max(low + 1.0e-3);
            linspace(low, high, 50)
        };

        let (demand, supply) = compute_supply_demand_curves(agents, good, &price_grid);
        let series = market_history.series_mut(good);
        series.prices.push(price_grid);
        series.demand_curves.push(demand);
        series.supply_curves.push(supply);
    }
}

pub fn linspace(start: f64, end: f64, bins: usize) -> Vec<f64> {
    if bins <= 1 {
        return vec![start];
    }
    let step = (end - start) / (bins as f64 - 1.0);
    (0..bins).map(|i| start + i as f64 * step).collect()
}

pub fn consume_from_inventory(agent: &mut Agent) -> (f64, f64) {
    let apple_energy = 1.0;
    let fish_energy = 2.5;
    let need = (agent.genome.metabolic_rate / 100.0).powf(1.5);
    let apples_available = agent.inventory[0].max(0.0);
    let fish_available = agent.inventory[1].max(0.0);
    let apples_eaten = apples_available.min(need / apple_energy);
    let mut energy_gained = apples_eaten * apple_energy;
    let remaining_need = (need - energy_gained).max(0.0);
    let fish_eaten = fish_available.min(remaining_need / fish_energy);
    energy_gained += fish_eaten * fish_energy;
    agent.inventory[0] -= apples_eaten;
    agent.inventory[1] -= fish_eaten;
    let surplus = (agent.inventory[0] + agent.inventory[1]).max(0.0);
    (energy_gained, surplus)
}

pub fn consume_from_inventory_dt(agent: &mut Agent, dt: f64) -> (f64, f64) {
    let apple_energy = 1.0;
    let fish_energy = 2.5;
    let need = (agent.genome.metabolic_rate / 100.0).powf(1.5) * dt;
    let apples_available = agent.inventory[0].max(0.0);
    let fish_available = agent.inventory[1].max(0.0);
    let apples_eaten = apples_available.min(need / apple_energy);
    let mut energy_gained = apples_eaten * apple_energy;
    let remaining_need = (need - energy_gained).max(0.0);
    let fish_eaten = fish_available.min(remaining_need / fish_energy);
    energy_gained += fish_eaten * fish_energy;
    agent.inventory[0] -= apples_eaten;
    agent.inventory[1] -= fish_eaten;
    let surplus = (agent.inventory[0] + agent.inventory[1]).max(0.0);
    (energy_gained, surplus)
}

pub fn determine_price(good: Good, mrs_1: f64, mrs_2: f64, price_history: &PriceHistory) -> f64 {
    let reservation_price = (mrs_1 + mrs_2) / 2.0;
    let prices = price_history.get(good);
    let n = prices.len();
    if n == 0 {
        reservation_price
    } else {
        let last_price = prices[n - 1];
        last_price + (1.0 / (n as f64 + 1.0)) * (reservation_price - last_price)
    }
}

pub fn total_food_inventory(agents: &[Agent]) -> f64 {
    agents.iter().map(|a| a.inventory[0] + a.inventory[1]).sum()
}

pub fn produce(agent: &mut Agent, resource_grid: &ResourceGrid) {
    let apple_yield = resource_grid.apple[agent.x][agent.y];
    let fish_yield = resource_grid.barracuda[agent.x][agent.y];
    let strength = agent.genome.strength / 100.0;
    let metabolism = agent.genome.metabolic_rate / 100.0;
    let apples = strength * apple_yield / (metabolism + EPS);
    let fish = strength * fish_yield / (metabolism + EPS);
    agent.inventory[0] += apples;
    agent.inventory[1] += fish;
}

pub fn produce_dt(agent: &mut Agent, resource_grid: &ResourceGrid, dt: f64) {
    let apple_yield = resource_grid.apple[agent.x][agent.y];
    let fish_yield = resource_grid.barracuda[agent.x][agent.y];
    let strength = agent.genome.strength / 100.0;
    let metabolism = agent.genome.metabolic_rate / 100.0;
    let apples = dt * strength * apple_yield / (metabolism + EPS);
    let fish = dt * strength * fish_yield / (metabolism + EPS);
    agent.inventory[0] += apples;
    agent.inventory[1] += fish;
}

pub fn build_relationships_bruteforce<R: Rng + ?Sized>(agents: &[Agent], rng: &mut R) -> HashMap<(usize, usize), Relationship> {
    let mut relationships = HashMap::new();
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let a = &agents[i];
            let b = &agents[j];
            let dx = a.position.x - b.position.x;
            let dy = a.position.y - b.position.y;
            let dist2 = dx * dx + dy * dy;
            let distance = dist2.sqrt();
            let r_sum = a.radius + b.radius;
            if distance <= 0.1 && dist2 <= r_sum * r_sum {
                let rel = Relationship::new(a, b, rng);
                relationships.insert(rel.key, rel);
            }
        }
    }
    relationships
}

pub fn build_friendships(relationships: &HashMap<(usize, usize), Relationship>) -> Vec<Relationship> {
    relationships.values().filter(|r| r.friends).cloned().collect()
}

pub fn id_to_index_map(agents: &[Agent]) -> HashMap<usize, usize> {
    let mut map = HashMap::with_capacity(agents.len());
    for (idx, agent) in agents.iter().enumerate() {
        map.insert(agent.id, idx);
    }
    map
}

pub fn two_mut<T>(items: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j, "indices must be distinct");
    if i < j {
        let (left, right) = items.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = items.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

pub fn build_marriages(agents: &mut [Agent], friendships: &[Relationship]) -> Vec<Relationship> {
    let mut sorted = friendships.to_vec();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut marriages = Vec::new();
    let map = id_to_index_map(agents);

    for mut rel in sorted {
        let Some(&ia) = map.get(&rel.a_id) else { continue; };
        let Some(&ib) = map.get(&rel.b_id) else { continue; };
        if ia == ib {
            continue;
        }
        let (a, b) = two_mut(agents, ia, ib);
        if a.genome.gender == b.genome.gender {
            continue;
        }
        if !a.married && !b.married && (a.birth_round - b.birth_round).abs() <= 2.0 {
            rel.married = true;
            a.married = true;
            b.married = true;
            a.partner_id = Some(b.id);
            b.partner_id = Some(a.id);
            marriages.push(rel);
        }
    }
    marriages
}

pub fn select_trade_pairs<R: Rng + ?Sized>(relationships: &HashMap<(usize, usize), Relationship>, rng: &mut R) -> Vec<Relationship> {
    let mut rels = relationships.values().cloned().collect::<Vec<_>>();
    rels.shuffle(rng);
    rels.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut used = HashSet::new();
    let mut trade_pairs = Vec::new();
    for rel in rels {
        if used.contains(&rel.a_id) || used.contains(&rel.b_id) {
            continue;
        }
        used.insert(rel.a_id);
        used.insert(rel.b_id);
        trade_pairs.push(rel);
    }
    trade_pairs
}

pub fn assign_food_to_relationships(
    relationships: &mut HashMap<(usize, usize), Relationship>,
    food_balance: &HashMap<usize, f64>,
) {
    for rel in relationships.values_mut() {
        let surplus_a = *food_balance.get(&rel.a_id).unwrap_or(&0.0);
        let surplus_b = *food_balance.get(&rel.b_id).unwrap_or(&0.0);
        rel.food_surplus = 0.5 * rel.food_surplus + (surplus_a + surplus_b).max(0.0);
    }
}

pub fn trade(
    agent_1: &mut Agent,
    agent_2: &mut Agent,
    good: Good,
    delta: f64,
    price_history: &mut PriceHistory,
    t_cost: f64,
) -> OrderRecord {
    let i = good.inventory_index();
    let (a_apple, a_barracuda) = agent_1.mrs();
    let (b_apple, b_barracuda) = agent_2.mrs();
    let mrs_1 = match good {
        Good::Apple => a_apple,
        Good::Barracuda => a_barracuda,
    };
    let mrs_2 = match good {
        Good::Apple => b_apple,
        Good::Barracuda => b_barracuda,
    };

    if (mrs_1 - mrs_2).abs() < 1.0e-8 {
        return OrderRecord::invalid(good, "no_mrs_gap");
    }

    let (buyer, seller) = if mrs_1 > mrs_2 {
        (agent_1, agent_2)
    } else {
        (agent_2, agent_1)
    };

    let buyer_mrs = match good {
        Good::Apple => buyer.mrs().0,
        Good::Barracuda => buyer.mrs().1,
    };
    let seller_mrs = match good {
        Good::Apple => seller.mrs().0,
        Good::Barracuda => seller.mrs().1,
    };

    let price = determine_price(good, mrs_1, mrs_2, price_history);
    let buyer_price = price + t_cost;
    let seller_price = price - t_cost;
    let mrs_gap = (mrs_1 - mrs_2).abs();
    let proposed_qty = (delta * mrs_gap).min(seller.inventory[i] * 0.25);

    let mut trade_record = OrderRecord {
        good,
        buyer_id: buyer.id,
        seller_id: seller.id,
        mid_price: price,
        buyer_price,
        seller_price,
        quantity: proposed_qty,
        t_cost,
        buyer_cash: buyer.inventory[2],
        seller_inventory: seller.inventory[i],
        buyer_mrs,
        seller_mrs,
        success: false,
        reason: String::new(),
        round: None,
    };

    if proposed_qty <= 1.0e-6 {
        trade_record.reason = "qty_too_small".to_string();
        return trade_record;
    }

    let cash_trade = buyer_price * proposed_qty;
    let seller_revenue = seller_price * proposed_qty;

    if seller.inventory[i] < proposed_qty {
        trade_record.reason = "seller_lacks_inventory".to_string();
        return trade_record;
    }
    if buyer.inventory[2] < cash_trade {
        trade_record.reason = "buyer_lacks_cash".to_string();
        return trade_record;
    }

    let mut proposed_buyer_inventory = buyer.inventory;
    let mut proposed_seller_inventory = seller.inventory;
    proposed_buyer_inventory[i] += proposed_qty;
    proposed_buyer_inventory[2] -= cash_trade;
    proposed_seller_inventory[i] -= proposed_qty;
    proposed_seller_inventory[2] += seller_revenue;

    if proposed_buyer_inventory.iter().any(|&x| x <= 0.0) || proposed_seller_inventory.iter().any(|&x| x <= 0.0) {
        trade_record.reason = "nonpositive_inventory".to_string();
        return trade_record;
    }

    let u_buyer_old = buyer.log_true_utility(buyer.inventory);
    let u_seller_old = seller.log_true_utility(seller.inventory);
    let u_buyer_new = buyer.log_expected_utility(proposed_buyer_inventory);
    let u_seller_new = seller.log_expected_utility(proposed_seller_inventory);

    if !(u_buyer_new > u_buyer_old) {
        trade_record.reason = "buyer_utility_fail".to_string();
        return trade_record;
    }
    if !(u_seller_new > u_seller_old) {
        trade_record.reason = "seller_utility_fail".to_string();
        return trade_record;
    }

    buyer.inventory = proposed_buyer_inventory;
    seller.inventory = proposed_seller_inventory;
    price_history.get_mut(good).push(price);
    trade_record.success = true;
    trade_record.reason = "executed".to_string();
    trade_record
}

pub fn bilateral_market(
    market_pairs: &[Relationship],
    agents: &mut [Agent],
    delta: f64,
    price_history: &mut PriceHistory,
) -> (usize, Vec<OrderRecord>) {
    let mut trades_completed = 0usize;
    let mut trade_log = Vec::with_capacity(market_pairs.len());
    let map = id_to_index_map(agents);

    for rel in market_pairs {
        let Some(&ia) = map.get(&rel.a_id) else { continue; };
        let Some(&ib) = map.get(&rel.b_id) else { continue; };
        if ia == ib {
            continue;
        }
        let good = {
            let a = &agents[ia];
            let b = &agents[ib];
            choose_trade_good(a, b)
        };
        let (a, b) = two_mut(agents, ia, ib);
        let result = trade(a, b, good, delta, price_history, rel.distance);
        if result.success {
            trades_completed += 1;
        }
        trade_log.push(result);
    }
    (trades_completed, trade_log)
}

pub fn record_order_flow(trade_log: &[OrderRecord], order_history: &mut OrderHistory, current_round: usize) {
    for trade in trade_log {
        let mut record = trade.clone();
        record.round = Some(current_round);
        order_history.get_mut(record.good).push(record);
    }
}

pub fn record_trade_log(trade_log: &[OrderRecord], market_history: &mut MarketHistory) {
    for good in Good::ALL {
        let good_trades = trade_log.iter().filter(|t| t.good == good && t.success).collect::<Vec<_>>();
        let series = market_history.series_mut(good);
        if good_trades.is_empty() {
            series.executed_prices.push(f64::NAN);
            series.executed_qty.push(0.0);
        } else {
            let total_qty = good_trades.iter().map(|t| t.quantity).sum::<f64>();
            let weighted_price = good_trades.iter().map(|t| t.mid_price * t.quantity).sum::<f64>() / (total_qty + EPS);
            series.executed_prices.push(weighted_price);
            series.executed_qty.push(total_qty);
        }
    }
}

pub fn record_population_state(agents: &[Agent], history: &mut History, food: f64) {
    history.population.push(agents.len());
    history.food.push(food);
    for key in TraitKey::ALL {
        let values = agents.iter().map(|a| a.genome.get(key)).collect::<Vec<_>>();
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        history.traits.get_mut(key.name()).unwrap().push(mean);
    }
}

pub fn record_population_snapshot(agents: &[Agent], snapshot_history: &mut Vec<Snapshot>) {
    let snapshot = Snapshot {
        positions: agents.iter().map(|a| a.position).collect(),
        birth_rounds: agents.iter().map(|a| a.birth_round).collect(),
        energies: agents.iter().map(|a| a.energy).collect(),
        ids: agents.iter().map(|a| a.id).collect(),
    };
    snapshot_history.push(snapshot);
}

pub fn survives(agent: &mut Agent, current_round: usize, consumed_food: f64) -> bool {
    agent.energy += consumed_food;
    let metabolic_cost = (agent.genome.metabolic_rate / 100.0).powf(1.5);
    agent.energy -= metabolic_cost;
    let age = (current_round as f64 - agent.birth_round).max(0.0);
    let age_cost = 0.01 * age;
    agent.energy -= age_cost;
    let constitution = agent.genome.constitution / 100.0;
    agent.energy += 0.2 * constitution * metabolic_cost;
    agent.energy > 0.0
}

pub fn survives_continuous(agent: &mut Agent, current_time: f64, consumed_food: f64, dt: f64) -> bool {
    agent.energy += consumed_food;
    let metabolic_cost = (agent.genome.metabolic_rate / 100.0).powf(1.5) * dt;
    agent.energy -= metabolic_cost;
    let age = (current_time - agent.birth_round).max(0.0);
    let age_cost = 0.01 * age * dt;
    agent.energy -= age_cost;
    let constitution = agent.genome.constitution / 100.0;
    agent.energy += 0.2 * constitution * metabolic_cost;
    agent.energy > 0.0
}

pub fn reproduce<R: Rng + ?Sized>(
    marriages: &[Relationship],
    agents: &mut [Agent],
    next_id_start: usize,
    current_round: f64,
    grid_size: usize,
    rng: &mut R,
) -> (Vec<Agent>, usize) {
    let mut children = Vec::new();
    let mut next_id = next_id_start;
    let map = id_to_index_map(agents);

    for marriage in marriages {
        let Some(&ia) = map.get(&marriage.a_id) else { continue; };
        let Some(&ib) = map.get(&marriage.b_id) else { continue; };
        if ia == ib {
            continue;
        }

        let (parent_a, parent_b) = two_mut(agents, ia, ib);
        let surplus = marriage.food_surplus;
        let reproduction_cost_a = 0.5 * (parent_a.genome.metabolic_rate / 100.0);
        let reproduction_cost_b = 0.5 * (parent_b.genome.metabolic_rate / 100.0);

        if parent_a.energy > reproduction_cost_a && parent_b.energy > reproduction_cost_b {
            parent_a.energy -= reproduction_cost_a / 2.0;
            parent_b.energy -= reproduction_cost_b / 2.0;
            let avg_fitness = (parent_a.fitness + parent_b.fitness) / 2.0;
            let fertility_modifier = 0.2 * surplus.tanh() + 0.7 * (avg_fitness / 100.0).tanh();
            if rng.gen::<f64>() > fertility_modifier {
                continue;
            }

            let surplus_a = (parent_a.energy - 1.0).max(0.0);
            let surplus_b = (parent_b.energy - 1.0).max(0.0);
            let available_energy = surplus_a + surplus_b;
            let avg_cost = (reproduction_cost_a + reproduction_cost_b) / 2.0;
            let num_children = (available_energy / (avg_cost + EPS)).floor() as usize;
            let energy_per_child = available_energy / (num_children as f64 + EPS);

            for _ in 0..num_children {
                let mut child = Agent::new(next_id, current_round, rng, grid_size);
                next_id += 1;
                for trait_key in TraitKey::ALL {
                    let p = rng.gen::<f64>();
                    let mut value = parent_a.genome.get(trait_key) * p + parent_b.genome.get(trait_key) * (1.0 - p);
                    value += Normal::new(0.0, (0.1 * value.abs()).max(EPS)).unwrap().sample(rng);
                    child.genome.set(trait_key, value.max(0.0));
                }
                let direction = parent_b.position.minus(parent_a.position);
                let orthogonal = direction.orthogonal().normalized();
                let t = rng.gen::<f64>();
                let base_pos = parent_a.position.plus(direction.scale(t));
                let offset = Normal::new(0.0, 0.02).unwrap().sample(rng);
                child.position = base_pos.plus(orthogonal.scale(offset)).clamp01();
                child.radius = child.genome.dexterity / 200.0;
                child.genome_vector = dictionary_to_vector(&child.genome);
                let vector_norm = norm(&child.genome_vector) + EPS;
                for i in 0..8 {
                    child.genome_vector_normalized[i] = child.genome_vector[i] / vector_norm;
                }
                child.energy = 0.5 * energy_per_child;
                child.inventory = [0.0, 0.0, 1.0];
                child.married = false;
                child.partner_id = None;
                children.push(child);
            }
        }
    }

    (children, next_id)
}

pub fn random_mating<R: Rng + ?Sized>(
    agents: &mut [Agent],
    next_id: usize,
    current_round: f64,
    grid_size: usize,
    rng: &mut R,
    rate: f64,
    marriage_penalty: f64,
) -> (Vec<Agent>, usize) {
    let mut children = Vec::new();
    let mut next_id_local = next_id;
    let n = agents.len();
    let num_attempts = (rate * n as f64).floor() as usize;

    for _ in 0..num_attempts {
        if n < 2 {
            break;
        }
        let ia = rng.gen_range(0..n);
        let mut ib = rng.gen_range(0..(n - 1));
        if ib >= ia {
            ib += 1;
        }
        let (parent_a, parent_b) = two_mut(agents, ia, ib);

        if parent_a.genome.gender == parent_b.genome.gender {
            continue;
        }
        let prob_a = if parent_a.married { marriage_penalty } else { 1.0 };
        let prob_b = if parent_b.married { marriage_penalty } else { 1.0 };
        if rng.gen::<f64>() > prob_a || rng.gen::<f64>() > prob_b {
            continue;
        }

        let rel = Relationship::new(parent_a, parent_b, rng);
        if rel.distance > 0.1 {
            continue;
        }

        let fitness_prob = ((parent_a.fitness + parent_b.fitness) / 200.0).tanh();
        if rng.gen::<f64>() > fitness_prob {
            continue;
        }

        let reproduction_cost_a = 0.5 * (parent_a.genome.metabolic_rate / 100.0);
        let reproduction_cost_b = 0.5 * (parent_b.genome.metabolic_rate / 100.0);
        if parent_a.energy <= reproduction_cost_a || parent_b.energy <= reproduction_cost_b {
            continue;
        }

        let surplus_a = (parent_a.energy - 1.0).max(0.0);
        let surplus_b = (parent_b.energy - 1.0).max(0.0);
        let available_energy = surplus_a + surplus_b;
        let avg_cost = (reproduction_cost_a + reproduction_cost_b) / 2.0;
        let num_children = (available_energy / (avg_cost + EPS)).floor() as usize;
        if num_children == 0 {
            continue;
        }

        parent_a.energy -= reproduction_cost_a / 2.0;
        parent_b.energy -= reproduction_cost_b / 2.0;
        let energy_per_child = available_energy / (num_children as f64 + EPS);

        for _ in 0..num_children {
            let mut child = Agent::new(next_id_local, current_round, rng, grid_size);
            next_id_local += 1;
            for trait_key in TraitKey::ALL {
                let p = rng.gen::<f64>();
                let mut value = parent_a.genome.get(trait_key) * p + parent_b.genome.get(trait_key) * (1.0 - p);
                value += Normal::new(0.0, (0.1 * value.abs()).max(EPS)).unwrap().sample(rng);
                child.genome.set(trait_key, value.max(0.0));
            }
            let direction = parent_b.position.minus(parent_a.position);
            let orthogonal = direction.orthogonal().normalized();
            let t = rng.gen::<f64>();
            let base_pos = parent_a.position.plus(direction.scale(t));
            let offset = Normal::new(0.0, 0.02).unwrap().sample(rng);
            child.position = base_pos.plus(orthogonal.scale(offset)).clamp01();
            child.radius = child.genome.dexterity / 200.0;
            child.genome_vector = dictionary_to_vector(&child.genome);
            let vector_norm = norm(&child.genome_vector) + EPS;
            for i in 0..8 {
                child.genome_vector_normalized[i] = child.genome_vector[i] / vector_norm;
            }
            child.inventory = [0.0, 0.0, 1.0];
            child.energy = 0.5 * energy_per_child;
            child.married = false;
            child.partner_id = None;
            children.push(child);
        }
    }

    (children, next_id_local)
}

pub fn build_empirical_supply_demand(
    order_history: &OrderHistory,
    good: Good,
    round_idx: Option<usize>,
    bins: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    build_empirical_supply_demand_filtered(order_history, good, round_idx, bins, None)
}

pub fn build_empirical_supply_demand_filtered(
    order_history: &OrderHistory,
    good: Good,
    round_idx: Option<usize>,
    bins: usize,
    include_reasons: Option<&[&str]>,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut records = order_history.get(good).clone();
    if let Some(round) = round_idx {
        records = records.into_iter().filter(|r| r.round == Some(round)).collect();
    }
    if let Some(reasons) = include_reasons {
        records = records
            .into_iter()
            .filter(|r| reasons.iter().any(|reason| *reason == r.reason))
            .collect();
    }

    let mut buyer_prices = Vec::new();
    let mut buyer_qtys = Vec::new();
    let mut seller_prices = Vec::new();
    let mut seller_qtys = Vec::new();

    for r in &records {
        let qty = r.quantity;
        if qty <= 1.0e-8 {
            continue;
        }
        buyer_prices.push(r.buyer_price);
        buyer_qtys.push(qty);
        seller_prices.push(r.seller_price);
        seller_qtys.push(qty);
    }

    if buyer_prices.is_empty() && seller_prices.is_empty() {
        return None;
    }

    let mut all_prices = buyer_prices.clone();
    all_prices.extend(seller_prices.iter().copied());
    let pmin = all_prices.iter().copied().fold(f64::INFINITY, f64::min).max(0.001);
    let pmax = all_prices.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(pmin + 1.0e-3);
    let price_grid = linspace(pmin, pmax, bins);

    let demand = price_grid
        .iter()
        .map(|&p| {
            buyer_prices
                .iter()
                .zip(buyer_qtys.iter())
                .map(|(&bp, &q)| if bp >= p { q } else { 0.0 })
                .sum()
        })
        .collect::<Vec<_>>();

    let supply = price_grid
        .iter()
        .map(|&p| {
            seller_prices
                .iter()
                .zip(seller_qtys.iter())
                .map(|(&sp, &q)| if sp <= p { q } else { 0.0 })
                .sum()
        })
        .collect::<Vec<_>>();

    Some((price_grid, demand, supply))
}

pub fn population_evolution_market(config: &SimulationConfig) -> SimulationResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let resource_grid = make_resource_grid(config.grid_size, config.gaussian_sigma, &mut rng);
    let mut agents = first_generation(config.num_agents, config.grid_size, &mut rng);
    let mut next_id = config.num_agents;
    let mut history = initialize_tracker();
    let mut marriages: Vec<Relationship> = Vec::new();
    let mut market_history = MarketHistory::default();
    let mut order_history = OrderHistory::default();
    let mut snapshot_history = Vec::new();
    let mut price_history = PriceHistory::default();

    for round in 0..config.total_rounds {
        println!("Round {}, population: {}", round + 1, agents.len());

        for a in &mut agents {
            produce(a, &resource_grid);
        }

        let mut relationships = build_relationships_bruteforce(&agents, &mut rng);
        let market_pairs = select_trade_pairs(&relationships, &mut rng);
        record_market_state(&agents, &mut market_history);
        let (trades_completed, trade_log) = bilateral_market(&market_pairs, &mut agents, config.delta, &mut price_history);
        record_order_flow(&trade_log, &mut order_history, round);
        agent_learning(&mut agents, &mut rng);
        record_trade_log(&trade_log, &mut market_history);

        let mut consumed: HashMap<usize, f64> = HashMap::new();
        let mut surplus_by_agent: HashMap<usize, f64> = HashMap::new();
        for a in &mut agents {
            let (energy_gained, surplus) = consume_from_inventory(a);
            consumed.insert(a.id, energy_gained);
            surplus_by_agent.insert(a.id, surplus);
        }
        assign_food_to_relationships(&mut relationships, &surplus_by_agent);

        let before_survival_count = agents.len();
        let mut survivors = Vec::with_capacity(agents.len());
        for mut a in agents.drain(..) {
            let ate = *consumed.get(&a.id).unwrap_or(&0.0);
            if survives(&mut a, round, ate) {
                survivors.push(a);
            }
        }
        agents = survivors;

        let alive_ids = agents.iter().map(|a| a.id).collect::<HashSet<_>>();
        let after_survival_count = alive_ids.len();
        marriages.retain(|m| alive_ids.contains(&m.a_id) && alive_ids.contains(&m.b_id));

        let relationships = build_relationships_bruteforce(&agents, &mut rng);
        let friendships = build_friendships(&relationships);
        let new_marriages = build_marriages(&mut agents, &friendships);
        marriages.extend(new_marriages);

        let (children1, next_after_marriage) = reproduce(
            &marriages,
            &mut agents,
            next_id,
            round as f64,
            config.grid_size,
            &mut rng,
        );
        let (children2, next_after_random) = random_mating(
            &mut agents,
            next_after_marriage,
            round as f64,
            config.grid_size,
            &mut rng,
            0.01,
            0.3,
        );
        next_id = next_after_random;

        let births = children1.len() + children2.len();
        agents.extend(children1);
        agents.extend(children2);

        let total_food_stock = total_food_inventory(&agents);
        record_population_state(&agents, &mut history, total_food_stock);
        record_population_snapshot(&agents, &mut snapshot_history);

        let avg_energy = if agents.is_empty() {
            0.0
        } else {
            agents.iter().map(|a| a.energy).sum::<f64>() / agents.len() as f64
        };
        let avg_food = if agents.is_empty() {
            0.0
        } else {
            agents.iter().map(|a| a.inventory[0] + a.inventory[1]).sum::<f64>() / agents.len() as f64
        };
        let deaths = before_survival_count.saturating_sub(after_survival_count);
        let avg_age = if agents.is_empty() {
            0.0
        } else {
            agents.iter().map(|a| round as f64 - a.birth_round).sum::<f64>() / agents.len() as f64
        };
        println!(
            "Trades: {}, Births: {}, Deaths: {}, Avg Energy: {:.2}, Avg Food: {:.2}, Avg Age: {:.2}",
            trades_completed, births, deaths, avg_energy, avg_food, avg_age
        );
    }

    SimulationResult {
        agents,
        history,
        market_history,
        order_history,
        snapshot_history,
        resource_grid,
    }
}

pub fn write_history_csv<P: AsRef<Path>>(path: P, history: &History) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    write!(w, "round,population,food")?;
    for key in TraitKey::ALL {
        write!(w, ",{}", key.name())?;
    }
    writeln!(w)?;

    for i in 0..history.population.len() {
        write!(w, "{},{},{}", i, history.population[i], history.food[i])?;
        for key in TraitKey::ALL {
            let values = history.traits.get(key.name()).unwrap();
            write!(w, ",{}", values[i])?;
        }
        writeln!(w)?;
    }
    Ok(())
}

pub fn write_market_csv<P: AsRef<Path>>(path: P, market_history: &MarketHistory) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "round,good,executed_price,executed_qty")?;
    for good in Good::ALL {
        let series = market_history.series(good);
        for i in 0..series.executed_prices.len() {
            writeln!(w, "{},{},{},{}", i, good.name(), series.executed_prices[i], series.executed_qty[i])?;
        }
    }
    Ok(())
}

pub fn write_orders_csv<P: AsRef<Path>>(path: P, order_history: &OrderHistory) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "good,round,buyer_id,seller_id,mid_price,buyer_price,seller_price,quantity,t_cost,buyer_cash,seller_inventory,buyer_mrs,seller_mrs,success,reason"
    )?;
    for good in Good::ALL {
        for record in order_history.get(good) {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                good.name(),
                record.round.unwrap_or(usize::MAX),
                record.buyer_id,
                record.seller_id,
                record.mid_price,
                record.buyer_price,
                record.seller_price,
                record.quantity,
                record.t_cost,
                record.buyer_cash,
                record.seller_inventory,
                record.buyer_mrs,
                record.seller_mrs,
                record.success,
                record.reason
            )?;
        }
    }
    Ok(())
}

pub fn write_snapshot_csv<P: AsRef<Path>>(path: P, snapshot_history: &[Snapshot]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "round,agent_id,x,y,birth_round,energy")?;
    for (round, snapshot) in snapshot_history.iter().enumerate() {
        for i in 0..snapshot.ids.len() {
            writeln!(
                w,
                "{},{},{},{},{},{}",
                round,
                snapshot.ids[i],
                snapshot.positions[i].x,
                snapshot.positions[i].y,
                snapshot.birth_rounds[i],
                snapshot.energies[i]
            )?;
        }
    }
    Ok(())
}

pub fn write_resource_grid_csv<P: AsRef<Path>>(path: P, resource_grid: &ResourceGrid) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "x,y,apple,barracuda")?;
    let n = resource_grid.size();
    for x in 0..n {
        for y in 0..n {
            writeln!(w, "{},{},{},{}", x, y, resource_grid.apple[x][y], resource_grid.barracuda[x][y])?;
        }
    }
    Ok(())
}

pub fn write_empirical_curve_csv<P: AsRef<Path>>(
    path: P,
    order_history: &OrderHistory,
    good: Good,
    round_idx: Option<usize>,
    bins: usize,
    include_reasons: Option<&[&str]>,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "price,demand,supply")?;
    if let Some((price_grid, demand, supply)) = build_empirical_supply_demand_filtered(order_history, good, round_idx, bins, include_reasons) {
        for i in 0..price_grid.len() {
            writeln!(w, "{},{},{}", price_grid[i], demand[i], supply[i])?;
        }
    }
    Ok(())
}

pub fn save_discrete_outputs<P: AsRef<Path>>(base_dir: P, sim: &SimulationResult) -> std::io::Result<()> {
    let base = base_dir.as_ref();
    create_dir_all(base)?;
    write_history_csv(base.join("history.csv"), &sim.history)?;
    write_market_csv(base.join("market.csv"), &sim.market_history)?;
    write_orders_csv(base.join("orders.csv"), &sim.order_history)?;
    write_snapshot_csv(base.join("snapshots.csv"), &sim.snapshot_history)?;
    write_resource_grid_csv(base.join("resource_grid.csv"), &sim.resource_grid)?;
    write_empirical_curve_csv(base.join("apple_empirical_all.csv"), &sim.order_history, Good::Apple, None, 40, None)?;
    write_empirical_curve_csv(base.join("barracuda_empirical_all.csv"), &sim.order_history, Good::Barracuda, None, 40, None)?;
    write_empirical_curve_csv(
        base.join("apple_empirical_filtered.csv"),
        &sim.order_history,
        Good::Apple,
        None,
        40,
        Some(&["executed", "buyer_lacks_cash", "seller_lacks_inventory"]),
    )?;
    write_empirical_curve_csv(
        base.join("barracuda_empirical_filtered.csv"),
        &sim.order_history,
        Good::Barracuda,
        None,
        40,
        Some(&["executed", "buyer_lacks_cash", "seller_lacks_inventory"]),
    )?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ContinuousConfig {
    pub initial_agents: usize,
    pub grid_size: usize,
    pub gaussian_sigma: f64,
    pub trade_delta: f64,
    pub learning_interval: f64,
    pub market_interval: f64,
    pub reproduction_interval: f64,
    pub random_mating_rate: f64,
    pub marriage_penalty: f64,
    pub seed: u64,
}

impl Default for ContinuousConfig {
    fn default() -> Self {
        Self {
            initial_agents: 1000,
            grid_size: 50,
            gaussian_sigma: 5.0,
            trade_delta: 0.1,
            learning_interval: 0.20,
            market_interval: 0.10,
            reproduction_interval: 0.50,
            random_mating_rate: 0.01,
            marriage_penalty: 0.3,
            seed: 67,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepStats {
    pub trades_completed: usize,
    pub births: usize,
    pub deaths: usize,
    pub avg_energy: f64,
    pub avg_food: f64,
    pub avg_age: f64,
    pub population: usize,
}

pub struct ContinuousWorld {
    pub config: ContinuousConfig,
    pub agents: Vec<Agent>,
    pub next_id: usize,
    pub resource_grid: ResourceGrid,
    pub history: History,
    pub market_history: MarketHistory,
    pub order_history: OrderHistory,
    pub snapshot_history: Vec<Snapshot>,
    pub price_history: PriceHistory,
    pub marriages: Vec<Relationship>,
    pub relationships: HashMap<(usize, usize), Relationship>,
    pub rng: StdRng,
    pub time: f64,
    pub learning_clock: f64,
    pub market_clock: f64,
    pub reproduction_clock: f64,
}

impl ContinuousWorld {
    pub fn new(config: ContinuousConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let resource_grid = make_resource_grid(config.grid_size, config.gaussian_sigma, &mut rng);
        let agents = first_generation(config.initial_agents, config.grid_size, &mut rng);
        let next_id = config.initial_agents;
        let mut world = Self {
            config,
            agents,
            next_id,
            resource_grid,
            history: initialize_tracker(),
            market_history: MarketHistory::default(),
            order_history: OrderHistory::default(),
            snapshot_history: Vec::new(),
            price_history: PriceHistory::default(),
            marriages: Vec::new(),
            relationships: HashMap::new(),
            rng,
            time: 0.0,
            learning_clock: 0.0,
            market_clock: 0.0,
            reproduction_clock: 0.0,
        };
        let food = total_food_inventory(&world.agents);
        record_population_state(&world.agents, &mut world.history, food);
        record_population_snapshot(&world.agents, &mut world.snapshot_history);
        world
    }

    pub fn step(&mut self, dt: f64) -> StepStats {
        self.time += dt;

        for agent in &mut self.agents {
            produce_dt(agent, &self.resource_grid, dt);
        }

        self.learning_clock += dt;
        while self.learning_clock >= self.config.learning_interval {
            agent_learning(&mut self.agents, &mut self.rng);
            self.learning_clock -= self.config.learning_interval;
        }

        self.relationships = build_relationships_bruteforce(&self.agents, &mut self.rng);

        let mut trades_completed = 0usize;
        self.market_clock += dt;
        while self.market_clock >= self.config.market_interval {
            let market_pairs = select_trade_pairs(&self.relationships, &mut self.rng);
            record_market_state(&self.agents, &mut self.market_history);
            let (n, trade_log) = bilateral_market(&market_pairs, &mut self.agents, self.config.trade_delta, &mut self.price_history);
            trades_completed += n;
            record_order_flow(&trade_log, &mut self.order_history, self.time.floor() as usize);
            record_trade_log(&trade_log, &mut self.market_history);
            self.market_clock -= self.config.market_interval;
        }

        let mut consumed: HashMap<usize, f64> = HashMap::new();
        let mut surplus_by_agent: HashMap<usize, f64> = HashMap::new();
        for agent in &mut self.agents {
            let (energy_gained, surplus) = consume_from_inventory_dt(agent, dt);
            consumed.insert(agent.id, energy_gained);
            surplus_by_agent.insert(agent.id, surplus);
        }
        assign_food_to_relationships(&mut self.relationships, &surplus_by_agent);

        let before_survival_count = self.agents.len();
        let mut survivors = Vec::with_capacity(self.agents.len());
        for mut agent in self.agents.drain(..) {
            let ate = *consumed.get(&agent.id).unwrap_or(&0.0);
            if survives_continuous(&mut agent, self.time, ate, dt) {
                survivors.push(agent);
            }
        }
        self.agents = survivors;
        let alive_ids = self.agents.iter().map(|a| a.id).collect::<HashSet<_>>();
        let after_survival_count = alive_ids.len();
        self.marriages.retain(|m| alive_ids.contains(&m.a_id) && alive_ids.contains(&m.b_id));

        self.relationships = build_relationships_bruteforce(&self.agents, &mut self.rng);
        let friendships = build_friendships(&self.relationships);
        let new_marriages = build_marriages(&mut self.agents, &friendships);
        self.marriages.extend(new_marriages);

        let mut births = 0usize;
        self.reproduction_clock += dt;
        while self.reproduction_clock >= self.config.reproduction_interval {
            let (children1, next_after_marriage) = reproduce(
                &self.marriages,
                &mut self.agents,
                self.next_id,
                self.time,
                self.config.grid_size,
                &mut self.rng,
            );
            let (children2, next_after_random) = random_mating(
                &mut self.agents,
                next_after_marriage,
                self.time,
                self.config.grid_size,
                &mut self.rng,
                self.config.random_mating_rate,
                self.config.marriage_penalty,
            );
            births += children1.len() + children2.len();
            self.next_id = next_after_random;
            self.agents.extend(children1);
            self.agents.extend(children2);
            self.reproduction_clock -= self.config.reproduction_interval;
        }

        let total_food_stock = total_food_inventory(&self.agents);
        record_population_state(&self.agents, &mut self.history, total_food_stock);
        record_population_snapshot(&self.agents, &mut self.snapshot_history);

        let avg_energy = if self.agents.is_empty() {
            0.0
        } else {
            self.agents.iter().map(|a| a.energy).sum::<f64>() / self.agents.len() as f64
        };
        let avg_food = if self.agents.is_empty() {
            0.0
        } else {
            self.agents.iter().map(|a| a.inventory[0] + a.inventory[1]).sum::<f64>() / self.agents.len() as f64
        };
        let avg_age = if self.agents.is_empty() {
            0.0
        } else {
            self.agents.iter().map(|a| self.time - a.birth_round).sum::<f64>() / self.agents.len() as f64
        };

        StepStats {
            trades_completed,
            births,
            deaths: before_survival_count.saturating_sub(after_survival_count),
            avg_energy,
            avg_food,
            avg_age,
            population: self.agents.len(),
        }
    }
}
