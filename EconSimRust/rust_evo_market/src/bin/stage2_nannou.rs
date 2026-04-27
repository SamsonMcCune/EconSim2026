use nannou::prelude::*;
use rust_evo_market::{id_to_index_map, ContinuousConfig, ContinuousWorld, StepStats, Vec2, EPS};
use std::collections::HashSet;

struct Model {
    world: ContinuousWorld,
    running: bool,
    speed: f64,
    substep: f64,
    last_stats: StepStats,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(1200, 900)
        .title("Continuous Evolution Market - nannou")
        .event(window_event)
        .view(view)
        .build()
        .unwrap();

    Model {
        world: ContinuousWorld::new(ContinuousConfig::default()),
        running: true,
        speed: 1.0,
        substep: 0.02,
        last_stats: StepStats::default(),
    }
}

fn window_event(_app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        WindowEvent::KeyPressed(Key::Space) => {
            model.running = !model.running;
        }
        WindowEvent::KeyPressed(Key::Up) => {
            model.speed = (model.speed * 1.25).min(64.0);
        }
        WindowEvent::KeyPressed(Key::Down) => {
            model.speed = (model.speed / 1.25).max(0.125);
        }
        WindowEvent::KeyPressed(Key::R) => {
            model.world = ContinuousWorld::new(ContinuousConfig::default());
            model.last_stats = StepStats::default();
            model.running = true;
            model.speed = 1.0;
        }
        WindowEvent::KeyPressed(Key::S) => {
            if let Err(err) = rust_evo_market::save_discrete_outputs("output/gui_snapshot", &rust_evo_market::SimulationResult {
                agents: model.world.agents.clone(),
                history: model.world.history.clone(),
                market_history: model.world.market_history.clone(),
                order_history: model.world.order_history.clone(),
                snapshot_history: model.world.snapshot_history.clone(),
                resource_grid: model.world.resource_grid.clone(),
            }) {
                eprintln!("snapshot save failed: {}", err);
            } else {
                println!("Saved GUI snapshot CSVs to output/gui_snapshot");
            }
        }
        _ => {}
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {
    if !model.running {
        return;
    }

    let real_dt = update.since_last.as_secs_f64();
    let scaled_dt = (real_dt * model.speed).min(0.5);
    let mut remaining = scaled_dt;
    let mut last_stats = StepStats::default();

    while remaining > 0.0 {
        let dt = remaining.min(model.substep);
        last_stats = model.world.step(dt);
        remaining -= dt;
    }

    model.last_stats = last_stats;
}

fn to_screen(pos: Vec2, rect: Rect) -> Point2 {
    let x = map_range(pos.x as f32, 0.0, 1.0, rect.left(), rect.right());
    let y = map_range(pos.y as f32, 0.0, 1.0, rect.bottom(), rect.top());
    pt2(x, y)
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let rect = app.window_rect();
    draw_resource_grid(&draw, rect, &model.world);
    draw_marriages(&draw, rect, &model.world);
    draw_agents(&draw, rect, &model.world);
    draw_overlay(&draw, rect, model);

    draw.to_frame(app, &frame).unwrap();
}

fn draw_resource_grid(draw: &Draw, rect: Rect, world: &ContinuousWorld) {
    let n = world.resource_grid.size();
    let cell_w = rect.w() / n as f32;
    let cell_h = rect.h() / n as f32;

    for x in 0..n {
        for y in 0..n {
            let apple = ((world.resource_grid.apple[x][y] - 0.5) as f32).clamp(0.0, 1.0);
            let barracuda = ((world.resource_grid.barracuda[x][y] - 0.5) as f32).clamp(0.0, 1.0);
            let px = map_range((x as f32) + 0.5, 0.0, n as f32, rect.left(), rect.right());
            let py = map_range((y as f32) + 0.5, 0.0, n as f32, rect.bottom(), rect.top());
            draw.rect()
                .x_y(px, py)
                .w_h(cell_w + 1.0, cell_h + 1.0)
                .rgba(0.0, apple, barracuda, 0.90);
        }
    }
}

fn draw_marriages(draw: &Draw, rect: Rect, world: &ContinuousWorld) {
    let id_map = id_to_index_map(&world.agents);
    let mut drawn = HashSet::new();

    for marriage in &world.marriages {
        if !drawn.insert(marriage.key) {
            continue;
        }
        let Some(&ia) = id_map.get(&marriage.a_id) else { continue; };
        let Some(&ib) = id_map.get(&marriage.b_id) else { continue; };
        let a = &world.agents[ia];
        let b = &world.agents[ib];
        let pa = to_screen(a.position, rect);
        let pb = to_screen(b.position, rect);
        draw.line()
            .start(pa)
            .end(pb)
            .weight(1.0)
            .rgba(1.0, 1.0, 1.0, 0.15);
    }
}

fn draw_agents(draw: &Draw, rect: Rect, world: &ContinuousWorld) {
    for agent in &world.agents {
        let p = to_screen(agent.position, rect);
        let energy_norm = ((agent.energy / 5.0) as f32).clamp(0.0, 1.0);
        let food_norm = (((agent.inventory[0] + agent.inventory[1]) / 10.0) as f32).clamp(0.0, 1.0);
        let radius_px = 2.0 + 14.0 * (agent.radius as f32).clamp(0.0, 1.0);

        let (r, g, b) = match agent.genome.gender {
            rust_evo_market::Gender::M => (1.0, 0.25 + 0.60 * energy_norm, 0.25 + 0.30 * food_norm),
            rust_evo_market::Gender::F => (0.25 + 0.60 * food_norm, 0.25, 1.0),
        };

        draw.ellipse()
            .xy(p)
            .radius(radius_px)
            .rgba(r, g, b, 0.80);
    }
}

fn draw_overlay(draw: &Draw, rect: Rect, model: &Model) {
    let population = model.world.agents.len();
    let time = model.world.time;
    let avg_energy = model.last_stats.avg_energy;
    let avg_food = model.last_stats.avg_food;
    let avg_age = model.last_stats.avg_age;
    let trades = model.last_stats.trades_completed;
    let births = model.last_stats.births;
    let deaths = model.last_stats.deaths;
    let marriages = model.world.marriages.len();
    let price_apple = model.world.price_history.apple.last().copied().unwrap_or(0.0);
    let price_barracuda = model.world.price_history.barracuda.last().copied().unwrap_or(0.0);

    let text = format!(
        "time: {:.2}\npopulation: {}\ntrades(step): {}\nbirths(step): {}\ndeaths(step): {}\nmarriages: {}\navg energy: {:.3}\navg food: {:.3}\navg age: {:.3}\napple px: {:.3}\nbarracuda px: {:.3}\nspeed: {:.3}x\nstate: {}\n\nspace = pause/resume\nup/down = speed\nr = reset\ns = dump CSV snapshot",
        time,
        population,
        trades,
        births,
        deaths,
        marriages,
        avg_energy,
        avg_food,
        avg_age,
        price_apple,
        price_barracuda,
        model.speed,
        if model.running { "running" } else { "paused" }
    );

    let left = rect.left() + 20.0;
    let top = rect.top() - 20.0;

    draw.rect()
        .x_y(left + 170.0, top - 120.0)
        .w_h(360.0, 260.0)
        .rgba(0.0, 0.0, 0.0, 0.45);

    draw.text(&text)
        .x_y(left + 10.0, top - 10.0)
        .left_justify()
        .align_top()
        .font_size(16)
        .rgba(1.0, 1.0, 1.0, 0.95);

    let history = &model.world.history.population;
    if history.len() > 2 {
        let width = 300.0;
        let height = 80.0;
        let x0 = rect.left() + 30.0;
        let y0 = rect.bottom() + 40.0;
        let min_pop = *history.iter().min().unwrap_or(&0) as f32;
        let max_pop = *history.iter().max().unwrap_or(&1) as f32;
        let pop_range = (max_pop - min_pop).max(EPS as f32);

        for i in 1..history.len() {
            let t0 = (i - 1) as f32 / (history.len() - 1) as f32;
            let t1 = i as f32 / (history.len() - 1) as f32;
            let p0 = history[i - 1] as f32;
            let p1 = history[i] as f32;
            let x_a = x0 + width * t0;
            let x_b = x0 + width * t1;
            let y_a = y0 + height * ((p0 - min_pop) / pop_range);
            let y_b = y0 + height * ((p1 - min_pop) / pop_range);
            draw.line()
                .start(pt2(x_a, y_a))
                .end(pt2(x_b, y_b))
                .weight(2.0)
                .rgba(1.0, 0.8, 0.2, 0.9);
        }

        draw.rect()
            .x_y(x0 + width / 2.0, y0 + height / 2.0)
            .w_h(width, height)
            .no_fill()
            .stroke(rgba(1.0, 1.0, 1.0, 0.25))
            .stroke_weight(1.0);

        draw.text("population history")
            .x_y(x0 + width / 2.0, y0 + height + 12.0)
            .font_size(14)
            .rgba(1.0, 1.0, 1.0, 0.85);
    }
}
