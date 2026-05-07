use nannou::prelude::*;
use rust_evo_market::{id_to_index_map, ContinuousConfig, ContinuousWorld, StepStats, Vec2, EPS};
use std::collections::HashSet;

struct Model {
    world: ContinuousWorld,
    running: bool,

    // Visual-time stepping: the GUI advances by exactly tick_dt per rendered update.
    // It intentionally does NOT simulate a large wall-clock backlog before drawing.
    tick_dt: f64,
    ticks_per_frame: usize,

    last_stats: StepStats,
    draw_marriages: bool,
    draw_resources: bool,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn make_world() -> ContinuousWorld {
    // These fields are the same names as before, but stage 2 now interprets the interval fields
    // as continuous-time mean waiting times rather than hard synchronized timers.
    let config = ContinuousConfig {
        initial_agents: 1000,
        grid_size: 50,
        gaussian_sigma: 5.0,
        trade_delta: 0.1,
        learning_interval: 0.20,
        market_interval: 0.10,
        reproduction_interval: 0.50,
        random_mating_rate: 0.01,
        marriage_penalty: 0.3,
        movement_speed: 0.04,
        movement_probe_distance: 1.0 / 50.0,
        movement_food_weight: 1.0,
        movement_spouse_weight: 10.0,
        movement_single_social_weight: 3.0,
        movement_single_social_scale: 0.12,
        movement_single_settle_distance: 0.08,
        movement_single_search_weight: 1.0,
        movement_food_lookahead: 1.0,
        movement_energy_cost: 1.0,
        movement_min_gradient: 1.0e-6,
        seed: 67,
    };
    ContinuousWorld::new(config)
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
        world: make_world(),
        running: true,
        tick_dt: 0.01,
        ticks_per_frame: 1,
        last_stats: StepStats::default(),
        draw_marriages: true,
        draw_resources: true,
    }
}

fn window_event(_app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        WindowEvent::KeyPressed(Key::Space) => {
            model.running = !model.running;
        }
        WindowEvent::KeyPressed(Key::Up) => {
            // More ticks per drawn frame speeds up model time, but the default remains one
            // visible 0.01s tick per update so behavior is easy to watch.
            model.ticks_per_frame = (model.ticks_per_frame + 1).min(20);
        }
        WindowEvent::KeyPressed(Key::Down) => {
            model.ticks_per_frame = model.ticks_per_frame.saturating_sub(1).max(1);
        }
        WindowEvent::KeyPressed(Key::Left) => {
            model.tick_dt = (model.tick_dt / 2.0).max(0.001);
        }
        WindowEvent::KeyPressed(Key::Right) => {
            model.tick_dt = (model.tick_dt * 2.0).min(0.05);
        }
        WindowEvent::KeyPressed(Key::M) => {
            model.draw_marriages = !model.draw_marriages;
        }
        WindowEvent::KeyPressed(Key::G) => {
            model.draw_resources = !model.draw_resources;
        }
        WindowEvent::KeyPressed(Key::R) => {
            model.world = make_world();
            model.last_stats = StepStats::default();
            model.running = true;
            model.tick_dt = 0.01;
            model.ticks_per_frame = 1;
        }
        WindowEvent::KeyPressed(Key::S) => {
            if let Err(err) = rust_evo_market::save_discrete_outputs(
                "output/gui_snapshot",
                &rust_evo_market::SimulationResult {
                    agents: model.world.agents.clone(),
                    history: model.world.history.clone(),
                    market_history: model.world.market_history.clone(),
                    order_history: model.world.order_history.clone(),
                    snapshot_history: model.world.snapshot_history.clone(),
                    resource_grid: model.world.resource_grid.clone(),
                },
            ) {
                eprintln!("snapshot save failed: {}", err);
            } else {
                println!("Saved GUI snapshot CSVs to output/gui_snapshot");
            }
        }
        _ => {}
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    if !model.running {
        return;
    }

    // This is the key change. The old code used wall-clock catch-up:
    //     real elapsed time * speed, then as many 0.01 substeps as needed.
    // That can process 0.3-0.5 model seconds before the next draw, which looks like a
    // chunky discrete jump. Here, every drawn update advances by a fixed number of
    // small stochastic ticks. If the computer is slow, the simulation slows down
    // instead of jumping ahead invisibly.
    let mut frame_stats = StepStats::default();
    for _ in 0..model.ticks_per_frame {
        let s = model.world.step(model.tick_dt);
        frame_stats.accumulate(s);
    }

    model.last_stats = frame_stats;
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
    if model.draw_resources {
        draw_resource_grid(&draw, rect, &model.world);
    }
    if model.draw_marriages {
        draw_marriages(&draw, rect, &model.world);
    }
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
            .rgba(1.0, 1.0, 1.0, 0.14);
    }
}

fn draw_agents(draw: &Draw, rect: Rect, world: &ContinuousWorld) {
    for agent in &world.agents {
        let p = to_screen(agent.position, rect);
        let energy_norm = ((agent.energy / 5.0) as f32).clamp(0.0, 1.0);
        let food_norm = (((agent.inventory[0] + agent.inventory[1]) / 10.0) as f32).clamp(0.0, 1.0);
        let radius_px = 2.0 + 14.0 * (agent.radius as f32).clamp(0.0, 1.0);

        // Gender gives the broad hue; energy and food brighten the point.
        let (r, g, b) = match agent.genome.gender {
            rust_evo_market::Gender::M => (
                1.0,
                0.20 + 0.65 * energy_norm,
                0.20 + 0.35 * food_norm,
            ),
            rust_evo_market::Gender::F => (
                0.20 + 0.50 * food_norm,
                0.20 + 0.35 * energy_norm,
                1.0,
            ),
        };

        draw.ellipse()
            .xy(p)
            .radius(radius_px)
            .rgba(r, g, b, 0.82);
    }
}

fn draw_overlay(draw: &Draw, rect: Rect, model: &Model) {
    let population = model.world.agents.len();
    let time = model.world.time;
    let avg_energy = model.last_stats.avg_energy;
    let avg_food = model.last_stats.avg_food;
    let avg_age = model.last_stats.avg_age;
    let trades = model.last_stats.trades_completed;
    let trade_attempts = model.last_stats.trade_attempts;
    let births = model.last_stats.births;
    let deaths = model.last_stats.deaths;
    let learning = model.last_stats.learning_events;
    let new_marriages = model.last_stats.new_marriages;
    let marriages = model.world.marriages.len();
    let price_apple = model.world.price_history.apple.last().copied().unwrap_or(0.0);
    let price_barracuda = model.world.price_history.barracuda.last().copied().unwrap_or(0.0);

    let text = format!(
        "time: {:.2}\npopulation: {}\ntrades/frame: {}\ntrade attempts/frame: {}\nbirths/frame: {}\ndeaths/frame: {}\nnew marriages/frame: {}\nmarriages: {}\nlearning events/frame: {}\navg energy: {:.3}\navg food: {:.3}\navg age: {:.3}\napple px: {:.3}\nbarracuda px: {:.3}\ntick dt: {:.4}\nticks/frame: {}\nstate: {}\n\nspace = pause/resume\nup/down = ticks per frame\nleft/right = tick dt\nm = marriages\ng = resources\nr = reset\ns = dump CSV snapshot",
        time,
        population,
        trades,
        trade_attempts,
        births,
        deaths,
        new_marriages,
        marriages,
        learning,
        avg_energy,
        avg_food,
        avg_age,
        price_apple,
        price_barracuda,
        model.tick_dt,
        model.ticks_per_frame,
        if model.running { "running" } else { "paused" }
    );

    let left = rect.left() + 20.0;
    let top = rect.top() - 20.0;
    let box_w = 280.0;
    let box_h = 430.0;
    let box_x = left + box_w / 2.0 - 10.0;
    let box_y = top - box_h / 2.0 + 10.0;

    draw.rect()
        .x_y(box_x, box_y)
        .w_h(box_w, box_h)
        .rgba(0.0, 0.0, 0.0, 0.70)
        .no_fill()
        .stroke(WHITE)
        .stroke_weight(2.0);

    draw.text(&text)
        .x_y(left + 130.0, top - 115.0)
        .left_justify()
        .align_text_top()
        .font_size(16)
        .rgba(1.0, 1.0, 1.0, 1.0);

    draw_population_history(draw, rect, model);
}

fn draw_population_history(draw: &Draw, rect: Rect, model: &Model) {
    let history = &model.world.history.population;
    if history.len() <= 2 {
        return;
    }

    let max_points = 500usize;
    let start = history.len().saturating_sub(max_points);
    let visible = &history[start..];
    if visible.len() <= 2 {
        return;
    }

    let width = 320.0;
    let height = 90.0;
    let x0 = rect.left() + 30.0;
    let y0 = rect.bottom() + 40.0;
    let min_pop = *visible.iter().min().unwrap_or(&0) as f32;
    let max_pop = *visible.iter().max().unwrap_or(&1) as f32;
    let pop_range = (max_pop - min_pop).max(EPS as f32);

    for i in 1..visible.len() {
        let t0 = (i - 1) as f32 / (visible.len() - 1) as f32;
        let t1 = i as f32 / (visible.len() - 1) as f32;
        let p0 = visible[i - 1] as f32;
        let p1 = visible[i] as f32;
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
