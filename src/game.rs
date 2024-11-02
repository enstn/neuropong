use ggez::*;
use ggez::graphics;
use std::sync::{Arc, Mutex};
use crate::{PyGameState, PaddleAction};

const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;
const SCREEN_WIDTH_MID: f32 = SCREEN_WIDTH / 2.0;
const SCREEN_HEIGHT_MID: f32 = SCREEN_HEIGHT / 2.0;

const BALL_VELOCITY: f32 = 300.0;
const BALL_RADIUS: f32 = 10.0;
const BALL_RADIUS_MID: f32 = BALL_RADIUS / 2.0;

const PAD_VELOCITY: f32 = 500.0;
const PAD_LENGTH: f32 = 100.0;
const PAD_WIDTH: f32 = 10.0;
const PAD_LEFT_EDGE: f32 = 40.0;

struct Ball {
    pos: mint::Point2<f32>,
    vel: mint::Vector2<f32>,
}

impl Ball {
    pub fn new() -> Self {
        Ball {
            pos: mint::Point2{
                x: SCREEN_WIDTH_MID - BALL_RADIUS_MID, 
                y: SCREEN_HEIGHT_MID - BALL_RADIUS_MID
            },
            vel: mint::Vector2{
                x: BALL_VELOCITY, 
                y: BALL_VELOCITY
            },
        }
    }
}

struct Pad {
    rect: graphics::Rect,
    velocity: f32,
}

impl Pad {
    pub fn new() -> Self {
        Pad {
            rect: graphics::Rect::new(
                PAD_LEFT_EDGE,
                SCREEN_HEIGHT_MID - (PAD_LENGTH / 2.0),
                PAD_WIDTH,
                PAD_LENGTH,
            ),
            velocity: 0.0,
        }
    }
}

struct GameState {
    ball: Ball,
    pad: Pad,
    score: i32,
    py_state: Arc<Mutex<PyGameState>>,
    py_action: Arc<Mutex<PaddleAction>>,
}

impl GameState {
    pub fn new(py_state: Arc<Mutex<PyGameState>>, py_action: Arc<Mutex<PaddleAction>>) -> Self {
        GameState {
            ball: Ball::new(),
            pad: Pad::new(),
            score: 0,
            py_state,
            py_action,
        }
    }

    fn update_python_state(&mut self) {
        if let Ok(mut state) = self.py_state.lock() {
            state.ball_x = self.ball.pos.x;
            state.ball_y = self.ball.pos.y;
            state.paddle_y = self.pad.rect.y;
            state.score = self.score;
        }
    }

    fn handle_paddle_action(&mut self, dt: f32) {
        if let Ok(action) = self.py_action.lock() {
            match *action {
                PaddleAction::Up => {
                    if self.pad.rect.y > 0.0 {
                        self.pad.velocity = -PAD_VELOCITY;
                        self.pad.rect.y += self.pad.velocity * dt;
                    }
                },
                PaddleAction::Down => {
                    if self.pad.rect.y < (SCREEN_HEIGHT - PAD_LENGTH) {
                        self.pad.velocity = PAD_VELOCITY;
                        self.pad.rect.y += self.pad.velocity * dt;
                    }
                },
                PaddleAction::Stay => self.pad.velocity = 0.0,
            }
        }
    }
}

impl event::EventHandler<GameError> for GameState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = ctx.time.delta().as_secs_f32();

        // Update ball position
        self.ball.pos.x += self.ball.vel.x * dt;
        self.ball.pos.y += self.ball.vel.y * dt;

        // Handle ball collisions
        if self.ball.pos.x > SCREEN_WIDTH || self.ball.pos.x < 0.0 {
            self.ball.vel.x *= -1.0;
        }
        if self.ball.pos.y > SCREEN_HEIGHT || self.ball.pos.y < 0.0 {
            self.ball.vel.y *= -1.0;
        }

        // Handle paddle movement
        self.handle_paddle_action(dt);

        // Update Python state
        self.update_python_state();

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);

        // Draw ball
        let ball = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            self.ball.pos,
            BALL_RADIUS,
            0.1,
            graphics::Color::WHITE,
        )?;

        // Draw paddle
        let pad = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::fill(),
            self.pad.rect,
            graphics::Color::WHITE,
        )?;

        canvas.draw(&ball, graphics::DrawParam::default());
        canvas.draw(&pad, graphics::DrawParam::default());
        canvas.finish(ctx)?;

        Ok(())
    }
}

pub fn run_game(py_state: Arc<Mutex<PyGameState>>, py_action: Arc<Mutex<PaddleAction>>) {
    let cb = ContextBuilder::new("pong", "you")
        .window_mode(conf::WindowMode::default()
            .dimensions(SCREEN_WIDTH, SCREEN_HEIGHT));

    if let Ok((ctx, event_loop)) = cb.build() {
        let state = GameState::new(py_state, py_action);
        let _ = event::run(ctx, event_loop, state);
    }
}
