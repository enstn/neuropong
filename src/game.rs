use ggez::*;
use ggez::graphics;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use crate::{GameState, PaddleAction};

const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;
const BALL_RADIUS: f32 = 10.0;
const PAD_LENGTH: f32 = 100.0;
const PAD_WIDTH: f32 = 10.0;

struct GameInstance {
    ball_x: f32,
    ball_y: f32,
    ball_vel_x: f32,
    ball_vel_y: f32,
    paddle_y: f32,
    score: i32,
    action: Arc<Mutex<PaddleAction>>,
    callback: Arc<Mutex<Option<PyObject>>>,
}

impl GameInstance {
    fn new(action: Arc<Mutex<PaddleAction>>, callback: Arc<Mutex<Option<PyObject>>>) -> Self {
        GameInstance {
            ball_x: 400.0,
            ball_y: 300.0,
            ball_vel_x: -300.0,
            ball_vel_y: 100.0,
            paddle_y: 250.0,
            score: 0,
            action,
            callback,
        }
    }

    fn notify_python(&self) {
        if let Ok(callback_guard) = self.callback.lock() {
            if let Some(callback) = &*callback_guard {
                Python::with_gil(|py| {
                    let state = GameState {
                        ball_x: self.ball_x,
                        ball_y: self.ball_y,
                        paddle_y: self.paddle_y,
                        score: self.score,
                    };
                    
                    // Call Python callback with current state
                    let _ = callback.call1(py, (state,));
                });
            }
        }
    }
}

impl event::EventHandler<GameError> for GameInstance {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        // Update ball position
        self.ball_x += self.ball_vel_x * 0.016;
        self.ball_y += self.ball_vel_y * 0.016;

        // Ball bouncing
        if self.ball_y <= 0.0 || self.ball_y >= 600.0 {
            self.ball_vel_y = -self.ball_vel_y;
        }
        if self.ball_x >= 800.0 {
            self.ball_vel_x = -self.ball_vel_x;
        }

        // Reset on miss
        if self.ball_x <= 0.0 {
            self.ball_x = 400.0;
            self.ball_y = 300.0;
            self.ball_vel_x = -300.0;
            self.ball_vel_y = 100.0;
            self.score = 0;
        }

        // Update paddle based on current action
        if let Ok(action) = self.action.lock() {
            match *action {
                PaddleAction::Up => {
                    if self.paddle_y > 0.0 {
                        self.paddle_y -= 500.0 * 0.016;
                    }
                }
                PaddleAction::Down => {
                    if self.paddle_y < 500.0 {
                        self.paddle_y += 500.0 * 0.016;
                    }
                }
                PaddleAction::Stay => {}
            }
        }

        // Check paddle collision
        if self.ball_x <= 50.0 && self.ball_x >= 40.0 &&
           self.ball_y >= self.paddle_y && self.ball_y <= self.paddle_y + PAD_LENGTH {
            self.ball_vel_x = -self.ball_vel_x;
            self.score += 1;
        }

        // Notify Python of state change
        self.notify_python();

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);

        // Draw ball
        let ball = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            mint::Point2{
                x: self.ball_x,
                y: self.ball_y
            },
            BALL_RADIUS,
            0.1,
            graphics::Color::WHITE,
        )?;

        // Draw paddle
        let pad = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::fill(),
            graphics::Rect::new(40.0, self.paddle_y, PAD_WIDTH, PAD_LENGTH),
            graphics::Color::WHITE,
        )?;

        // Draw score
        let score_text = graphics::Text::new(format!("Score: {}", self.score));

        canvas.draw(&ball, graphics::DrawParam::default());
        canvas.draw(&pad, graphics::DrawParam::default());
        canvas.draw(&score_text, graphics::DrawParam::default().dest([10.0, 10.0]));
        
        canvas.finish(ctx)?;
        Ok(())
    }
}

pub fn run_game(
    action: Arc<Mutex<PaddleAction>>,
    callback: Arc<Mutex<Option<PyObject>>>,
) -> GameResult {
    let cb = ContextBuilder::new("pong", "you")
        .window_mode(conf::WindowMode::default()
            .dimensions(SCREEN_WIDTH, SCREEN_HEIGHT));

    let (ctx, event_loop) = cb.build()?;
    let state = GameInstance::new(action, callback);
    
    event::run(ctx, event_loop, state)
}