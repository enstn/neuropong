use ggez::audio::SoundSource;
use ggez::*;
use ggez::graphics;
use ggez::audio;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use rand::Rng;
use crate::{GameState, PaddleAction};

const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;
const SCREEN_WIDTH_MID: f32 = SCREEN_WIDTH / 2.0;
const SCREEN_HEIGHT_MID: f32 = SCREEN_HEIGHT / 2.0;

const BALL_VELOCITY: f32 = 400.0;
const BALL_RADIUS: f32 = 10.0;
const BALL_RADIUS_MID: f32 = BALL_RADIUS / 2.0;

const PAD_VELOCITY: f32 = 500.0;
const PAD_LENGTH: f32 = 100.0;
const PAD_WIDTH: f32 = 10.0;
const PAD_LEFT_EDGE: f32 = 40.0;

struct GameInstance {
    ball_pos: mint::Point2<f32>,
    ball_vel: mint::Vector2<f32>,
    paddle_y: f32,
    score: i32,
    action: Arc<Mutex<PaddleAction>>,
    callback: Arc<Mutex<Option<PyObject>>>,
    sound1: audio::Source,
    sound2: audio::Source,
    sound3: audio::Source,
}

impl GameInstance {
    fn new(ctx: &mut Context, action: Arc<Mutex<PaddleAction>>, callback: Arc<Mutex<Option<PyObject>>>) -> GameResult<Self> {
        let sound1 = audio::Source::new(ctx, "/plop.mp3")?;
        let sound2 = audio::Source::new(ctx, "/sound.ogg")?;
        let sound3 = audio::Source::new(ctx, "/stomp.mp3")?;

        Ok(GameInstance {
            ball_pos: mint::Point2 {
                x: SCREEN_WIDTH_MID - BALL_RADIUS_MID,
                y: SCREEN_HEIGHT_MID - BALL_RADIUS_MID
            },
            ball_vel: Self::generate_random_velocity(),
            paddle_y: SCREEN_HEIGHT_MID - (PAD_LENGTH / 2.0),
            score: 0,
            action,
            callback,
            sound1,
            sound2,
            sound3,
        })
    }

    fn generate_random_velocity() -> mint::Vector2<f32> {
        let mut rng = rand::thread_rng();
        
        // Random angle between -45 and 45 degrees (avoiding too vertical trajectories)
        let angle = rng.gen_range(-std::f32::consts::PI/4.0..std::f32::consts::PI/4.0);
        
        // Calculate x and y components
        let x = BALL_VELOCITY * angle.cos();  // positive x go give agent time to adjust
        let y = BALL_VELOCITY * angle.sin();
        
        mint::Vector2 { x, y }
    }

    fn reset_ball(&mut self) {
        self.ball_pos.x = SCREEN_WIDTH_MID - BALL_RADIUS_MID;
        self.ball_pos.y = SCREEN_HEIGHT_MID - BALL_RADIUS_MID;
        self.ball_vel = Self::generate_random_velocity();
    }

    fn check_collision(&self, ball_pos: mint::Point2<f32>, ball_radius: f32, paddle: &graphics::Rect) -> bool {
        // Find the closest point on the rectangle to the circle's center
        let closest_x = ball_pos.x.clamp(paddle.x, paddle.x + paddle.w);
        let closest_y = ball_pos.y.clamp(paddle.y, paddle.y + paddle.h);

        // Calculate the distance between the circle's center and the closest point
        let distance_x = ball_pos.x - closest_x;
        let distance_y = ball_pos.y - closest_y;

        // If the distance is less than the circle's radius, there is a collision
        (distance_x * distance_x + distance_y * distance_y) <= (ball_radius * ball_radius)
    }

    fn notify_python(&self) {
        if let Ok(callback_guard) = self.callback.lock() {
            if let Some(callback) = &*callback_guard {
                Python::with_gil(|py| {
                    let state = GameState {
                        ball_x: self.ball_pos.x,
                        ball_y: self.ball_pos.y,
                        paddle_y: self.paddle_y,
                        score: self.score,
                    };
                    let _ = callback.call1(py, (state,));
                });
            }
        }
    }
}

impl event::EventHandler<GameError> for GameInstance {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = 0.016; // ~60 FPS

        // Update ball position
        self.ball_pos.x += self.ball_vel.x * dt;
        self.ball_pos.y += self.ball_vel.y * dt;

        // Ball bouncing off top and bottom
        if self.ball_pos.y <= BALL_RADIUS || self.ball_pos.y >= SCREEN_HEIGHT - BALL_RADIUS {
            self.ball_vel.y *= -1.0;
            self.ball_pos.y = self.ball_pos.y.clamp(
                BALL_RADIUS, 
                SCREEN_HEIGHT - BALL_RADIUS
            );
            let _ = self.sound2.play_detached(ctx);
        }

        // Ball bouncing off right wall
        if self.ball_pos.x >= SCREEN_WIDTH - BALL_RADIUS {
            self.ball_vel.x *= -1.0;
            self.ball_pos.x = SCREEN_WIDTH - BALL_RADIUS;
            let _ = self.sound2.play_detached(ctx);
        }

        // Ball passing left wall (reset)
        if self.ball_pos.x <= BALL_RADIUS {
            self.reset_ball();
            self.score = 0;
            let _ = self.sound3.play_detached(ctx);
        }

        // Update paddle based on current action
        if let Ok(action) = self.action.lock() {
            match *action {
                PaddleAction::Up => {
                    self.paddle_y = (self.paddle_y - PAD_VELOCITY * dt)
                        .max(0.0);
                }
                PaddleAction::Down => {
                    self.paddle_y = (self.paddle_y + PAD_VELOCITY * dt)
                        .min(SCREEN_HEIGHT - PAD_LENGTH);
                }
                PaddleAction::Stay => {}
            }
        }

        // Check paddle collision with improved logic
        let paddle_rect = graphics::Rect::new(
            PAD_LEFT_EDGE,
            self.paddle_y,
            PAD_WIDTH,
            PAD_LENGTH
        );

        if self.check_collision(self.ball_pos, BALL_RADIUS, &paddle_rect) {
            // Check if collision is from top or bottom of paddle
            if self.ball_pos.y < self.paddle_y || self.ball_pos.y > (self.paddle_y + PAD_LENGTH) {
                // Top/bottom collision - reverse vertical velocity
                self.ball_vel.y *= -1.0;
                self.ball_pos.y += self.ball_vel.y * dt;
            } else if self.check_collision(self.ball_pos, BALL_RADIUS, &paddle_rect) {
                // Side collision - reverse horizontal velocity
                self.ball_vel.x *= -1.0;
                self.ball_pos.x += self.ball_vel.x * dt;
                
                // Increment score
                self.score += 1;
                let _ = self.sound1.play_detached(ctx);
            }
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
            self.ball_pos,
            BALL_RADIUS,
            0.1,
            graphics::Color::WHITE,
        )?;

        // Draw paddle
        let pad = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::stroke(1.0),
            graphics::Rect::new(
                PAD_LEFT_EDGE,
                self.paddle_y,
                PAD_WIDTH,
                PAD_LENGTH
            ),
            graphics::Color::WHITE,
        )?;

        // Draw score
        let mut score_text = graphics::Text::new(format!("Score: {}", self.score));
        score_text.set_scale(30.0);

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
    let cb = ContextBuilder::new("pong", "enstn")
        .window_mode(conf::WindowMode::default()
            .dimensions(SCREEN_WIDTH, SCREEN_HEIGHT)
            .transparent(true))
        .window_setup(conf::WindowSetup::default()
            .title("neuropong"))
        .add_resource_path("res");
        

    let (mut ctx, event_loop) = cb.build()?;
    let state = GameInstance::new(&mut ctx, action, callback)?;
    
    event::run(ctx, event_loop, state)
}