use::pyo3::prelude::*;
use ggez::*;
use ggez::graphics::Rect;
use ggez::input::keyboard;
// use ggez::audio;
// use ggez::audio::SoundSource;

use crate::testagent::{TestAgent,Action};
// use crate::agent1::{Agent1, Action};

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

const SCORE_FONT_SIZE: f32 = 30.0;

struct Pad {
    rect: Rect,
}

impl Pad {
    pub fn new() -> Self {
        Pad {
            rect: ggez::graphics::Rect::new(
                PAD_LEFT_EDGE,
                SCREEN_HEIGHT_MID - (PAD_LENGTH / 2.0),
                PAD_WIDTH,
                PAD_LENGTH,
            ),
        }
    }
}

struct Ball {
    pos: mint::Point2<f32>,
    vel: mint::Vector2<f32>,
}

impl Ball {
    pub fn new() -> Self {
        Ball {
            pos: mint::Point2{x: SCREEN_WIDTH_MID - BALL_RADIUS_MID, y: SCREEN_HEIGHT_MID - BALL_RADIUS_MID},
            vel: mint::Vector2{x: BALL_VELOCITY, y: BALL_VELOCITY},
        }
    }
}

struct State {
    ball: Ball,
    pad: Pad,
    score: u32,
    highscore: u32,
    agent: Option<TestAgent>,
    // sound: audio::Source,
}

impl State {
    pub fn new(_ctx: &mut Context, agent: Option<TestAgent>) -> Self {
        
        // let sound_result = audio::Source::new(ctx, "/res/sound.ogg");
        // let sound = match sound_result {
        //     Ok(s) => s,
        //     Err(e) => return Err(GameError::ResourceLoadError(format!("Failed to load sound resources: {}", e))),
        // };

        State {
            ball: Ball::new(),
            pad: Pad::new(),
            score: 0,
            highscore: 0, 
            agent,
        }
    }
    
    pub fn reset(&mut self) {
        self.ball.pos = mint::Point2{x: SCREEN_WIDTH_MID - BALL_RADIUS_MID, y: SCREEN_HEIGHT_MID - BALL_RADIUS_MID};
        self.ball.vel = mint::Vector2{x: BALL_VELOCITY, y: BALL_VELOCITY};
        self.pad.rect.y = SCREEN_HEIGHT_MID - (PAD_LENGTH / 2.0);
        self.score = 0;
        println!("New game iteration.");
    }

    pub fn gogetball(&mut self) {
        self.ball.pos = mint::Point2{x: SCREEN_WIDTH_MID - BALL_RADIUS_MID, y: SCREEN_HEIGHT_MID - BALL_RADIUS_MID};
        self.ball.vel = mint::Vector2{x: BALL_VELOCITY, y: BALL_VELOCITY};
    }
}

// TODO: this has to be modified to fix edge bug!
fn check_collision (ball_pos: mint::Point2<f32>, ball_radius: f32, rect: &ggez::graphics::Rect) -> bool {
    let closest_x = ball_pos.x.min(rect.x + rect.w).max(rect.x);
    let closest_y = ball_pos.y.min(rect.y + rect.h).max(rect.y);

    let distance_x = ball_pos.x - closest_x;
    let distance_y = ball_pos.y - closest_y;

    if (distance_x.powi(2) + distance_y.powi(2)) < ball_radius.powi(2) && ball_pos.x > PAD_LEFT_EDGE + PAD_WIDTH / 2.0 {
        return true;
    }

    return false;
}



impl ggez::event::EventHandler for State {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let delta_time = ctx.time.delta().as_secs_f32();

        // ball movement
        self.ball.pos = mint::Point2 {
            x: self.ball.pos.x + self.ball.vel.x * delta_time,
            y: self.ball.pos.y + self.ball.vel.y * delta_time
        };

        // ball collisions
        if self.ball.pos.x > SCREEN_WIDTH {
            self.ball.vel.x *= -1.0;
            self.ball.pos.x += self.ball.vel.x * delta_time;
            self.ball.pos.y += self.ball.vel.y * delta_time;
        }
        if self.ball.pos.y > SCREEN_HEIGHT || self.ball.pos.y < 0.0 {
            self.ball.vel.y *= -1.0;
            self.ball.pos.x += self.ball.vel.x * delta_time;
            self.ball.pos.y += self.ball.vel.y * delta_time;

        }

        // pad movement w/ agent
        if let Some(ref agent) = self.agent {
            match agent.choose_action() {
                Action::Up => {
                    if self.pad.rect.y > 0.0 {
                        self.pad.rect.y -= PAD_VELOCITY * delta_time;
                    }
                }
                Action::Down => {
                    if self.pad.rect.y < (SCREEN_HEIGHT - PAD_LENGTH) {
                        self.pad.rect.y += PAD_VELOCITY * delta_time;
                    }
                }
                Action::Stay => {
                    // nothing should happen of course
                }
            } 
        } else {
            // pad movement w/ key inputs
            if ctx.keyboard.is_key_pressed( keyboard::KeyCode::Up) {
                if self.pad.rect.y > 0.0 {
                    self.pad.rect.y -= PAD_VELOCITY * delta_time;
                }
            }
            if ctx.keyboard.is_key_pressed( keyboard::KeyCode::Down) {
                if self.pad.rect.y < (SCREEN_HEIGHT - PAD_LENGTH) {
                    self.pad.rect.y += PAD_VELOCITY * delta_time;
                }
            }
        }

        // ball & pad collisions
        if check_collision(mint::Point2{x: self.ball.pos.x, y: self.ball.pos.y}, BALL_RADIUS, &self.pad.rect) {
            if self.ball.pos.y < self.pad.rect.y || self.ball.pos.y > (self.pad.rect.y + PAD_LENGTH) {
                self.ball.vel.y *= -1.0;
                self.ball.pos.y += self.ball.vel.y * delta_time;
            } else if check_collision(mint::Point2{x: self.ball.pos.x, y: self.ball.pos.y}, BALL_RADIUS, &self.pad.rect) {
                self.ball.vel.x *= -1.0;
                self.ball.pos.x += self.ball.vel.x * delta_time;
                self.score += 1;
                if self.score > self.highscore {
                    self.highscore = self.score;
                }
            }
        } 

        // score keeping & reset
        if self.ball.pos.x <= 0.0 {
            self.reset();
        }

        // edge cases 
        if self.ball.pos.x > SCREEN_WIDTH || self.ball.pos.y < 0.0 || self.ball.pos.y > SCREEN_HEIGHT {
            self.gogetball();
            println!("Oops, ball went out of bound!")
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        // create a canvas first to draw on
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);

        // make the ball mesh
        let ball = graphics::Mesh::new_circle(
            ctx, // sagt ggez wohin er malen soll - hier also auf unseren context
            graphics::DrawMode::fill(), // mal modus für unser ball
            self.ball.pos,
            BALL_RADIUS,
            0.1, // tolerance (idk) 
            graphics::Color::WHITE,
        )?;

        // make the pad mesh
        let pad = graphics::Mesh::new_rectangle(
                ctx,
                graphics::DrawMode::stroke(1.0),
                self.pad.rect,
                graphics::Color::WHITE,
            )?;
        
        let mut score = ggez::graphics::Text::new(format!("Score: {}", self.score));
        score.set_scale(SCORE_FONT_SIZE);
        let score_dimensions = score.measure(ctx);
        let score_width = score_dimensions.unwrap().x;

        let mut highscore = ggez::graphics::Text::new(format!("Highscore: {}", self.highscore));
        highscore.set_scale(SCORE_FONT_SIZE);
        let highscore_dimensions = highscore.measure(ctx);
        let highscore_width = highscore_dimensions.unwrap().x;

        // set the params for drawing (this gets the position done)
        let draw_parameters = graphics::DrawParam::default();
        
        draw_parameters.dest(self.ball.pos);
        canvas.draw(&ball, draw_parameters);

        draw_parameters.dest(mint::Point2{x: self.pad.rect.x, y: self.pad.rect.y}); // pad update
        canvas.draw(&pad, draw_parameters);
        
        canvas.draw(&score, mint::Point2{x: SCREEN_WIDTH - score_width, y: 2.0});
        canvas.draw(&highscore, mint::Point2{x: SCREEN_WIDTH - highscore_width, y: 32.0});
        
        // I like it, picasso
        let _ = canvas.finish(ctx);
        Ok(())
    }
}

pub fn main(agent: Option<TestAgent>) {

    let context_builder = ggez::ContextBuilder::new("pingpangpong", "Lai")
        .window_mode(ggez::conf::WindowMode::default().dimensions(SCREEN_WIDTH, SCREEN_HEIGHT).borderless(false));
    let (mut context, event_loop) = context_builder.build().unwrap();
    let state = State::new(&mut context, agent);
    context.gfx.set_window_title("pingpangpong");

    ggez::event::run(context, event_loop, state);
}
