// because ggez requires main thread to render 

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

#[pyclass]
#[derive(Clone, Copy, PartialEq)]
pub enum PaddleAction {
    Up,
    Down,
    Stay,
}

#[pymethods]
impl PaddleAction {
    #[classattr]
    const UP: Self = Self::Up;
    #[classattr]
    const DOWN: Self = Self::Down;
    #[classattr]
    const STAY: Self = Self::Stay;
}

#[pyclass]
#[derive(Clone)]
pub struct GameState {
    #[pyo3(get)]
    pub ball_x: f32,
    #[pyo3(get)]
    pub ball_y: f32,
    #[pyo3(get)]
    pub paddle_y: f32,
    #[pyo3(get)]
    pub score: i32,
}

#[pymethods]
impl GameState {
    #[new]
    fn new() -> Self {
        GameState {
            ball_x: 400.0,
            ball_y: 300.0,
            paddle_y: 250.0,
            score: 0,
        }
    }
}

#[pyclass]
pub struct PongController {
    action: Arc<Mutex<PaddleAction>>,
    callback: Arc<Mutex<Option<PyObject>>>,
}

#[pymethods]
impl PongController {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PongController {
            action: Arc::new(Mutex::new(PaddleAction::Stay)),
            callback: Arc::new(Mutex::new(None)),
        })
    }

    fn set_action(&self, action: PaddleAction) -> PyResult<()> {
        *self.action.lock().unwrap() = action;
        Ok(())
    }

    fn register_callback(&self, callback: PyObject) -> PyResult<()> {
        *self.callback.lock().unwrap() = Some(callback);
        Ok(())
    }

    fn start_game(&self) -> PyResult<()> {
        let action = self.action.clone();
        let callback = self.callback.clone();
        
        // Run the game on the main thread
        crate::game::run_game(action, callback)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pymodule]
fn neuropong(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PaddleAction>()?;
    m.add_class::<GameState>()?;
    m.add_class::<PongController>()?;
    Ok(())
}

pub mod game;