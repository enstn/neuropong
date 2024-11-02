use pyo3::{prelude::*, types::PyModule};
use std::sync::{Arc,Mutex};
// use pyo3::types::IntoPyDict;

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
pub struct PyGameState {
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
impl PyGameState {
    #[new]
    fn new() -> Self {
        PyGameState {
            ball_x: 0.0,
            ball_y: 0.0,
            paddle_y: 0.0,
            score: 0,
        }
    }
}

#[pyclass]
pub struct PyPongController {
    state: Arc<Mutex<PyGameState>>,
    action: Arc<Mutex<PaddleAction>>,
}

#[pymethods]
impl PyPongController {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyPongController {
            state: Arc::new(Mutex::new(PyGameState::new())),
            action: Arc::new(Mutex::new(PaddleAction::Stay)),
        })
    }

    fn get_state(&self) -> PyResult<PyGameState> {
        Ok(self.state
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .clone())
    }

    fn set_action(&self, action: PaddleAction) -> PyResult<()> {
        *self.action
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))? = action;
        Ok(())
    }

    fn start_game(&self) -> PyResult<()> {
        let state = self.state.clone();
        let action = self.action.clone();
        
        std::thread::spawn(move || {
            crate::game::run_game(state, action);
        });
        
        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn neuropong(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("PaddleAction", py.get_type::<PaddleAction>())?;
    m.add("PyGameState", py.get_type::<PyGameState>())?;
    m.add("PongController", py.get_type::<PyPongController>())?;
    Ok(())
}

pub mod game;