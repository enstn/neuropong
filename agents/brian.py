from brian2 import *
import numpy as np
from collections import deque
import neuropong
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
from datetime import datetime

# Neuron Parameters
TAU_M = 20*ms  # Membrane time constant
TAU_E = 5*ms   # Excitatory synaptic time constant
TAU_I = 10*ms  # Inhibitory synaptic time constant
VT = -50*mV    # Threshold potential
VR = -60*mV    # Reset potential
EL = -49*mV    # Leak potential

# Network Architecture
N_INPUT = 4     # ball_x, ball_y, paddle_y, ball_velocity_y
N_HIDDEN = 1000 # Hidden layer neurons
N_OUTPUT = 3    # Up, Down, Stay

class BrainPongVisualizer:
    def __init__(self, update_interval=1000):  # update_interval in ms
        self.update_interval = update_interval
        
        # Training metrics
        self.scores = []
        self.rewards = []
        self.avg_spike_rates = []
        self.timestamps = []
        
        # Create real-time plotting figure
        plt.ion()  # Enable interactive mode
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Brain Pong Training Metrics')
        
        # Initialize subplot titles
        self.axs[0, 0].set_title('Score History')
        self.axs[0, 1].set_title('Reward History')
        self.axs[1, 0].set_title('Neural Activity')
        self.axs[1, 1].set_title('Average Spike Rates')
        
        # Initialize lines for real-time plots
        self.score_line, = self.axs[0, 0].plot([], [])
        self.reward_line, = self.axs[0, 1].plot([], [])
        self.spike_rate_lines = [self.axs[1, 1].plot([], [], label=f'Layer {i}')[0] 
                               for i in ['Input', 'Hidden', 'Output']]
        self.axs[1, 1].legend()
        
        # Raster plot placeholder
        self.raster_scatter = None
        
        plt.tight_layout()
        
    def update_metrics(self, score, reward, spike_monitors):
        current_time = time.time()
        self.timestamps.append(current_time)
        self.scores.append(score)
        self.rewards.append(reward)
        
        # Calculate spike rates for each layer
        spike_rates = []
        for monitor in spike_monitors:
            # Calculate spikes per second over the last update interval
            recent_spikes = len([t for t in monitor.t/ms 
                               if t > (current_time*1000 - self.update_interval)])
            rate = recent_spikes / (self.update_interval / 1000)  # Convert to Hz
            spike_rates.append(rate)
        
        self.avg_spike_rates.append(spike_rates)
        
        # Update plots if enough time has passed
        if len(self.timestamps) % 10 == 0:  # Update every 10 data points
            self.update_plots(spike_monitors)
    
    def update_plots(self, spike_monitors):
        # Update score history
        self.axs[0, 0].clear()
        self.axs[0, 0].set_title('Score History')
        self.axs[0, 0].plot(self.scores)
        self.axs[0, 0].set_xlabel('Episodes')
        self.axs[0, 0].set_ylabel('Score')
        
        # Update reward history
        self.axs[0, 1].clear()
        self.axs[0, 1].set_title('Reward History')
        self.axs[0, 1].plot(self.rewards)
        self.axs[0, 1].set_xlabel('Steps')
        self.axs[0, 1].set_ylabel('Reward')
        
        # Update raster plot
        self.axs[1, 0].clear()
        self.axs[1, 0].set_title('Neural Activity (Last 100ms)')
        colors = ['r', 'b', 'g']
        labels = ['Input', 'Hidden', 'Output']
        
        for idx, monitor in enumerate(spike_monitors):
            recent_spikes = [(t/ms, i) for i, t in zip(monitor.i, monitor.t) 
                           if t/ms > (time.time()*1000 - 100)]
            if recent_spikes:
                times, indices = zip(*recent_spikes)
                self.axs[1, 0].scatter(times, indices, 
                                     c=colors[idx], label=labels[idx], s=1)
        
        self.axs[1, 0].set_xlabel('Time (ms)')
        self.axs[1, 0].set_ylabel('Neuron Index')
        self.axs[1, 0].legend()
        
        # Update spike rates
        self.axs[1, 1].clear()
        self.axs[1, 1].set_title('Average Spike Rates')
        for i in range(3):
            rates = [sr[i] for sr in self.avg_spike_rates]
            self.axs[1, 1].plot(rates, label=labels[i])
        self.axs[1, 1].set_xlabel('Steps')
        self.axs[1, 1].set_ylabel('Spikes/second')
        self.axs[1, 1].legend()
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def save_metrics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_metrics_{timestamp}.npz"
        np.savez(filename,
                 scores=np.array(self.scores),
                 rewards=np.array(self.rewards),
                 spike_rates=np.array(self.avg_spike_rates),
                 timestamps=np.array(self.timestamps))
        print(f"Metrics saved to {filename}")

class BrianPongAgent:
    def __init__(self):
        # Define neuron equations
        self.neuron_eqs = '''
        dv/dt  = (ge+gi-(v-EL))/TAU_M : volt (unless refractory)
        dge/dt = -ge/TAU_E : volt
        dgi/dt = -gi/TAU_I : volt
        '''
        
        # Create neuron groups
        self.setup_network()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Learning parameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        
        # State preprocessing
        self.state_normalizer = np.array([800.0, 600.0, 600.0, 500.0])  # Max values
        
        # Initialize visualizer
        self.visualizer = BrainPongVisualizer()
        
        # Training episodes counter
        self.episode_count = 0
        self.total_reward = 0
        
        # Previous ball_y for velocity calculation
        self.prev_ball_y = None
        self.prev_time = None
        
    def setup_network(self):
        # Input layer (sensory neurons)
        self.input_layer = NeuronGroup(N_INPUT, self.neuron_eqs,
                                     threshold='v>VT', reset='v=VR',
                                     refractory=5*ms, method='exact')
        
        # Hidden layer
        self.hidden_layer = NeuronGroup(N_HIDDEN, self.neuron_eqs,
                                      threshold='v>VT', reset='v=VR',
                                      refractory=5*ms, method='exact')
        
        # Output layer
        self.output_layer = NeuronGroup(N_OUTPUT, self.neuron_eqs,
                                      threshold='v>VT', reset='v=VR',
                                      refractory=5*ms, method='exact')
        
        # Initialize membrane potentials
        self.input_layer.v = 'VR + rand() * (VT - VR)'
        self.hidden_layer.v = 'VR + rand() * (VT - VR)'
        self.output_layer.v = 'VR + rand() * (VT - VR)'
        
        # Create synaptic connections
        we = (60*0.27/10)*mV  # Excitatory weight
        wi = (-20*4.5/10)*mV  # Inhibitory weight
        
        # Input to hidden connections
        self.ih_synapses = Synapses(self.input_layer, self.hidden_layer,
                                   model='w : volt',
                                   on_pre='ge += w')
        self.ih_synapses.connect(p=0.3)  # 30% connectivity
        self.ih_synapses.w = 'we * rand()'
        
        # Hidden to output connections
        self.ho_synapses = Synapses(self.hidden_layer, self.output_layer,
                                   model='w : volt',
                                   on_pre='ge += w')
        self.ho_synapses.connect(p=0.3)
        self.ho_synapses.w = 'we * rand()'
        
        # Create spike monitors
        self.input_monitor = SpikeMonitor(self.input_layer)
        self.hidden_monitor = SpikeMonitor(self.hidden_layer)
        self.output_monitor = SpikeMonitor(self.output_layer)
    
    def calculate_velocity(self, state):
        """Calculate ball velocity from consecutive states"""
        current_time = time.time()
        velocity = 0.0
        
        if self.prev_ball_y is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                velocity = (state.ball_y - self.prev_ball_y) / dt
        
        self.prev_ball_y = state.ball_y
        self.prev_time = current_time
        
        return velocity
    
    def preprocess_state(self, state):
        """Normalize the game state"""
        velocity = self.calculate_velocity(state)
        return np.array([
            state.ball_x,
            state.ball_y,
            state.paddle_y,
            velocity
        ]) / self.state_normalizer
    
    def select_action(self, state):
        """Run network simulation and select action based on output layer activity"""
        # Reset spike monitors
        self.input_monitor.reset()
        self.output_monitor.reset()
        
        # Set input layer activity based on state
        normalized_state = self.preprocess_state(state)
        self.input_layer.v = VR + normalized_state * (VT - VR)
        
        # Run simulation for a short period
        run(50*ms)
        
        # Count spikes in output layer
        spike_counts = np.zeros(N_OUTPUT)
        for i, t in zip(self.output_monitor.i, self.output_monitor.t):
            spike_counts[i] += 1
        
        # Add exploration (epsilon-greedy)
        epsilon = max(0.01, 0.1 - 0.001 * self.episode_count)  # Decay exploration
        if np.random.random() < epsilon:
            action_idx = np.random.randint(N_OUTPUT)
        else:
            action_idx = np.argmax(spike_counts)
        
        # Convert to PaddleAction
        actions = [
            neuropong.PaddleAction.UP,
            neuropong.PaddleAction.DOWN,
            neuropong.PaddleAction.STAY
        ]
        return actions[action_idx]
    
    def update_weights(self, state, action, reward, next_state):
        """Update synaptic weights using reward prediction error"""
        # Store experience in replay buffer
        self.replay_buffer.append((state, action, reward, next_state))
        
        # Sample random batch from replay buffer
        if len(self.replay_buffer) >= 32:
            batch = np.random.choice(self.replay_buffer, size=32)
            
            for exp in batch:
                state, action, reward, next_state = exp
                
                # Compute target Q-value
                target = reward + self.gamma * self.get_max_q_value(next_state)
                
                # Update weights using eligibility traces
                self.ih_synapses.w += self.learning_rate * (target - self.get_q_value(state, action))
                self.ho_synapses.w += self.learning_rate * (target - self.get_q_value(state, action))
        
        # Update total reward and visualization
        self.total_reward += reward
        self.visualizer.update_metrics(
            self.total_reward,
            reward,
            [self.input_monitor, self.hidden_monitor, self.output_monitor]
        )
    
    def get_q_value(self, state, action):
        """Estimate Q-value for state-action pair"""
        normalized_state = self.preprocess_state(state)
        self.input_layer.v = VR + normalized_state * (VT - VR)
        run(50*ms)
        
        spike_counts = np.zeros(N_OUTPUT)
        for i, t in zip(self.output_monitor.i, self.output_monitor.t):
            spike_counts[i] += 1
        
        return spike_counts[self.action_to_index(action)]
    
    def get_max_q_value(self, state):
        """Get maximum Q-value for state"""
        normalized_state = self.preprocess_state(state)
        self.input_layer.v = VR + normalized_state * (VT - VR)
        run(50*ms)
        
        spike_counts = np.zeros(N_OUTPUT)
        for i, t in zip(self.output_monitor.i, self.output_monitor.t):
            spike_counts[i] += 1
        
        return np.max(spike_counts)
    
    def action_to_index(self, action):
        """Convert PaddleAction to index"""
        if action == neuropong.PaddleAction.UP:
            return 0
        elif action == neuropong.PaddleAction.DOWN:
            return 1
        return 2
    
    def on_episode_end(self):
        """Called when an episode ends (ball is missed)"""
        self.episode_count += 1
        print(f"Episode {self.episode_count} ended. Score: {self.total_reward}")
        self.total_reward = 0

def main():
    print("Initializing Brian2 Pong Agent...")
    controller = neuropong.PongController()
    agent = BrianPongAgent()
    
    # Previous state for computing velocity
    prev_state = None
    prev_score = 0
    
    def on_state_update(state):
        nonlocal prev_state, prev_score
        
        # Detect episode end (ball missed)
        if prev_state and state.score < prev_state.score:
            agent.on_episode_end()
        
        # Compute reward
        reward = 0.0
        if prev_state:
            # Positive reward for scoring
            if state.score > prev_score:
                reward = 1.0
            # Small negative reward for missing
            elif state.score < prev_score:
                reward = -0.5
            # Small positive reward for keeping the ball in play
            else:
                reward = 0.1
        
        # Update network and select action
        if prev_state:
            agent.update_weights(prev_state, None, reward, state)
        action = agent.select_action(state)
        controller.set_action(action)
        
        prev_state = state
        prev_score = state.score
    
    # Register callback
    controller.register_callback(on_state_update)
    
    print("Starting game...")
    try:
        controller.start_game()
    except KeyboardInterrupt:
        print("\nSaving metrics and exiting...")
        agent.visualizer.save_metrics()

if __name__ == "__main__":
    main()