import neuropong
import threading
import time

def main():
    print("Initializing Pong controller...")
    controller = neuropong.PongController()

    def on_state_update(state):
        # AI logic
        if state.ball_y > state.paddle_y + 50:  # +50 is half paddle length
            controller.set_action(neuropong.PaddleAction.DOWN)
        elif state.ball_y < state.paddle_y + 50:
            controller.set_action(neuropong.PaddleAction.UP)
        else:
            controller.set_action(neuropong.PaddleAction.STAY)

    # Register our callback
    controller.register_callback(on_state_update)
    
    print("Starting game...")
    try:
        controller.start_game()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()