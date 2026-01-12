import numpy as np
import random
import time
import os


# --- The Environment (Same as before) ---
class CheeseBoard:
    def __init__(self, size=5):
        self.size = size
        self.traps = [(1, 1), (2, 2), (3, 1), (0, 4), (4, 0)]

    def reset(self):
        self.mouse_pos = (0, 0)  # Fixed Start
        while True:
            r, c = random.randint(0, 4), random.randint(0, 4)
            if (r, c) != (0, 0) and (r, c) not in self.traps:
                self.cheese_pos = (r, c)  # Random Cheese
                break
        return (self.mouse_pos[0],  self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1])

    def step(self, action):
        r, c = self.mouse_pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.size - 1, c + 1)

        self.mouse_pos = (r, c)
        state = (self.mouse_pos[0], self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1])

        if self.mouse_pos == self.cheese_pos: return state, 20, True
        if self.mouse_pos in self.traps: return state, -10, True
        return state, -1, False

    def render(self):
        print("\n" * 2)
        for r in range(self.size):
            row = "".join([" 🐭 " if (r, c) == self.mouse_pos else " 🧀 " if (r, c) == self.cheese_pos else " ❌ " if (r,
                                                                                                                    c) in self.traps else " ⬜ "
                           for c in range(self.size)])
            print(row)


# --- Logic for Saving/Loading ---
MODEL_FILE = "mouse_brain.npy"
env = CheeseBoard()

if os.path.exists(MODEL_FILE):
    print("🧠 Memory found! Loading the pre-trained brain...")
    q_table = np.load(MODEL_FILE)
else:
    print("👶 No memory found. Training the mouse (this may take a moment)...")
    q_table = np.zeros((5, 5, 5, 5, 4))
    lr, gamma, epsilon = 0.2, 0.9, 0.1

    for episode in range(20000):
        state = env.reset()
        done = False
        while not done:
            action = random.randint(0, 3) if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])
            next_state, reward, done = env.step(action)
            old_q = q_table[state][action]
            next_max = np.max(q_table[next_state])
            q_table[state][action] = old_q + lr * (reward + gamma * next_max - old_q)
            state = next_state

    # SAVE the brain after training
    np.save(MODEL_FILE, q_table)
    print(f"✅ Training complete. Brain saved to {MODEL_FILE}")

# --- Demonstration ---
print("\nWatch the mouse use its memory:")
state = env.reset()
done = False
while not done:
    env.render()
    time.sleep(0.5)
    action = np.argmax(q_table[state])
    state, _, done = env.step(action)
env.render()
print("🎯 Found the cheese! Stopping.")