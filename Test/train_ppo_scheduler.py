from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from telescope_env import TelescopeSchedulingEnv

# Create the environment
env = TelescopeSchedulingEnv()

# Optional: Check environment for compatibility
check_env(env, warn=True)

# Initialize the PPO model
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./logs"
)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_telescope_scheduler")

print("Model trained and saved as 'ppo_telescope_scheduler.zip'")
