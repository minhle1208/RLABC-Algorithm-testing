import argparse
import logging
import os
import platform
import sys
import tempfile
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from libscratch.Agents.DDPG import DDPGAgent
from libscratch.Environment import ACCElegantEnvironment
from libscratch.Utils import setLogger


tempfile.tempdir = "/tmp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Allowed values: "DDPG", "SAC", "PPO", "TD3".
DEFAULT_AGENT_TYPE = "SAC"


def get_conda_env_name() -> Optional[str]:
    """Return the current active Conda environment name, if present."""
    return os.environ.get("CONDA_DEFAULT_ENV")


class RLConfig:
    """
    Configuration and orchestration class for RL training.
    """

    def __init__(self, agent_type: str = DEFAULT_AGENT_TYPE):
        # Platform and path configuration
        self.os_type = platform.system().lower()
        self._setup_platform_paths()

        # Elegant execution configuration
        self.override_dynamic_command = True
        self.overridden_command = "elegant"

        # Environment configuration
        self.stage = None
        self.n_bins = 5
        self.reset_specific_keys_bool = False
        self.init_num_particles = 1000
        self.input_beamline_file = "machine.lte"
        self.input_beam_file = "track"  # without ".ele"
        self.output_beamline_file = "updated_machine.lte"
        self.beamline_name = "machine"

        # General training configuration
        self.seed = 0
        self.cpu = False
        self.max_steps = None
        self.results_path = "results/"
        self.run_elegant_preflight = True
        self.evaluate_episodes = 1
        self.checkpoint_dir = "models/checkpoints"
        self.checkpoint_freq = 10_000

        # Shared DDPG-style configuration
        self.load_model = True
        self.n_episodes = 5000
        self.greedy = 0.5
        self.load_buffer_bool = False
        self.load_buffer_filepath = "DDPG_bigbeamline_buffer.pkl"
        self.alpha = 1e-4
        self.beta = 1e-3
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.max_size = 1_000_000
        self.noise_type = "gaussian"
        self.log_interval = 100
        self.eval_interval = 1000
        self.convert = True
        self.save_buffer_filepath = "rbBig_beamline_Orig.pkl"

        # SAC configuration
        self.sac_policy = "MlpPolicy"
        self.sac_learning_rate = 3e-4
        self.sac_learning_starts = 1_000
        self.sac_train_freq = 1
        self.sac_gradient_steps = 1
        self.sac_ent_coef = "auto"
        self.sac_net_arch = [256, 256]
        self.sac_replay_buffer_path = "models/sac_beamline_replay_buffer.pkl"

        # PPO configuration
        self.ppo_policy = "MlpPolicy"
        self.ppo_learning_rate = 3e-4
        self.ppo_n_steps = 2048
        self.ppo_batch_size = 64
        self.ppo_n_epochs = 10
        self.ppo_gae_lambda = 0.95
        self.ppo_clip_range = 0.2
        self.ppo_ent_coef = 0.0
        self.ppo_vf_coef = 0.5
        self.ppo_max_grad_norm = 0.5
        self.ppo_net_arch = [dict(pi=[256, 256], vf=[256, 256])]
        #self.ppo_replay_buffer_path = "models/ppo_beamline_replay_buffer.pkl"

        # TD3 configuration, aligned with the DDPG baseline where possible
        self.td3_policy = "MlpPolicy"
        self.td3_learning_rate = 3e-4
        self.td3_buffer_size = self.max_size
        self.td3_learning_starts = 1_000
        self.td3_batch_size = self.batch_size
        self.td3_tau = self.tau
        self.td3_gamma = self.gamma
        self.td3_train_freq = (1, "step")
        self.td3_gradient_steps = 1
        self.td3_policy_delay = 2
        self.td3_target_policy_noise = 0.2
        self.td3_target_noise_clip = 0.5
        self.td3_net_arch = [256, 256]
        self.td3_replay_buffer_path = "models/td3_replay_buffer.pkl"

        self.headers = (
            "reward",
            "initial_number_of_particles",
            "number_of_particles",
            "done",
            "itteration",
            "current_element",
            "dict_vars",
        )

        # These values are set by apply_agent_preset().
        self.agent_type = None
        self.total_timesteps = None
        self.tb_file_name = None
        self.logger_file_name = None
        self.save_model_file_name = None

        self.apply_agent_preset(agent_type)
        self.device = self._setup_device()

    def apply_agent_preset(self, agent_type: str):
        """
        Apply the same per-algorithm file names and default timesteps used in the
        separate training scripts.
        """
        agent_type = agent_type.upper()
        valid_agents = {"DDPG", "SAC", "PPO", "TD3"}
        if agent_type not in valid_agents:
            raise ValueError(f"Unsupported agent type: {agent_type}. Expected one of {sorted(valid_agents)}")

        self.agent_type = agent_type

        if agent_type == "SAC":
            self.total_timesteps = 100_000
            self.tb_file_name = "big_beamline_Orig"
            self.logger_file_name = "big_beamline_Orig.csv"
            self.save_model_file_name = "models/sac_beamline"

        elif agent_type == "PPO":
            self.total_timesteps = 100_000
            self.tb_file_name = "big_beamline_Orig_ppo"
            self.logger_file_name = "big_beamline_Orig_ppo.csv"
            self.save_model_file_name = "models/ppo_beamline"

        elif agent_type == "TD3":
            self.total_timesteps = 100_000
            self.tb_file_name = "big_beamline_Orig_td3"
            self.logger_file_name = "big_beamline_Orig_ppo_td3.csv"
            self.save_model_file_name = "models/td3_beamline"

        elif agent_type == "DDPG":
            self.total_timesteps = 100_000
            self.tb_file_name = "big_beamline_Orig_ddpg"
            self.logger_file_name = "big_beamline_Orig_ddpg.csv"
            self.save_model_file_name = "models/ddpg_beamline"

    def _setup_platform_paths(self):
        """Setup platform-specific Elegant/SDDS paths."""
        if self.os_type == "darwin":
            self.elegant_path = "/Users/Downloads/sdds/darwin-x86/"
            self.sdds_path = "/Users/Downloads/sdds/defns.rpn"
        elif self.os_type == "linux":
            self.elegant_path = ""
            self.sdds_path = "defns.rpn"
        elif self.os_type == "windows":
            self.elegant_path = ""
            self.sdds_path = ""
        else:
            raise RuntimeError(f"Unsupported operating system: {self.os_type}")

    def _setup_device(self) -> torch.device:
        """Setup computation device based on availability."""
        if not self.cpu and torch.cuda.is_available():
            return torch.device("cuda:0")
        if not self.cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_binary(self, binary_name: str) -> str:
        """Resolve a binary either from elegant_path or from PATH."""
        if self.elegant_path:
            candidate = os.path.join(self.elegant_path, binary_name)
            if os.path.exists(candidate):
                return candidate
        return binary_name

    def make_checkpoint_callback(self):
        """Create a checkpoint callback for Stable-Baselines3 agents."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return CheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix=self.agent_type.lower(),
        )

    def setup_environment(self):
        """Setup and return the RLABC environment."""
        csv_logger, file_handler = setLogger(
            self.load_model,
            self.logger_file_name,
            self.headers,
        )

        env = ACCElegantEnvironment(
            stage=self.stage,
            n_bins=self.n_bins,
            init_num_particles=self.init_num_particles,
            logger=csv_logger,
            file_handler=file_handler,
            reset_specific_keys_bool=self.reset_specific_keys_bool,
            input_beamline_file=self.input_beamline_file,
            beamline_name=self.beamline_name,
            output_beamline_file=self.output_beamline_file,
            input_beam_file=self.input_beam_file,
            elegant_path=self.elegant_path,
            sddsPath=self.sdds_path,
            override_dynamic_command=self.override_dynamic_command,
            overridden_command=self.overridden_command,
            results_path=self.results_path,
        )

        if self.max_steps is None:
            self.max_steps = env.max_num_of_vars

        return env, csv_logger, file_handler

    def setup_agent(self, env):
        """Create the selected agent."""
        if self.agent_type == "DDPG":
            return DDPGAgent(
                env=env,
                alpha=self.alpha,
                beta=self.beta,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                max_size=self.max_size,
                noise_type=self.noise_type,
                log_interval=self.log_interval,
                eval_interval=self.eval_interval,
                seed=self.seed,
                exp=self.tb_file_name,
                load=self.load_model,
                convert=self.convert,
            )

        os.makedirs("models", exist_ok=True)
        monitored_env = Monitor(env)
        model_path = self.save_model_file_name + ".zip"

        if self.agent_type == "SAC":
            if self.load_model and os.path.exists(model_path):
                print(f"Loading SAC model from {model_path}")
                agent = SAC.load(
                    model_path,
                    env=monitored_env,
                    device="cpu" if self.cpu else "auto",
                )
                if os.path.exists(self.sac_replay_buffer_path):
                    agent.load_replay_buffer(self.sac_replay_buffer_path)
                    print(f"Loaded SAC replay buffer from {self.sac_replay_buffer_path}")
                else:
                    print("No SAC replay buffer found, continuing with weights only.")
                return agent

            print("Creating a new SAC model from scratch")
            return SAC(
                policy=self.sac_policy,
                env=monitored_env,
                learning_rate=self.sac_learning_rate,
                buffer_size=self.max_size,
                learning_starts=self.sac_learning_starts,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.sac_train_freq,
                gradient_steps=self.sac_gradient_steps,
                ent_coef=self.sac_ent_coef,
                policy_kwargs={"net_arch": self.sac_net_arch},
                tensorboard_log="summary",
                verbose=1,
                seed=self.seed,
                device="cpu" if self.cpu else "auto",
            )

        if self.agent_type == "PPO":
            if self.load_model and os.path.exists(model_path):
                print(f"Loading PPO model from {model_path}")
                agent = PPO.load(
                    model_path,
                    env=monitored_env,
                    device="cpu" if self.cpu else "auto",
                )
                if hasattr(agent, "load_replay_buffer") and os.path.exists(self.ppo_replay_buffer_path):
                    agent.load_replay_buffer(self.ppo_replay_buffer_path)
                    print(f"Loaded PPO replay buffer from {self.ppo_replay_buffer_path}")
                return agent

            print("Creating a new PPO model from scratch")
            return PPO(
                policy=self.ppo_policy,
                env=monitored_env,
                learning_rate=self.ppo_learning_rate,
                n_steps=self.ppo_n_steps,
                batch_size=self.ppo_batch_size,
                n_epochs=self.ppo_n_epochs,
                gamma=self.gamma,
                gae_lambda=self.ppo_gae_lambda,
                clip_range=self.ppo_clip_range,
                ent_coef=self.ppo_ent_coef,
                vf_coef=self.ppo_vf_coef,
                max_grad_norm=self.ppo_max_grad_norm,
                policy_kwargs={"net_arch": self.ppo_net_arch},
                tensorboard_log="summary",
                verbose=1,
                seed=self.seed,
                device="cpu" if self.cpu else "auto",
            )

        if self.agent_type == "TD3":
            if self.load_model and os.path.exists(model_path):
                print(f"Loading TD3 model from {model_path}")
                agent = TD3.load(
                    model_path,
                    env=monitored_env,
                    device="cpu" if self.cpu else "auto",
                )
                if os.path.exists(self.td3_replay_buffer_path):
                    agent.load_replay_buffer(self.td3_replay_buffer_path)
                    print(f"Loaded TD3 replay buffer from {self.td3_replay_buffer_path}")
                else:
                    print("No TD3 replay buffer found, continuing without it.")
                return agent

            print("Creating a new TD3 model from scratch")
            return TD3(
                policy=self.td3_policy,
                env=monitored_env,
                learning_rate=self.td3_learning_rate,
                buffer_size=self.td3_buffer_size,
                learning_starts=self.td3_learning_starts,
                batch_size=self.td3_batch_size,
                tau=self.td3_tau,
                gamma=self.td3_gamma,
                train_freq=self.td3_train_freq,
                gradient_steps=self.td3_gradient_steps,
                policy_delay=self.td3_policy_delay,
                target_policy_noise=self.td3_target_policy_noise,
                target_noise_clip=self.td3_target_noise_clip,
                policy_kwargs={"net_arch": self.td3_net_arch},
                tensorboard_log="summary",
                verbose=1,
                seed=self.seed,
                device="cpu" if self.cpu else "auto",
            )

        raise ValueError(f"Unsupported agent type: {self.agent_type}")

    def load_buffer(self, agent):
        """Load replay buffer for the custom DDPG agent."""
        if self.agent_type != "DDPG":
            return
        if os.path.exists(self.load_buffer_filepath):
            agent.buffer.load(self.load_buffer_filepath)
            print(f"Loaded replay buffer from {self.load_buffer_filepath}")

    def save_buffer(self, agent):
        """Save replay buffer for the custom DDPG agent."""
        if self.agent_type != "DDPG":
            return
        agent.buffer.save(self.save_buffer_filepath)
        print(f"Saved replay buffer to {self.save_buffer_filepath}")

    def load_agent_weights(self, agent, step: Optional[int] = None):
        """Load custom DDPG checkpoint weights."""
        if self.agent_type != "DDPG":
            return
        if self.load_model:
            if step is not None:
                agent.load_models(step)
                print(f"Loaded agent weights from step {step}")
            else:
                agent.load_models()
                print("Loaded latest agent weights")
        else:
            print("Training from scratch")

    def check_elegant_setup(self) -> bool:
        """
        Run a simulator preflight:
        - checks input files;
        - builds the environment;
        - calls reset();
        - calls one zero-action step();
        - checks that result files were produced.
        """
        print("=== Elegant preflight check ===")

        lte_file = self.input_beamline_file
        ele_file = f"{self.input_beam_file}.ele"

        if not os.path.exists(lte_file):
            raise FileNotFoundError(f"Missing lattice file: {lte_file}")
        if not os.path.exists(ele_file):
            raise FileNotFoundError(f"Missing Elegant input file: {ele_file}")

        print(f"Found input files: {lte_file}, {ele_file}")

        if self.override_dynamic_command:
            print(f"Simulation command: {self.overridden_command} {self.input_beam_file}.ele")
        else:
            elegant_bin = self._resolve_binary("elegant")
            print(
                f"Simulation command: {elegant_bin} -rpnDefns={self.sdds_path} "
                f"{self.input_beam_file}.ele"
            )

        env, _, file_handler = self.setup_environment()
        try:
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            print(f"reset() OK, observation shape = {obs.shape}")

            zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(zero_action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            print(f"step() OK, next observation shape = {next_obs.shape}")
            print(f"reward = {reward:.6f}")
            print(f"terminated = {terminated}, truncated = {truncated}")
            print(f"output_file = {info.get('output_file')}")
            print(f"number_of_particles = {info.get('number_of_particles')}")

            if not os.path.exists(self.results_path):
                raise RuntimeError(f"Results folder was not created: {self.results_path}")

            produced_files = sorted(os.listdir(self.results_path))
            print(f"Produced {len(produced_files)} files in {self.results_path}")
            if produced_files:
                print("Sample result files:", produced_files[:10])

            if len(produced_files) == 0:
                raise RuntimeError(
                    "Elegant run completed but no files were produced in the results folder."
                )
        finally:
            if file_handler is not None:
                file_handler.close()

        print("=== Elegant preflight passed ===")
        return True

    def train(self, agent, env):
        """Run training with the current configuration."""
        if self.agent_type == "DDPG":
            print(f"Starting DDPG training for {self.n_episodes} episodes...")
            scores = agent.train(
                n_episodes=self.n_episodes,
                max_steps=self.max_steps,
                greedy=self.greedy,
            )
            print("DDPG training completed")
            return scores

        print(f"=== STARTING {self.agent_type} LEARNING ===")
        print(f"Additional timesteps this run: {self.total_timesteps}")

        checkpoint_callback = self.make_checkpoint_callback()

        try:
            agent.learn(
                total_timesteps=self.total_timesteps,
                tb_log_name=self.tb_file_name,
                reset_num_timesteps=False,
                callback=checkpoint_callback,
            )
        except KeyboardInterrupt:
            print(f"Training interrupted. Saving {self.agent_type} model...")

        agent.save(self.save_model_file_name)
        print(f"{self.agent_type} model saved to {self.save_model_file_name}.zip")

        if self.agent_type == "SAC" and hasattr(agent, "save_replay_buffer"):
            agent.save_replay_buffer(self.sac_replay_buffer_path)
            print(f"SAC replay buffer saved to {self.sac_replay_buffer_path}")

        if self.agent_type == "TD3" and hasattr(agent, "save_replay_buffer"):
            agent.save_replay_buffer(self.td3_replay_buffer_path)
            print(f"TD3 replay buffer saved to {self.td3_replay_buffer_path}")

        return []

    def evaluate(self, agent, env, episodes: int = 1):
        """Evaluate the trained agent."""
        print("Evaluating the agent...")

        if self.agent_type == "DDPG":
            agent.evaluate(episodes=episodes)
            return

        if self.agent_type in {"SAC", "PPO", "TD3"}:
            for ep in range(episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0.0
                info = {}

                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                print(
                    f"{self.agent_type} evaluation episode {ep + 1}: "
                    f"reward={episode_reward:.6f}, "
                    f"final_particles={info.get('number_of_particles')}, "
                    f"last_output={info.get('output_file')}"
                )
            return

        raise ValueError(f"Unsupported agent type: {self.agent_type}")

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = [
            "RL Configuration:",
            f"  Agent: {self.agent_type}",
            f"  Device: {self.device}",
            f"  Results Path: {self.results_path}",
            f"  Elegant Override Command: {self.override_dynamic_command}",
            f"  Beamline File: {self.input_beamline_file}",
            f"  Beam File Prefix: {self.input_beam_file}",
        ]

        if self.agent_type == "DDPG":
            lines.extend(
                [
                    f"  Episodes: {self.n_episodes}",
                    f"  Max Steps: {self.max_steps}",
                    f"  Actor LR: {self.alpha}",
                    f"  Critic LR: {self.beta}",
                    f"  Buffer Size: {self.max_size}",
                    f"  Load Buffer: {self.load_buffer_bool}",
                ]
            )
        elif self.agent_type == "SAC":
            lines.extend(
                [
                    f"  Total Timesteps: {self.total_timesteps}",
                    f"  Batch Size: {self.batch_size}",
                    f"  Gamma: {self.gamma}",
                    f"  Tau: {self.tau}",
                    f"  SAC LR: {self.sac_learning_rate}",
                    f"  SAC Policy: {self.sac_policy}",
                    f"  Save Model Path: {self.save_model_file_name}.zip",
                    f"  Replay Buffer Path: {self.sac_replay_buffer_path}",
                ]
            )
        elif self.agent_type == "PPO":
            lines.extend(
                [
                    f"  Total Timesteps: {self.total_timesteps}",
                    f"  PPO Learning Rate: {self.ppo_learning_rate}",
                    f"  PPO n_steps: {self.ppo_n_steps}",
                    f"  PPO Batch Size: {self.ppo_batch_size}",
                    f"  PPO n_epochs: {self.ppo_n_epochs}",
                    f"  Gamma: {self.gamma}",
                    f"  GAE Lambda: {self.ppo_gae_lambda}",
                    f"  Clip Range: {self.ppo_clip_range}",
                    f"  Save Model Path: {self.save_model_file_name}.zip",
                ]
            )
        elif self.agent_type == "TD3":
            lines.extend(
                [
                    f"  Total Timesteps: {self.total_timesteps}",
                    f"  TD3 Learning Rate: {self.td3_learning_rate}",
                    f"  TD3 Batch Size: {self.td3_batch_size}",
                    f"  Gamma: {self.td3_gamma}",
                    f"  Tau: {self.td3_tau}",
                    f"  Policy Delay: {self.td3_policy_delay}",
                    f"  Save Model Path: {self.save_model_file_name}.zip",
                    f"  Replay Buffer Path: {self.td3_replay_buffer_path}",
                ]
            )

        return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC, PPO, TD3, or DDPG on the RLABC environment.")
    parser.add_argument(
        "--agent",
        choices=["DDPG", "SAC", "PPO", "TD3"],
        default=DEFAULT_AGENT_TYPE,
        help="Agent type to train. Default is set by DEFAULT_AGENT_TYPE in the file.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps for SAC/PPO/TD3 for this run.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Train from scratch instead of loading an existing model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip Elegant preflight check.",
    )
    return parser.parse_args()


def main():
    print("in python main")

    args = parse_args()

    conda_env_name = get_conda_env_name()
    conda_env_path = os.environ.get("CONDA_PREFIX")

    if conda_env_name:
        print(f"Current Conda environment name: {conda_env_name}")
    if conda_env_path:
        print(f"Current Conda environment path: {conda_env_path}")

    config = RLConfig(agent_type=args.agent)

    if args.timesteps is not None:
        if config.agent_type == "DDPG":
            print("Warning: --timesteps is ignored for custom DDPG. Use n_episodes instead.")
        else:
            config.total_timesteps = args.timesteps

    if args.fresh:
        config.load_model = False

    if args.cpu:
        config.cpu = True
        config.device = config._setup_device()

    if args.no_preflight:
        config.run_elegant_preflight = False

    print(config)
    sys.stdout.flush()

    if config.run_elegant_preflight:
        config.check_elegant_setup()

    env, _, file_handler = config.setup_environment()

    try:
        agent = config.setup_agent(env)
        print("Finished setting up the environment and agent")

        if config.agent_type == "DDPG" and config.load_buffer_bool:
            config.load_buffer(agent)

        if config.agent_type == "DDPG" and config.load_model:
            config.load_agent_weights(agent)

        config.train(agent, env)

        if config.agent_type == "DDPG":
            config.save_buffer(agent)

        config.evaluate(agent, env, episodes=config.evaluate_episodes)

    finally:
        if file_handler is not None:
            file_handler.close()


if __name__ == "__main__":
    main()
