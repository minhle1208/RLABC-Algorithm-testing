import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from libscratch.Environment import ACCElegantEnvironment


# TARGET_NAME = "simple_sbend"
TARGET_NAME = "machine_3b"


TARGET_CONFIGS: Dict[str, Dict[str, str]] = {
    "simple_sbend": {
        "source_lte": "simple_sbend.lte",
        "work_lte": "simple_sbend_finetune_work.lte",
        "track_file_stem": "simple_sbend_track",  # without .ele
        "beamline_name": "machine",
        "output_lte": "updated_simple_sbend_finetune.lte",
    },
    "machine_3b": {
        "source_lte": "machine_3b.lte",
        "work_lte": "machine_3b_finetune_work.lte",
        "track_file_stem": "track_3b",  # without .ele
        "beamline_name": "machine",
        "output_lte": "updated_machine_3b_finetune.lte",
    },
}


# Pretrained source models
PRETRAINED_MODELS = {
    "SAC": "models/sac_beamline.zip",
    "PPO": "models/ppo_beamline.zip",
    "TD3": "models/td3_beamline.zip",
}

FINETUNE_TIMESTEPS = {
    "SAC": 10_000,
    "PPO": 10_000,
    "TD3": 10_000,
}

ALG_MAP = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}


def get_target_config() -> Dict[str, str]:
    if TARGET_NAME not in TARGET_CONFIGS:
        raise ValueError(
            f"Unknown TARGET_NAME={TARGET_NAME}. "
            f"Available targets: {list(TARGET_CONFIGS.keys())}"
        )
    return TARGET_CONFIGS[TARGET_NAME]


def prepare_lattice_copy() -> str:
    """
    Create a clean working copy of the lattice file.

    This function also removes problematic non-ASCII bytes because some RLABC
    helper functions read lattice files using the Windows default encoding.
    """
    cfg = get_target_config()

    source_path = Path(cfg["source_lte"])
    work_path = Path(cfg["work_lte"])

    if not source_path.exists():
        raise FileNotFoundError(f"Could not find lattice file: {source_path.resolve()}")

    raw = source_path.read_bytes()

    # Some files use SBEN while the parser expects SBEND.
    raw = re.sub(rb"\bSBEN\b", b"SBEND", raw)

    # Keep tabs, newlines, carriage returns, and printable ASCII.
    cleaned = bytes(
        b for b in raw
        if b in (9, 10, 13) or 32 <= b <= 126
    )

    removed = len(raw) - len(cleaned)
    if removed > 0:
        print(f"Sanitized lattice file. Removed {removed} non-ASCII bytes.")

    work_path.write_bytes(cleaned)

    return str(work_path)


def check_required_files() -> None:
    cfg = get_target_config()

    required_files = [
        cfg["source_lte"],
        f"{cfg['track_file_stem']}.ele",
        *PRETRAINED_MODELS.values(),
    ]

    missing = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print("\nMissing required files:")
        for file_path in missing:
            print(f"  - {file_path}")
        raise FileNotFoundError("Some required files are missing.")


def make_env(
    algorithm_name: str,
    init_num_particles: int = 1000,
    mode: str = "finetune",
    monitor: bool = True,
):
    """
    Create target beamline environment for fine-tuning or evaluation.
    """
    cfg = get_target_config()
    work_lte = prepare_lattice_copy()

    results_path = (
        f"results_generalization/"
        f"{TARGET_NAME}/"
        f"{mode}/"
        f"{algorithm_name.lower()}_{init_num_particles}/"
    )
    os.makedirs(results_path, exist_ok=True)

    env = ACCElegantEnvironment(
        stage=None,
        n_bins=5,
        init_num_particles=init_num_particles,
        results_path=results_path,

        input_beamline_file=work_lte,
        input_beam_file=cfg["track_file_stem"],
        beamline_name=cfg["beamline_name"],
        output_beamline_file=cfg["output_lte"],

        reset_specific_keys_bool=False,
        logger=None,
        file_handler=None,
        elegant_path="",
        sddsPath="",

        override_dynamic_command=True,
        overridden_command="elegant",
    )

    print("\nEnvironment created successfully.")
    print("Target:", TARGET_NAME)
    print("Mode:", mode)
    print("Lattice file:", work_lte)
    print("Track file:", f"{cfg['track_file_stem']}.ele")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Number of control steps:", env.max_num_of_vars)
    print("Controllable elements:")
    print(env.wrapper.chroneological_order_controllable_vars)
    print("Variables:")
    print(env.wrapper.chroneological_variables)

    if monitor:
        env = Monitor(env)

    return env


def make_td3_action_noise(env) -> NormalActionNoise:
    n_actions = env.action_space.shape[-1]

    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )


def evaluate_model(
    algorithm_name: str,
    model_path: str,
    init_num_particles: int,
    label: str,
) -> Dict[str, Any]:
    """
    Deterministically evaluate a saved model on the target beamline.
    """
    model_cls = ALG_MAP[algorithm_name]

    env = make_env(
        algorithm_name=algorithm_name,
        init_num_particles=init_num_particles,
        mode=f"eval_{label}",
        monitor=False,
    )

    model = model_cls.load(model_path, device="cpu")

    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    step_rows = []
    last_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        episode_reward += reward
        last_info = info

        step_rows.append({
            "target": TARGET_NAME,
            "label": label,
            "algorithm": algorithm_name,
            "watch_point": info.get("output_file"),
            "iteration": info.get("itteration"),
            "reward": reward,
            "episode_reward_so_far": episode_reward,
            "number_of_particles": info.get("number_of_particles"),
            "done": info.get("done"),
        })

    final_particles = last_info.get("number_of_particles", 0)
    transmission = final_particles / init_num_particles

    out_csv = (
        f"generalization_{TARGET_NAME}_"
        f"{label}_{algorithm_name.lower()}_{init_num_particles}.csv"
    )
    pd.DataFrame(step_rows).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULT: {algorithm_name} | {label}")
    print("=" * 80)
    print(f"Target: {TARGET_NAME}")
    print(f"Model path: {model_path}")
    print(f"Initial particles: {init_num_particles}")
    print(f"Final output: {last_info.get('output_file')}")
    print(f"Final particles: {final_particles}")
    print(f"Transmission: {transmission:.6f}")
    print(f"Transmission percent: {100 * transmission:.3f}%")
    print(f"Episode reward: {episode_reward:.6f}")
    print(f"Saved step log to: {out_csv}")

    return {
        "target": TARGET_NAME,
        "label": label,
        "algorithm": algorithm_name,
        "model_path": model_path,
        "init_num_particles": init_num_particles,
        "final_particles": final_particles,
        "transmission": transmission,
        "transmission_percent": 100 * transmission,
        "episode_reward": episode_reward,
        "final_output": last_info.get("output_file"),
    }


def fine_tune_algorithm(
    algorithm_name: str,
    pretrained_model_path: str,
    total_timesteps: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load pretrained model and fine-tune it on the target beamline.

    For SAC and TD3, the old replay buffer is intentionally NOT loaded,
    because it contains transitions from the original beamline.
    """
    if algorithm_name not in ALG_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    if not Path(pretrained_model_path).exists():
        raise FileNotFoundError(
            f"Could not find pretrained model: {Path(pretrained_model_path).resolve()}"
        )

    model_cls = ALG_MAP[algorithm_name]

    env = make_env(
        algorithm_name=algorithm_name,
        init_num_particles=1000,
        mode="finetune_train",
        monitor=True,
    )

    print("\n" + "=" * 80)
    print(f"FINE-TUNING {algorithm_name}")
    print("=" * 80)
    print(f"Target beamline: {TARGET_NAME}")
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Fine-tuning timesteps: {total_timesteps}")

    model = model_cls.load(
        pretrained_model_path,
        env=env,
        device="cpu",
    )

    # Important for fine-tuning on a new environment:
    # do not continue the old timestep counter.
    reset_num_timesteps = True

    # For off-policy models, collect fresh target-beamline transitions first.
    if algorithm_name in ["SAC", "TD3"]:
        model.learning_starts = 1000

    # TD3 requires explicit exploration noise.
    if algorithm_name == "TD3":
        model.action_noise = make_td3_action_noise(env)

    checkpoint_dir = (
        f"models/generalization_checkpoints/"
        f"{TARGET_NAME}/"
        f"{algorithm_name.lower()}/"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm_name.lower()}_{TARGET_NAME}_finetune",
        save_replay_buffer=(algorithm_name in ["SAC", "TD3"]),
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=f"{algorithm_name.lower()}_{TARGET_NAME}_finetune",
        callback=checkpoint_callback,
    )

    output_dir = f"models/generalization_finetuned/{TARGET_NAME}/"
    os.makedirs(output_dir, exist_ok=True)

    output_model_path = (
        f"{output_dir}"
        f"{algorithm_name.lower()}_{TARGET_NAME}_finetuned_{total_timesteps}"
    )

    model.save(output_model_path)
    saved_zip_path = output_model_path + ".zip"

    # Save the new target-beamline replay buffer for possible continuation.
    if algorithm_name in ["SAC", "TD3"]:
        replay_buffer_path = output_model_path + "_replay_buffer.pkl"
        try:
            model.save_replay_buffer(replay_buffer_path)
            print(f"Saved replay buffer to: {replay_buffer_path}")
        except Exception as e:
            print(f"Could not save replay buffer: {e}")

    print(f"Saved fine-tuned model to: {saved_zip_path}")

    train_info = {
        "target": TARGET_NAME,
        "algorithm": algorithm_name,
        "pretrained_model_path": pretrained_model_path,
        "fine_tuned_model_path": saved_zip_path,
        "fine_tuning_timesteps": total_timesteps,
    }

    return saved_zip_path, train_info


if __name__ == "__main__":
    check_required_files()

    all_rows = []

    for alg_name, pretrained_path in PRETRAINED_MODELS.items():
        timesteps = FINETUNE_TIMESTEPS[alg_name]

        # 1. Evaluate pretrained model before fine-tuning.
        before_1000 = evaluate_model(
            algorithm_name=alg_name,
            model_path=pretrained_path,
            init_num_particles=1000,
            label="before_finetune",
        )
        all_rows.append(before_1000)

        before_100000 = evaluate_model(
            algorithm_name=alg_name,
            model_path=pretrained_path,
            init_num_particles=100000,
            label="before_finetune",
        )
        all_rows.append(before_100000)

        # 2. Fine-tune on target beamline.
        fine_tuned_model_path, train_info = fine_tune_algorithm(
            algorithm_name=alg_name,
            pretrained_model_path=pretrained_path,
            total_timesteps=timesteps,
        )

        # 3. Evaluate after fine-tuning.
        after_1000 = evaluate_model(
            algorithm_name=alg_name,
            model_path=fine_tuned_model_path,
            init_num_particles=1000,
            label="after_finetune",
        )
        after_1000.update(train_info)
        all_rows.append(after_1000)

        after_100000 = evaluate_model(
            algorithm_name=alg_name,
            model_path=fine_tuned_model_path,
            init_num_particles=100000,
            label="after_finetune",
        )
        after_100000.update(train_info)
        all_rows.append(after_100000)

    summary = pd.DataFrame(all_rows)

    summary_path = f"fine_tuning_summary_{TARGET_NAME}.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("FINE-TUNING SUMMARY")
    print("=" * 80)
    print(summary)
    print(f"\nSaved summary to: {summary_path}")