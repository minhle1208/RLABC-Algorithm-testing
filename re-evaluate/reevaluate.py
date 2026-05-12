import os
import re
import ast
import shutil
import traceback
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from main import RLConfig
from libscratch.Utils import change_num_initial_particles

N_REEVAL = 100_000
TOP_K = 5

CSV_FILES = {
    "SAC": "big_beamline_Orig.csv",
    "PPO": "big_beamline_Orig_ppo.csv",
    "TD3": "big_beamline_Orig_ppo_td3.csv",
}

OUTPUT_DETAILED = "reevaluation_top5_detailed.csv"
OUTPUT_SUMMARY = "reevaluation_top5_summary.csv"


def parse_dict_vars(s: str) -> Dict[str, float]:
    """
    Convert logged dict_vars string into a normal Python dictionary.

    Handles values like:
        np.float32(0.123)
        np.float64(0.123)
        array(0.123)
    """
    if not isinstance(s, str):
        raise ValueError("dict_vars is not a string")

    s = re.sub(r"np\.float(?:16|32|64)\(([^)]+)\)", r"\1", s)
    s = re.sub(r"array\(([^)]+)\)", r"\1", s)

    return ast.literal_eval(s)


def load_top_k_rows(csv_path: str, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Load top-k successful final_WP rows by number_of_particles.
    """
    df = pd.read_csv(csv_path)

    required_cols = {
        "reward",
        "initial_number_of_particles",
        "number_of_particles",
        "current_element",
        "dict_vars",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing columns: {missing_cols}")

    successful = df[df["current_element"] == "final_WP"].copy()

    if successful.empty:
        raise ValueError(f"No final_WP episodes found in {csv_path}")

    successful["original_row_index"] = successful.index

    top_rows = successful.sort_values(
        "number_of_particles",
        ascending=False,
    ).head(top_k)

    return top_rows.reset_index(drop=True)


def clear_results_folder(results_path: str):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)


def make_ordered_values(env, best_vars: Dict[str, float]) -> np.ndarray:
    """
    Convert dict_vars into the exact variable order expected by ElegantWrapper.
    """
    ordered_values = []
    missing = []

    for var_name in env.wrapper.chroneological_variables:
        if var_name not in best_vars:
            missing.append(var_name)
        else:
            ordered_values.append(float(best_vars[var_name]))

    if missing:
        raise ValueError(
            f"dict_vars is missing {len(missing)} variables. "
            f"First missing variables: {missing[:10]}"
        )

    return np.asarray(ordered_values, dtype=np.float32)


def get_max_watchpoint_reads(env) -> int:
    """
    Safety limit so the script cannot run forever.
    """
    possible_attrs = [
        "chronolgical_order_watch_points",
        "chronological_order_watch_points",
        "chroneological_order_watch_points",
    ]

    for attr in possible_attrs:
        if hasattr(env.wrapper, attr):
            return len(getattr(env.wrapper, attr)) + 5

    return 100


def reevaluate_one_row(
    algo_name: str,
    csv_path: str,
    rank: int,
    row: pd.Series,
    env,
    config: RLConfig,
    n_particles: int = N_REEVAL,
) -> Dict[str, Any]:
    """
    Re-evaluate one selected training row.
    """

    training_initial_particles = float(row["initial_number_of_particles"])
    training_best_particles = float(row["number_of_particles"])
    training_best_transmission = training_best_particles / training_initial_particles

    print("\n" + "-" * 80)
    print(f"{algo_name} | Re-evaluating rank #{rank}")
    print("-" * 80)
    print(f"CSV: {csv_path}")
    print(f"Original row index: {row.get('original_row_index', 'N/A')}")
    print(f"Training particles: {training_best_particles:.0f}")
    print(f"Training transmission: {training_best_transmission:.4f}")
    print(f"Training transmission percent: {training_best_transmission * 100:.2f}%")
    print(f"Reward: {row['reward']}")
    print(f"Terminal element: {row['current_element']}")

    best_vars = parse_dict_vars(row["dict_vars"])
    ordered_values = make_ordered_values(env, best_vars)

    _, initial_particles = change_num_initial_particles(
        config.input_beam_file + ".ele",
        n_particles,
    )

    clear_results_folder(config.results_path)

    env.wrapper.itteration = 0

    elegant_input, success, applied_dict_vars = env.wrapper.run_elegant_simulation(
        ordered_values
    )

    if not success:
        raise RuntimeError(f"Elegant failed during re-evaluation for {algo_name}, rank {rank}")

    done = False
    output_file = None
    final_particles = None
    reward_like_value = None
    max_reads = get_max_watchpoint_reads(env)

    watchpoint_trace = []

    print("\nReading watch points:")
    for read_idx in range(max_reads):
        observation, reward_like_value, output_file, done = env.wrapper.get_results(
            initial_particles
        )

        final_particles = env.wrapper.get_num_particles()

        watchpoint_trace.append({
            "watchpoint": output_file,
            "particles": final_particles,
            "done": done,
        })

        print(
            f"  [{read_idx + 1:02d}] "
            f"watch_point={output_file} | "
            f"particles={final_particles} | "
            f"done={done}"
        )

        if output_file == "final_WP":
            done = True
            break

        if done:
            break

    if output_file != "final_WP":
        raise RuntimeError(
            f"Invalid re-evaluation for {algo_name}, rank {rank}: "
            f"ended at {output_file}, not final_WP."
        )

    reevaluated_transmission = final_particles / initial_particles

    print("\n=== Valid Re-evaluation Result ===")
    print(f"Algorithm: {algo_name}")
    print(f"Rank: {rank}")
    print(f"N0: {initial_particles}")
    print(f"Final output file: {output_file}")
    print(f"Final particles: {final_particles}")
    print(f"Transmission: {reevaluated_transmission:.6f}")
    print(f"Transmission percent: {reevaluated_transmission * 100:.2f}%")

    # Store compact trace as a readable string for later analysis
    trace_str = " -> ".join(
        [f"{x['watchpoint']}:{x['particles']}" for x in watchpoint_trace]
    )

    return {
        "algorithm": algo_name,
        "csv_path": csv_path,
        "rank": rank,
        "original_row_index": row.get("original_row_index", None),
        "n_particles": initial_particles,
        "final_output_file": output_file,
        "final_particles": final_particles,
        "reevaluated_transmission": reevaluated_transmission,
        "reevaluated_transmission_percent": reevaluated_transmission * 100,
        "training_particles": training_best_particles,
        "training_transmission": training_best_transmission,
        "training_transmission_percent": training_best_transmission * 100,
        "training_reward": row["reward"],
        "valid": True,
        "error": "",
        "watchpoint_trace": trace_str,
    }


def reevaluate_algorithm_top_k(
    algo_name: str,
    csv_path: str,
    top_k: int = TOP_K,
    n_particles: int = N_REEVAL,
) -> List[Dict[str, Any]]:
    """
    Re-evaluate top-k training configurations for one algorithm.
    """

    print("\n" + "=" * 100)
    print(f"Re-evaluating top {top_k} configurations for {algo_name}")
    print("=" * 100)

    top_rows = load_top_k_rows(csv_path, top_k=top_k)

    print(f"\nSelected top {len(top_rows)} rows from {csv_path}:")
    print(
        top_rows[
            [
                "original_row_index",
                "reward",
                "initial_number_of_particles",
                "number_of_particles",
                "current_element",
            ]
        ]
    )

    config = RLConfig()
    config.run_elegant_preflight = False
    config.init_num_particles = n_particles
    config.override_dynamic_command = True
    config.overridden_command = "elegant"

    # setup_environment() will run one reset internally.
    # Every actual re-evaluation clears results afterward.
    env, _, file_handler = config.setup_environment()

    results = []

    try:
        for i, row in top_rows.iterrows():
            rank = i + 1

            try:
                result = reevaluate_one_row(
                    algo_name=algo_name,
                    csv_path=csv_path,
                    rank=rank,
                    row=row,
                    env=env,
                    config=config,
                    n_particles=n_particles,
                )
                results.append(result)

            except Exception as exc:
                print("\n" + "!" * 80)
                print(f"Failed re-evaluation: {algo_name}, rank {rank}")
                print("!" * 80)
                traceback.print_exc()

                results.append({
                    "algorithm": algo_name,
                    "csv_path": csv_path,
                    "rank": rank,
                    "original_row_index": row.get("original_row_index", None),
                    "valid": False,
                    "error": str(exc),
                    "training_particles": float(row["number_of_particles"]),
                    "training_transmission": float(row["number_of_particles"]) / float(row["initial_number_of_particles"]),
                    "training_transmission_percent": 100.0 * float(row["number_of_particles"]) / float(row["initial_number_of_particles"]),
                })

    finally:
        if file_handler is not None:
            file_handler.close()

    return results


def build_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate top-k re-evaluation results per algorithm.
    """
    valid = detailed_df[detailed_df["valid"] == True].copy()

    summary_rows = []

    for algo, part in valid.groupby("algorithm"):
        part = part.sort_values("reevaluated_transmission", ascending=False)

        best = part.iloc[0]

        summary_rows.append({
            "algorithm": algo,
            "n_valid_configs": len(part),
            "best_rank_after_reeval": int(best["rank"]),
            "best_reevaluated_particles": int(best["final_particles"]),
            "best_reevaluated_transmission": float(best["reevaluated_transmission"]),
            "best_reevaluated_transmission_percent": float(best["reevaluated_transmission_percent"]),
            "mean_topk_reevaluated_particles": float(part["final_particles"].mean()),
            "mean_topk_reevaluated_transmission": float(part["reevaluated_transmission"].mean()),
            "mean_topk_reevaluated_transmission_percent": float(part["reevaluated_transmission_percent"].mean()),
            "std_topk_reevaluated_transmission_percent": float(part["reevaluated_transmission_percent"].std(ddof=0)),
            "min_topk_reevaluated_transmission_percent": float(part["reevaluated_transmission_percent"].min()),
            "max_topk_reevaluated_transmission_percent": float(part["reevaluated_transmission_percent"].max()),
            "mean_training_transmission_percent": float(part["training_transmission_percent"].mean()),
            "best_training_transmission_percent": float(part["training_transmission_percent"].max()),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        "best_reevaluated_transmission",
        ascending=False,
    )

    return summary


def main():
    all_results: List[Dict[str, Any]] = []

    for algo_name, csv_path in CSV_FILES.items():
        if not os.path.exists(csv_path):
            print(f"\nSkipping {algo_name}: missing file {csv_path}")
            all_results.append({
                "algorithm": algo_name,
                "csv_path": csv_path,
                "valid": False,
                "error": "CSV file not found",
            })
            continue

        results = reevaluate_algorithm_top_k(
            algo_name=algo_name,
            csv_path=csv_path,
            top_k=TOP_K,
            n_particles=N_REEVAL,
        )

        all_results.extend(results)

    detailed_df = pd.DataFrame(all_results)
    detailed_df.to_csv(OUTPUT_DETAILED, index=False)

    summary_df = build_summary(detailed_df)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n" + "=" * 100)
    print(f"Saved detailed results to {OUTPUT_DETAILED}")
    print(f"Saved summary results to {OUTPUT_SUMMARY}")
    print("=" * 100)

    print("\nDetailed results:")
    print(detailed_df)

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()