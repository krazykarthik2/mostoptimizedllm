import os
import shutil
import time
import subprocess
import sys
import argparse


def run_command(command):
    print(f"\nExecuting: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with exit code: {return_code}")
        sys.exit(return_code)
    print("Command completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline: Gemma-3 -> EML-KAN (Clone + Distill + Benchmark)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="Delete cached outputs and rerun from scratch")
    parser.add_argument("--max_steps", type=int, default=5000, help="Distillation training steps (Phase B)")
    parser.add_argument("--kd_alpha", type=float, default=0.5, help="KD loss weight (0=pure CE, 1=pure KD)")
    parser.add_argument("--kd_temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    print("=" * 80)
    print("   PIPELINE: GEMMA-3 -> EML-KAN (Clone -> Distill -> Benchmark)")
    print("=" * 80)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")

    if args.force:
        print("\n[FORCE] Deleting existing outputs...")
        for d in ["gemma3_eml_kan"]:
            if os.path.exists(d):
                shutil.rmtree(d)

    phase_times = {}

    # ---- Phase A: Structural Clone (weight copy, no training) ----
    t0 = time.time()
    model_state_path = os.path.join("gemma3_eml_kan", "model_state.pt")
    if os.path.exists(model_state_path):
        print("\n[PHASE A] Cloned model found. Skipping...")
        phase_times["Phase A: Clone"] = 0.0
    else:
        print("\n[PHASE A] Cloning Gemma-3 weights into EML-KAN architecture...")
        run_command([
            "python3", "clone_model.py",
            "--model_id", "google/gemma-3-1b-it",
            "--save_path", "gemma3_eml_kan",
            "--device", args.device
        ])
        phase_times["Phase A: Clone"] = time.time() - t0

    # ---- Phase B: Knowledge Distillation ----
    t0 = time.time()
    tuned_path = os.path.join("gemma3_eml_kan", "model_state_tuned.pt")
    if os.path.exists(tuned_path) and not args.force:
        print("\n[PHASE B] Distilled model found. Skipping...")
        phase_times["Phase B: Distill"] = 0.0
    else:
        print("\n[PHASE B] Knowledge Distillation: teacher=Gemma-3, student=EML-KAN...")
        run_command([
            "python3", "fine_tune.py",
            "--model_id", "google/gemma-3-1b-it",
            "--weights_path", "gemma3_eml_kan/model_state.pt",
            "--save_path", "gemma3_eml_kan",
            "--lr", str(args.lr),
            "--max_steps", str(args.max_steps),
            "--kd_alpha", str(args.kd_alpha),
            "--kd_temperature", str(args.kd_temperature),
            "--seq_len", str(args.seq_len)
        ])
        phase_times["Phase B: Distill"] = time.time() - t0

    # ---- Phase C: Compression & Benchmarking ----
    t0 = time.time()
    print("\n[PHASE C] Edge Compression & Benchmarking...")
    run_command([
        "python3", "compress_benchmark.py",
        "--model_id", "google/gemma-3-1b-it",
        "--weights_path", "gemma3_eml_kan/model_state_tuned.pt"
    ])
    phase_times["Phase C: Benchmark"] = time.time() - t0

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("                    PIPELINE COMPLETE")
    print("=" * 80)
    total = 0.0
    for name, elapsed in phase_times.items():
        total += elapsed
        if elapsed > 0:
            print(f"  {name:<30} {elapsed:>8.1f}s  ({elapsed/60:.1f}m)")
        else:
            print(f"  {name:<30} {'Skipped':>8}")
    print(f"  {'Total':<30} {total:>8.1f}s  ({total/60:.1f}m)")
    print("=" * 80)


if __name__ == "__main__":
    main()
