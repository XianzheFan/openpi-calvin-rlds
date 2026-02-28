#!/usr/bin/env python3
"""
Checkpoint Manager Script

This script helps manage experiment checkpoints by listing all experiments
and providing options to delete unnecessary ones or manage checkpoint steps.

Usage:
    python checkpoint_manager.py [--dry-run] [--interactive] [--keep-latest-steps N] [--delete-steps "N1,N2,N3"]
    python checkpoint_manager.py --delete-train-state [--dry-run]
"""

import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class CheckpointManager:
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict:
        """Get information about a specific checkpoint step."""
        info = {"path": str(checkpoint_path), "step": int(checkpoint_path.name), "size": 0, "modified_time": None}

        try:
            # Get directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(checkpoint_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            info["size"] = total_size

            # Get modification time
            info["modified_time"] = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

        except Exception as e:
            print(f"Warning: Could not get info for checkpoint {checkpoint_path}: {e}")

        return info

    def get_experiment_info(self, exp_path: Path) -> Dict:
        """Get information about an experiment directory."""
        info = {
            "path": str(exp_path),
            "name": exp_path.name,
            "size": 0,
            "modified_time": None,
            "checkpoints": [],
            "checkpoint_details": [],
            "wandb_id": None,
        }

        try:
            # Get directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(exp_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            info["size"] = total_size

            # Get modification time
            info["modified_time"] = datetime.fromtimestamp(exp_path.stat().st_mtime)

            # Find checkpoint directories (numeric directories) and get detailed info
            checkpoint_dirs = []
            checkpoint_details = []
            for item in exp_path.iterdir():
                if item.is_dir() and item.name.isdigit():
                    step_num = int(item.name)
                    checkpoint_dirs.append(step_num)
                    checkpoint_info = self.get_checkpoint_info(item)
                    checkpoint_details.append(checkpoint_info)

            info["checkpoints"] = sorted(checkpoint_dirs)
            info["checkpoint_details"] = sorted(checkpoint_details, key=lambda x: x["step"])

            # Read wandb_id if exists
            wandb_id_file = exp_path / "wandb_id.txt"
            if wandb_id_file.exists():
                info["wandb_id"] = wandb_id_file.read_text().strip()

        except Exception as e:
            print(f"Warning: Could not get info for {exp_path}: {e}")

        return info

    def list_all_experiments(self) -> List[Dict]:
        """List all experiments in the checkpoints directory."""
        experiments = []

        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_info = self.get_experiment_info(exp_dir)
                exp_info["model_type"] = model_dir.name
                experiments.append(exp_info)

        return experiments

    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def print_experiments(
        self, experiments: List[Dict], show_details: bool = True, show_checkpoint_steps: bool = False
    ):
        """Print experiments in a formatted table."""
        if not experiments:
            print("No experiments found.")
            return

        print(f"\nFound {len(experiments)} experiments:")
        print("=" * 120)

        if show_details:
            print(
                f"{'Model':<20} {'Experiment':<30} {'Size':<10} {'Modified':<20} {'Checkpoints':<15} {'Wandb ID':<15}"
            )
            print("-" * 120)

            for exp in experiments:
                modified_str = exp["modified_time"].strftime("%Y-%m-%d %H:%M") if exp["modified_time"] else "Unknown"
                checkpoints_str = f"{len(exp['checkpoints'])} steps" if exp["checkpoints"] else "None"
                wandb_id = exp["wandb_id"] or "N/A"

                print(
                    f"{exp['model_type']:<20} {exp['name']:<30} {self.format_size(exp['size']):<10} "
                    f"{modified_str:<20} {checkpoints_str:<15} {wandb_id:<15}"
                )

                # Show checkpoint steps if requested
                if show_checkpoint_steps and exp["checkpoint_details"]:
                    print(f"  Checkpoint steps:")
                    for ckpt in exp["checkpoint_details"]:
                        ckpt_modified = (
                            ckpt["modified_time"].strftime("%Y-%m-%d %H:%M") if ckpt["modified_time"] else "Unknown"
                        )
                        print(f"    Step {ckpt['step']:<8} {self.format_size(ckpt['size']):<10} {ckpt_modified}")
        else:
            for exp in experiments:
                print(f"  {exp['model_type']}/{exp['name']} ({self.format_size(exp['size'])})")
                if show_checkpoint_steps and exp["checkpoint_details"]:
                    steps_str = ", ".join([f"step {c['step']}" for c in exp["checkpoint_details"]])
                    print(f"    Steps: {steps_str}")

    def delete_checkpoint_step(self, checkpoint_path: str, dry_run: bool = True) -> bool:
        """Delete a specific checkpoint step directory."""
        path = Path(checkpoint_path)

        if not path.exists():
            print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
            return False

        if dry_run:
            print(f"[DRY RUN] Would delete checkpoint: {checkpoint_path}")
            return True

        try:
            shutil.rmtree(path)
            print(f"Deleted checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error deleting checkpoint {checkpoint_path}: {e}")
            return False

    def delete_experiment(self, exp_path: str, dry_run: bool = True) -> bool:
        """Delete an experiment directory."""
        path = Path(exp_path)

        if not path.exists():
            print(f"Warning: Path does not exist: {exp_path}")
            return False

        if dry_run:
            print(f"[DRY RUN] Would delete: {exp_path}")
            return True

        try:
            shutil.rmtree(path)
            print(f"Deleted: {exp_path}")
            return True
        except Exception as e:
            print(f"Error deleting {exp_path}: {e}")
            return False

    def delete_train_state(self, checkpoint_step_path: Path, dry_run: bool = True) -> tuple[bool, int]:
        """Delete train_state directory from a checkpoint step.

        Returns:
            tuple: (success: bool, size_freed: int)
        """
        train_state_path = checkpoint_step_path / "train_state"

        if not train_state_path.exists():
            return False, 0

        # Calculate size before deletion
        size_freed = 0
        try:
            for dirpath, dirnames, filenames in os.walk(train_state_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        size_freed += os.path.getsize(filepath)
        except Exception as e:
            print(f"Warning: Could not calculate size for {train_state_path}: {e}")

        if dry_run:
            print(f"[DRY RUN] Would delete train_state: {train_state_path} ({self.format_size(size_freed)})")
            return True, size_freed

        try:
            shutil.rmtree(train_state_path)
            print(f"Deleted train_state: {train_state_path} ({self.format_size(size_freed)})")
            return True, size_freed
        except Exception as e:
            print(f"Error deleting train_state {train_state_path}: {e}")
            return False, 0

    def delete_all_train_states(self, experiments: List[Dict], dry_run: bool = True) -> tuple[int, int]:
        """Delete all train_state directories from all checkpoints in all experiments.

        Returns:
            tuple: (deleted_count: int, total_size_freed: int)
        """
        deleted_count = 0
        total_size_freed = 0

        for experiment in experiments:
            if not experiment["checkpoint_details"]:
                continue

            print(f"\nProcessing {experiment['model_type']}/{experiment['name']}:")
            exp_deleted = 0
            exp_size_freed = 0

            for checkpoint_info in experiment["checkpoint_details"]:
                checkpoint_path = Path(checkpoint_info["path"])
                success, size_freed = self.delete_train_state(checkpoint_path, dry_run)
                if success:
                    exp_deleted += 1
                    exp_size_freed += size_freed

            if exp_deleted > 0:
                action = "would be deleted" if dry_run else "deleted"
                print(f"  {exp_deleted} train_state directories {action} " f"({self.format_size(exp_size_freed)})")
                deleted_count += exp_deleted
                total_size_freed += exp_size_freed
            else:
                print("  No train_state directories found")

        return deleted_count, total_size_freed

    def manage_checkpoint_steps(
        self,
        experiment: Dict,
        keep_latest: int = None,
        delete_steps: List[int] = None,
        skip_steps: List[int] = None,
        dry_run: bool = True,
    ) -> int:
        """Manage checkpoint steps within an experiment."""
        if not experiment["checkpoint_details"]:
            print(f"No checkpoints found in {experiment['name']}")
            return 0

        steps_to_delete = []

        if keep_latest is not None:
            # Keep only the latest N steps
            if len(experiment["checkpoint_details"]) > keep_latest:
                # Sort by step number (descending) and take the oldest ones to delete
                sorted_steps = sorted(experiment["checkpoint_details"], key=lambda x: x["step"], reverse=True)
                candidate_steps = sorted_steps[keep_latest:]

                # Filter out steps that should be skipped
                if skip_steps:
                    candidate_steps = [step for step in candidate_steps if step["step"] not in skip_steps]

                steps_to_delete.extend(candidate_steps)

        if delete_steps is not None:
            # Delete specific steps
            for step_num in delete_steps:
                # Skip if this step is in the skip list
                if skip_steps and step_num in skip_steps:
                    print(f"Skipping step {step_num} in {experiment['name']} (in skip list)")
                    continue

                step_info = next((s for s in experiment["checkpoint_details"] if s["step"] == step_num), None)
                if step_info:
                    steps_to_delete.append(step_info)
                else:
                    print(f"Warning: Step {step_num} not found in {experiment['name']}")

        # Remove duplicates while preserving order
        seen_steps = set()
        unique_steps_to_delete = []
        for step in steps_to_delete:
            if step["step"] not in seen_steps:
                unique_steps_to_delete.append(step)
                seen_steps.add(step["step"])
        steps_to_delete = unique_steps_to_delete

        if not steps_to_delete:
            print(f"No checkpoint steps to delete in {experiment['name']}")
            return 0

        print(f"\nManaging checkpoints in {experiment['model_type']}/{experiment['name']}:")
        total_size = sum(step["size"] for step in steps_to_delete)
        print(f"Steps to delete: {[s['step'] for s in steps_to_delete]}")
        if skip_steps:
            print(f"Steps skipped: {skip_steps}")
        print(f"Total size to be freed: {self.format_size(total_size)}")

        deleted_count = 0
        for step_info in steps_to_delete:
            if self.delete_checkpoint_step(step_info["path"], dry_run):
                deleted_count += 1

        return deleted_count

    def manage_all_checkpoint_steps(
        self,
        experiments: List[Dict],
        keep_latest: int = None,
        delete_steps: List[int] = None,
        skip_steps: List[int] = None,
        dry_run: bool = True,
    ) -> int:
        """Manage checkpoint steps across all experiments."""
        total_deleted = 0

        for experiment in experiments:
            if experiment["checkpoint_details"]:
                deleted = self.manage_checkpoint_steps(experiment, keep_latest, delete_steps, skip_steps, dry_run)
                total_deleted += deleted

        return total_deleted

    def interactive_delete(self, experiments: List[Dict], dry_run: bool = True):
        """Interactive deletion mode."""
        if not experiments:
            print("No experiments to delete.")
            return

        print(f"\nInteractive deletion mode {'(DRY RUN)' if dry_run else '(LIVE)'}")
        print("Commands:")
        print("  y - delete this experiment")
        print("  n - skip this experiment")
        print("  c - manage checkpoint steps in this experiment")
        print("  a - delete all remaining experiments")
        print("  q - quit")
        print("  s - show experiment details")

        deleted_count = 0
        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] {exp['model_type']}/{exp['name']}")
            print(f"  Size: {self.format_size(exp['size'])}")
            print(
                f"  Modified: {exp['modified_time'].strftime('%Y-%m-%d %H:%M') if exp['modified_time'] else 'Unknown'}"
            )
            print(f"  Checkpoints: {len(exp['checkpoints'])} steps")
            print(f"  Wandb ID: {exp['wandb_id'] or 'N/A'}")

            while True:
                choice = input("Action [y/n/c/a/q/s]: ").lower().strip()

                if choice == "y":
                    if self.delete_experiment(exp["path"], dry_run):
                        deleted_count += 1
                    break
                elif choice == "n":
                    break
                elif choice == "c":
                    # Manage checkpoint steps
                    self.interactive_checkpoint_management(exp, dry_run)
                    break
                elif choice == "a":
                    # Delete all remaining
                    for remaining_exp in experiments[i:]:
                        if self.delete_experiment(remaining_exp["path"], dry_run):
                            deleted_count += 1
                    print(f"\nDeleted {deleted_count} experiments total.")
                    return
                elif choice == "q":
                    print(f"\nQuit. Deleted {deleted_count} experiments.")
                    return
                elif choice == "s":
                    print(f"\nFull path: {exp['path']}")
                    if exp["checkpoints"]:
                        print(f"Checkpoint steps: {exp['checkpoints']}")
                        if exp["checkpoint_details"]:
                            print("Checkpoint details:")
                            for ckpt in exp["checkpoint_details"]:
                                ckpt_modified = (
                                    ckpt["modified_time"].strftime("%Y-%m-%d %H:%M")
                                    if ckpt["modified_time"]
                                    else "Unknown"
                                )
                                print(f"  Step {ckpt['step']}: {self.format_size(ckpt['size'])} ({ckpt_modified})")
                else:
                    print("Invalid choice. Please enter y, n, c, a, q, or s.")

        print(f"\nCompleted. Deleted {deleted_count} experiments.")

    def interactive_checkpoint_management(self, experiment: Dict, dry_run: bool = True):
        """Interactive checkpoint step management for a single experiment."""
        if not experiment["checkpoint_details"]:
            print(f"No checkpoints found in {experiment['name']}")
            return

        print(f"\nManaging checkpoints in {experiment['model_type']}/{experiment['name']}")
        print("Available checkpoint steps:")
        for ckpt in experiment["checkpoint_details"]:
            ckpt_modified = ckpt["modified_time"].strftime("%Y-%m-%d %H:%M") if ckpt["modified_time"] else "Unknown"
            print(f"  Step {ckpt['step']:<8} {self.format_size(ckpt['size']):<10} {ckpt_modified}")

        print("\nCommands:")
        print("  k N - keep only the latest N steps")
        print("  d N1,N2,N3 - delete specific steps (comma-separated)")
        print("  s N1,N2,N3 - skip specific steps when deleting (comma-separated)")
        print("  q - quit checkpoint management")

        while True:
            choice = input("Checkpoint action [k N/d N1,N2,N3/s N1,N2,N3/q]: ").strip()

            if choice.lower() == "q":
                break
            elif choice.startswith("k "):
                try:
                    keep_count = int(choice[2:])
                    if keep_count < 1:
                        print("Keep count must be at least 1")
                        continue

                    # Ask for skip steps
                    skip_input = input("Steps to skip (comma-separated, or press Enter for none): ").strip()
                    skip_steps = None
                    if skip_input:
                        try:
                            skip_steps = [int(x.strip()) for x in skip_input.split(",")]
                        except ValueError:
                            print("Invalid skip step numbers. Ignoring skip list.")

                    self.manage_checkpoint_steps(
                        experiment, keep_latest=keep_count, skip_steps=skip_steps, dry_run=dry_run
                    )
                    break
                except ValueError:
                    print("Invalid number for keep count")
            elif choice.startswith("d "):
                try:
                    step_nums = [int(x.strip()) for x in choice[2:].split(",")]

                    # Ask for skip steps
                    skip_input = input("Steps to skip (comma-separated, or press Enter for none): ").strip()
                    skip_steps = None
                    if skip_input:
                        try:
                            skip_steps = [int(x.strip()) for x in skip_input.split(",")]
                        except ValueError:
                            print("Invalid skip step numbers. Ignoring skip list.")

                    self.manage_checkpoint_steps(
                        experiment, delete_steps=step_nums, skip_steps=skip_steps, dry_run=dry_run
                    )
                    break
                except ValueError:
                    print("Invalid step numbers. Use comma-separated integers (e.g., 1000,2000,3000)")
            elif choice.startswith("s "):
                try:
                    skip_nums = [int(x.strip()) for x in choice[2:].split(",")]
                    print(f"Steps {skip_nums} will be skipped in future operations for this experiment.")
                    print("Use 'k N' or 'd N1,N2,N3' commands to perform deletions with these skip steps.")
                except ValueError:
                    print("Invalid skip step numbers. Use comma-separated integers (e.g., 1000,2000,3000)")
            else:
                print("Invalid choice. Use 'k N', 'd N1,N2,N3', 's N1,N2,N3', or 'q'")


def main():
    parser = argparse.ArgumentParser(description="Manage experiment checkpoints")
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints", help="Path to checkpoints directory (default: checkpoints)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--interactive", action="store_true", help="Interactive deletion mode")
    parser.add_argument("--list-only", action="store_true", help="Only list experiments, don't delete anything")
    parser.add_argument("--summary", action="store_true", help="Show summary only (no detailed table)")
    parser.add_argument("--show-steps", action="store_true", help="Show detailed checkpoint steps information")

    # Checkpoint step management options
    parser.add_argument(
        "--keep-latest-steps", type=int, default=1, help="Keep only the latest N checkpoint steps in each experiment"
    )
    parser.add_argument(
        "--delete-steps",
        type=str,
        default="",
        help="Delete specific checkpoint steps (comma-separated, e.g., 1000,2000,3000)",
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default="30000, 50000",
        help="Skip specific checkpoint steps when deleting (comma-separated, e.g., 50000,100000)",
    )
    parser.add_argument(
        "--manage-steps-only", action="store_true", help="Only manage checkpoint steps, don't delete entire experiments"
    )
    parser.add_argument(
        "--delete-train-state", action="store_true", help="Delete all train_state directories from existing checkpoints"
    )

    args = parser.parse_args()

    try:
        manager = CheckpointManager(args.checkpoints_dir)

        # List all experiments
        experiments = manager.list_all_experiments()

        # Parse delete steps if provided
        delete_steps = None
        if args.delete_steps:
            try:
                delete_steps = [int(x.strip()) for x in args.delete_steps.split(",")]
            except ValueError:
                print("Error: Invalid step numbers. Use comma-separated integers (e.g., 1000,2000,3000)")
                return 1

        # Parse skip steps if provided
        skip_steps = None
        if args.skip_steps:
            try:
                skip_steps = [int(x.strip()) for x in args.skip_steps.split(",")]
            except ValueError:
                print("Error: Invalid skip step numbers. Use comma-separated integers (e.g., 50000,100000)")
                return 1

        # Handle train_state deletion
        if args.delete_train_state:
            print(f"\nDeleting train_state directories from {len(experiments)} experiments:")
            manager.print_experiments(experiments, show_details=False, show_checkpoint_steps=True)

            if not args.dry_run:
                confirm = input(f"\nDelete all train_state directories in {len(experiments)} experiments? (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return 0

            deleted_count, total_size_freed = manager.delete_all_train_states(experiments, dry_run=args.dry_run)
            action = "Would delete" if args.dry_run else "Deleted"
            print(f"\n{action} {deleted_count} train_state directories " f"({manager.format_size(total_size_freed)})")

        # Handle checkpoint step management
        elif args.keep_latest_steps or delete_steps or args.manage_steps_only:
            print(f"\nManaging checkpoint steps in {len(experiments)} experiments:")
            manager.print_experiments(experiments, show_details=False, show_checkpoint_steps=True)

            if args.interactive:
                manager.interactive_delete(experiments, dry_run=args.dry_run)
            else:
                # Auto-manage checkpoint steps
                if not args.dry_run:
                    action_desc = []
                    if args.keep_latest_steps:
                        action_desc.append(f"keep latest {args.keep_latest_steps} steps")
                    if delete_steps:
                        action_desc.append(f"delete steps {delete_steps}")
                    if skip_steps:
                        action_desc.append(f"skip steps {skip_steps}")

                    confirm = input(f"\n{' and '.join(action_desc)} in {len(experiments)} experiments? (y/N): ")
                    if confirm.lower() != "y":
                        print("Cancelled.")
                        return

                deleted_count = manager.manage_all_checkpoint_steps(
                    experiments,
                    keep_latest=args.keep_latest_steps,
                    delete_steps=delete_steps,
                    skip_steps=skip_steps,
                    dry_run=args.dry_run,
                )

                print(f"\n{'Would delete' if args.dry_run else 'Deleted'} {deleted_count} checkpoint steps.")

        elif args.list_only or not args.interactive:
            # Just list experiments
            manager.print_experiments(experiments, show_details=not args.summary, show_checkpoint_steps=args.show_steps)

            if not args.list_only:
                total_size = sum(exp["size"] for exp in experiments)
                print(f"\nTotal size: {manager.format_size(total_size)}")
        else:
            # Interactive mode for experiment deletion
            print(f"\nFound {len(experiments)} experiments:")
            manager.print_experiments(experiments, show_details=False, show_checkpoint_steps=args.show_steps)

            total_size = sum(exp["size"] for exp in experiments)
            print(f"\nTotal size: {manager.format_size(total_size)}")

            manager.interactive_delete(experiments, dry_run=args.dry_run)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
