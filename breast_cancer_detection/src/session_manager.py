"""
Session Manager for Multi-Session Training on Kaggle.

Handles:
- Time budget tracking with configurable limits
- Safe stopping before timeout
- Progress estimation and reporting
- Auto-save triggers
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict


class SessionManager:
    """
    Manages training session time budgets to prevent Kaggle timeout.

    Designed for Kaggle's 30-hour session limit, with conservative
    time management to ensure checkpoints are saved before timeout.

    Example:
        >>> manager = SessionManager(max_session_hours=22, buffer_minutes=30)
        >>> while not manager.should_stop():
        ...     # Run training iteration
        ...     pass
    """

    def __init__(self, max_session_hours: float = 22.0, buffer_minutes: float = 30.0):
        """
        Initialize session manager with time limits.

        Args:
            max_session_hours: Maximum session duration in hours (default: 22)
            buffer_minutes: Safety buffer before timeout in minutes (default: 30)
                           This ensures we stop early enough to save checkpoints

        Example:
            Conservative (recommended): max_session_hours=22, buffer_minutes=30
            Aggressive: max_session_hours=28, buffer_minutes=15
        """
        self.start_time = time.time()
        self.max_duration_seconds = max_session_hours * 3600
        self.buffer_seconds = buffer_minutes * 60
        self.max_session_hours = max_session_hours
        self.buffer_minutes = buffer_minutes

        # Effective duration = max - buffer
        self.effective_duration = self.max_duration_seconds - self.buffer_seconds

        print(f"Session Manager initialized:")
        print(f"  Start time: {self.get_start_time_str()}")
        print(f"  Max session: {max_session_hours:.1f} hours")
        print(f"  Safety buffer: {buffer_minutes:.0f} minutes")
        print(f"  Effective duration: {self.effective_duration / 3600:.1f} hours")
        print(f"  Expected stop time: {self.get_expected_stop_time_str()}")

    def time_elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def time_elapsed_hours(self) -> float:
        """Get elapsed time in hours."""
        return self.time_elapsed_seconds() / 3600

    def time_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return self.time_elapsed_seconds() / 60

    def time_remaining_seconds(self) -> float:
        """Get remaining time before effective deadline (in seconds)."""
        return max(0, self.effective_duration - self.time_elapsed_seconds())

    def time_remaining_hours(self) -> float:
        """Get remaining time before effective deadline (in hours)."""
        return self.time_remaining_seconds() / 3600

    def time_remaining_minutes(self) -> float:
        """Get remaining time before effective deadline (in minutes)."""
        return self.time_remaining_seconds() / 60

    def should_stop(self) -> bool:
        """
        Check if we should stop to avoid timeout.

        Returns:
            True if elapsed time >= effective duration, False otherwise
        """
        return self.time_elapsed_seconds() >= self.effective_duration

    def estimate_evaluations_left(self, avg_eval_time_minutes: float = 30.0) -> int:
        """
        Estimate how many more evaluations can fit in remaining time.

        Args:
            avg_eval_time_minutes: Average time per evaluation in minutes

        Returns:
            Number of evaluations that can fit
        """
        remaining_minutes = self.time_remaining_minutes()
        return max(0, int(remaining_minutes / avg_eval_time_minutes))

    def estimate_iterations_left(
        self,
        avg_eval_time_minutes: float = 30.0,
        batch_size: int = 5
    ) -> int:
        """
        Estimate how many more BO iterations can fit in remaining time.

        Args:
            avg_eval_time_minutes: Average time per evaluation in minutes
            batch_size: Number of evaluations per iteration

        Returns:
            Number of iterations that can fit
        """
        evals_left = self.estimate_evaluations_left(avg_eval_time_minutes)
        return max(0, evals_left // batch_size)

    def get_progress_fraction(self, current_iter: int, total_iters: int) -> float:
        """
        Calculate progress fraction.

        Args:
            current_iter: Current iteration number
            total_iters: Total iterations planned

        Returns:
            Progress fraction [0, 1]
        """
        return min(1.0, current_iter / total_iters) if total_iters > 0 else 0.0

    def get_progress_report(
        self,
        current_iter: int,
        total_iters: int,
        total_evaluations: int = 0,
        target_evaluations: int = 200,
        avg_eval_time_minutes: float = 30.0
    ) -> Dict[str, any]:
        """
        Generate comprehensive progress report.

        Args:
            current_iter: Current iteration number
            total_iters: Total iterations planned
            total_evaluations: Total evaluations completed so far
            target_evaluations: Target total evaluations
            avg_eval_time_minutes: Average evaluation time

        Returns:
            Dictionary with progress metrics
        """
        progress_fraction = self.get_progress_fraction(current_iter, total_iters)

        return {
            'start_time': self.get_start_time_str(),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_hours': self.time_elapsed_hours(),
            'elapsed_minutes': self.time_elapsed_minutes(),
            'remaining_hours': self.time_remaining_hours(),
            'remaining_minutes': self.time_remaining_minutes(),
            'should_stop': self.should_stop(),
            'progress_fraction': progress_fraction,
            'progress_percent': progress_fraction * 100,
            'current_iter': current_iter,
            'total_iters': total_iters,
            'evaluations_completed': total_evaluations,
            'target_evaluations': target_evaluations,
            'evaluations_left': self.estimate_evaluations_left(avg_eval_time_minutes),
            'iterations_left': self.estimate_iterations_left(avg_eval_time_minutes),
            'expected_stop_time': self.get_expected_stop_time_str()
        }

    def print_progress_report(
        self,
        current_iter: int,
        total_iters: int,
        total_evaluations: int = 0,
        target_evaluations: int = 200,
        avg_eval_time_minutes: float = 30.0
    ):
        """
        Print formatted progress report to console.

        Args:
            current_iter: Current iteration number
            total_iters: Total iterations planned
            total_evaluations: Total evaluations completed
            target_evaluations: Target total evaluations
            avg_eval_time_minutes: Average evaluation time
        """
        report = self.get_progress_report(
            current_iter, total_iters, total_evaluations,
            target_evaluations, avg_eval_time_minutes
        )

        print("\n" + "="*80)
        print("SESSION PROGRESS REPORT")
        print("="*80)

        print(f"\nTime:")
        print(f"  Start:     {report['start_time']}")
        print(f"  Current:   {report['current_time']}")
        print(f"  Elapsed:   {report['elapsed_hours']:.1f} hours ({report['elapsed_minutes']:.0f} min)")
        print(f"  Remaining: {report['remaining_hours']:.1f} hours ({report['remaining_minutes']:.0f} min)")
        print(f"  Stop at:   {report['expected_stop_time']}")

        print(f"\nIteration Progress:")
        print(f"  Current:   {report['current_iter']}/{report['total_iters']}")
        print(f"  Completed: {report['progress_percent']:.1f}%")

        print(f"\nEvaluation Progress:")
        print(f"  Completed: {report['evaluations_completed']}/{report['target_evaluations']}")
        print(f"  Progress:  {100 * report['evaluations_completed'] / report['target_evaluations']:.1f}%")

        print(f"\nEstimates:")
        print(f"  Evaluations left in session: ~{report['evaluations_left']}")
        print(f"  Iterations left in session:  ~{report['iterations_left']}")

        if report['should_stop']:
            print(f"\n⚠️  WARNING: Time budget exceeded! Should stop now.")
        else:
            print(f"\n✓ Session has {report['remaining_hours']:.1f} hours remaining")

        print("="*80)

    def get_start_time_str(self) -> str:
        """Get formatted start time string."""
        return datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')

    def get_expected_stop_time_str(self) -> str:
        """Get formatted expected stop time string."""
        stop_time = self.start_time + self.effective_duration
        return datetime.fromtimestamp(stop_time).strftime('%Y-%m-%d %H:%M:%S')

    def get_current_time_str(self) -> str:
        """Get formatted current time string."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def get_time_until_stop(self) -> timedelta:
        """Get time remaining as timedelta object."""
        return timedelta(seconds=self.time_remaining_seconds())

    def reset(self):
        """Reset the session timer to current time."""
        self.start_time = time.time()
        print(f"Session timer reset at {self.get_start_time_str()}")


class SessionLogger:
    """
    Logs session metadata for tracking across multiple sessions.

    Useful for tracking optimization progress across Kaggle sessions.
    """

    def __init__(self, log_file: str = "session_log.csv"):
        """
        Initialize session logger.

        Args:
            log_file: Path to CSV log file
        """
        self.log_file = log_file
        self._ensure_header()

    def _ensure_header(self):
        """Ensure CSV header exists."""
        from pathlib import Path
        import csv

        log_path = Path(self.log_file)

        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'session_id', 'start_time', 'end_time',
                    'duration_hours', 'evaluations_completed',
                    'iterations_completed', 'status', 'notes'
                ])

    def log_session(
        self,
        session_id: int,
        start_time: str,
        end_time: str,
        duration_hours: float,
        evaluations_completed: int,
        iterations_completed: int,
        status: str = 'completed',
        notes: str = ''
    ):
        """
        Log a completed session.

        Args:
            session_id: Session identifier
            start_time: Session start time (ISO format)
            end_time: Session end time (ISO format)
            duration_hours: Session duration in hours
            evaluations_completed: Number of evaluations in this session
            iterations_completed: Number of iterations in this session
            status: Session status ('completed', 'timeout', 'error')
            notes: Additional notes
        """
        import csv

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id, start_time, end_time,
                f"{duration_hours:.2f}", evaluations_completed,
                iterations_completed, status, notes
            ])

    def get_last_session_id(self) -> int:
        """Get the ID of the last logged session."""
        import csv
        from pathlib import Path

        log_path = Path(self.log_file)
        if not log_path.exists():
            return 0

        with open(log_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)

        if not rows:
            return 0

        return int(rows[-1][0])

    def get_total_evaluations(self) -> int:
        """Get total evaluations across all sessions."""
        import csv
        from pathlib import Path

        log_path = Path(self.log_file)
        if not log_path.exists():
            return 0

        total = 0
        with open(log_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 5:
                    total += int(row[4])

        return total
