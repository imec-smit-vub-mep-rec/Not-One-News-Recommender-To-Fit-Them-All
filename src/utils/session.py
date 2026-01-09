"""
Session management for the RICON analysis pipeline.

Provides a Session class that manages output directories, logging,
and configuration for a single analysis run.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import asdict
import json
import shutil

from ..config.settings import PipelineConfig, SessionConfig, save_config
from .logging import setup_logging, get_logger
from .io import ensure_dir, save_json


class Session:
    """Manages a single analysis session/run.
    
    A session creates a timestamped output directory and handles:
    - Directory structure creation
    - Logging configuration
    - Configuration saving
    - Progress tracking
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        session_config: Optional[SessionConfig] = None,
        dataset_name: str = "",
        base_output_dir: str = "runs",
    ):
        """Initialize a new session.
        
        Args:
            config: Full pipeline configuration (takes precedence)
            session_config: Session-specific configuration
            dataset_name: Name of the dataset being processed
            base_output_dir: Base directory for output
        """
        if config:
            self.config = config
            self.session_config = config.session
        elif session_config:
            self.session_config = session_config
            self.config = None
        else:
            self.session_config = SessionConfig(
                base_output_dir=base_output_dir,
                dataset_name=dataset_name,
            )
            self.config = None
        
        self._setup_directories()
        self._setup_logging()
        self._save_initial_config()
        
        self.logger.info(f"Session started: {self.session_config.run_id}")
        self.logger.info(f"Output directory: {self.session_dir}")
    
    @property
    def session_dir(self) -> Path:
        """Get the main session directory."""
        return self.session_config.session_dir
    
    @property
    def data_dir(self) -> Path:
        """Get the data output directory."""
        return self.session_config.data_dir
    
    @property
    def clusters_dir(self) -> Path:
        """Get the clusters output directory."""
        return self.session_config.clusters_dir
    
    @property
    def evaluation_dir(self) -> Path:
        """Get the evaluation output directory."""
        return self.session_config.evaluation_dir
    
    @property
    def figures_dir(self) -> Path:
        """Get the figures output directory."""
        return self.session_config.figures_dir
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.session_config.logs_dir
    
    def _setup_directories(self):
        """Create the session directory structure."""
        ensure_dir(self.session_dir)
        ensure_dir(self.data_dir)
        ensure_dir(self.clusters_dir)
        ensure_dir(self.evaluation_dir)
        ensure_dir(self.figures_dir)
        ensure_dir(self.logs_dir)
    
    def _setup_logging(self):
        """Configure logging for this session."""
        log_file = self.logs_dir / "session.log"
        setup_logging(log_file=log_file)
        self.logger = get_logger("session")
    
    def _save_initial_config(self):
        """Save the configuration used for this session."""
        config_path = self.session_dir / "config.json"
        
        if self.config:
            save_config(self.config, str(config_path))
        else:
            save_json(asdict(self.session_config), config_path)
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json"):
        """Save additional metadata to the session directory.
        
        Args:
            metadata: Metadata dictionary to save
            filename: Name of the metadata file
        """
        path = self.session_dir / filename
        save_json(metadata, path)
        self.logger.info(f"Saved metadata to {path}")
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save a session summary.
        
        Args:
            summary: Summary dictionary
        """
        summary["session_id"] = self.session_config.run_id
        summary["timestamp"] = self.session_config.timestamp
        summary["completed_at"] = datetime.now().isoformat()
        
        path = self.session_dir / "summary.json"
        save_json(summary, path)
        self.logger.info(f"Saved session summary to {path}")
    
    def get_subdir(self, name: str) -> Path:
        """Get or create a subdirectory in the session directory.
        
        Args:
            name: Name of the subdirectory
            
        Returns:
            Path to the subdirectory
        """
        subdir = self.session_dir / name
        ensure_dir(subdir)
        return subdir

    def get_path(self, filename: str, subdir: str = "data") -> Path:
        """Get the full path for a file in the session.
        
        Args:
            filename: Name of the file (or directory if no extension)
            subdir: Subdirectory (default: "data")
            
        Returns:
            Full path to the file or directory
        """
        # If filename looks like a directory name (no extension), treat it as a subdir
        if '.' not in filename:
            path = self.session_dir / filename
            ensure_dir(path)
            return path
        
        if subdir == "data":
            return self.data_dir / filename
        elif subdir == "clusters":
            return self.clusters_dir / filename
        elif subdir == "evaluation":
            return self.evaluation_dir / filename
        elif subdir == "figures":
            return self.figures_dir / filename
        elif subdir == "logs":
            return self.logs_dir / filename
        else:
            # For custom subdirs
            custom_dir = self.session_dir / subdir
            ensure_dir(custom_dir)
            return custom_dir / filename

    def copy_input_file(self, source: Path, dest_name: Optional[str] = None) -> Path:
        """Copy an input file to the session's data directory.
        
        Args:
            source: Path to the source file
            dest_name: Optional destination filename (uses source name if None)
            
        Returns:
            Path to the copied file
        """
        source = Path(source)
        dest_name = dest_name or source.name
        dest = self.data_dir / dest_name
        
        shutil.copy2(source, dest)
        self.logger.info(f"Copied {source} to {dest}")
        
        return dest
    
    def log_step(self, step_name: str, status: str = "started", **kwargs):
        """Log a pipeline step.
        
        Args:
            step_name: Name of the step
            status: Status of the step ('started', 'completed', 'failed')
            **kwargs: Additional information to log
        """
        msg = f"Step '{step_name}': {status}"
        if kwargs:
            msg += f" - {kwargs}"
        
        if status == "failed":
            self.logger.error(msg)
        else:
            self.logger.info(msg)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.logger.info(f"Session completed successfully: {self.session_config.run_id}")
        else:
            self.logger.error(
                f"Session failed: {self.session_config.run_id} - "
                f"{exc_type.__name__}: {exc_val}"
            )
        return False


def list_sessions(base_dir: str = "runs") -> list:
    """List all existing sessions.
    
    Args:
        base_dir: Base directory containing session folders
        
    Returns:
        List of session info dictionaries
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    sessions = []
    for session_dir in sorted(base_path.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        
        config_path = session_dir / "config.json"
        summary_path = session_dir / "summary.json"
        
        info = {
            "run_id": session_dir.name,
            "path": str(session_dir),
            "has_config": config_path.exists(),
            "has_summary": summary_path.exists(),
        }
        
        # Try to load summary for more info
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                info["completed_at"] = summary.get("completed_at")
                info["status"] = summary.get("status", "unknown")
            except Exception:
                pass
        
        sessions.append(info)
    
    return sessions


def load_session(run_id: str, base_dir: str = "runs") -> Session:
    """Load an existing session.
    
    Args:
        run_id: The session run ID
        base_dir: Base directory containing session folders
        
    Returns:
        Session instance
    """
    session_dir = Path(base_dir) / run_id
    
    if not session_dir.exists():
        raise ValueError(f"Session not found: {run_id}")
    
    config_path = session_dir / "config.json"
    
    if config_path.exists():
        from ..config.settings import load_config
        config = load_config(str(config_path))
        return Session(config=config)
    else:
        session_config = SessionConfig(
            base_output_dir=base_dir,
            run_id=run_id,
        )
        return Session(session_config=session_config)
