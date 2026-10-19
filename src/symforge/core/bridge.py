"""Bidirectional FFI/IPC bridge between Python and Lean 4 IR."""

from __future__ import annotations
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from symforge.core.ast import ASTNode, deserialize_node, serialize_node

logger = logging.getLogger(__name__)


class LeanBridge:
    """IPC Bridge communicating with the Lean 4 formal IR CLI."""

    def __init__(self, cli_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parents[3]
        default_cli = (
            self.project_root
            / "packages"
            / "lean"
            / "symforge_ir"
            / ".lake"
            / "build"
            / "bin"
            / "symforge_cli"
        )
        if cli_path:
            self.cli_path = Path(cli_path)
        else:
            self.cli_path = default_cli

        self.lean_bin_dir = Path("/home/racoci/.elan/bin")
        self._ensure_binary()

    def _ensure_binary(self) -> None:
        """Verify CLI executable exists; if not, build it using lake."""
        if self.cli_path.is_file() and os.access(self.cli_path, os.X_OK):
            return

        lake_pkg_dir = self.project_root / "packages" / "lean" / "symforge_ir"
        if not lake_pkg_dir.is_dir():
            return

        env = os.environ.copy()
        if self.lean_bin_dir.is_dir():
            env["PATH"] = f"{self.lean_bin_dir}:{env.get('PATH', '')}"

        lake_cmd = shutil.which("lake", path=env.get("PATH"))
        if not lake_cmd:
            return

        try:
            subprocess.run(
                [lake_cmd, "build"],
                cwd=str(lake_pkg_dir),
                env=env,
                check=True,
                capture_output=True,
                timeout=30.0,
            )
        except Exception as e:
            logger.warning(
                "Failed to build Lean CLI via Lake: %s (cwd=%s, env_path=%s)",
                e,
                lake_pkg_dir,
                env.get("PATH"),
            )

    def is_available(self) -> bool:
        """Check if Lean CLI binary is available and executable."""
        return self.cli_path.is_file() and os.access(self.cli_path, os.X_OK)

    def _execute_cmd(
        self,
        args: list[str],
        input_data: Optional[str] = None,
        timeout: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        """Execute symforge_cli with explicit timeout and context capture."""
        if not self.is_available():
            raise FileNotFoundError(
                f"Lean CLI executable not found at '{self.cli_path}'. Please build packages/lean/symforge_ir with Lake."
            )

        cmd = [str(self.cli_path)] + args
        try:
            return subprocess.run(
                cmd,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error(
                "Lean CLI execution timed out: cmd=%s, timeout=%s, input_len=%d",
                cmd, timeout, len(input_data or "")
            )
            raise TimeoutError(
                f"Lean CLI execution timed out after {timeout}s: cmd={cmd}, input_len={len(input_data or '')}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Lean CLI failed: cmd=%s, returncode=%d, stdout=%s, stderr=%s",
                cmd, exc.returncode, exc.stdout, exc.stderr
            )
            raise RuntimeError(
                f"Lean CLI failed (exit {exc.returncode}): stdout={exc.stdout!r}, stderr={exc.stderr!r}, cmd={cmd}"
            ) from exc

    def get_example_audio(self) -> ASTNode:
        """Fetch verified example audio AST from Lean."""
        proc = self._execute_cmd(["example-audio"])
        data = json.loads(proc.stdout.strip())
        return deserialize_node(data)

    def get_example_sdf(self) -> ASTNode:
        """Fetch verified example SDF AST from Lean."""
        proc = self._execute_cmd(["example-sdf"])
        data = json.loads(proc.stdout.strip())
        return deserialize_node(data)

    def validate_audio(self, node: ASTNode) -> Dict[str, Any]:
        """Validate an audio AST via Lean 4 deserialization."""
        serialized = json.dumps(serialize_node(node))
        proc = self._execute_cmd(["parse-audio"], input_data=serialized)
        return json.loads(proc.stdout.strip())

    def validate_sdf(self, node: ASTNode) -> Dict[str, Any]:
        """Validate an SDF AST via Lean 4 deserialization."""
        serialized = json.dumps(serialize_node(node))
        proc = self._execute_cmd(["parse-sdf"], input_data=serialized)
        return json.loads(proc.stdout.strip())
