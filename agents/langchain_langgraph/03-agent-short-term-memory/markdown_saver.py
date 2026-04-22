import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
)

"""
markdown_saver.py — LangGraph-compatible checkpointer that persists
conversation memory as human-readable Markdown files.

Each thread gets its own .md file under the checkpoints/ directory:
  checkpoints/
    thread_alice.md
    thread_bob.md
    ...
"""

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def _thread_path(thread_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
    return CHECKPOINTS_DIR / f"thread_{safe}.md"


def _serialize_messages(messages: list) -> str:
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):          # multi-part content blocks
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        lines.append(f"**{role}**: {content}")
    return "\n\n".join(lines)


class MarkdownSaver(BaseCheckpointSaver):
    """
    A file-based LangGraph checkpointer that writes conversation state
    to Markdown files — no database required.

    Usage:
        with MarkdownSaver() as checkpointer:
            checkpointer.setup()
            agent = create_agent(..., checkpointer=checkpointer)
    """

    def setup(self) -> None:
        """Create the checkpoints directory (idempotent)."""
        CHECKPOINTS_DIR.mkdir(exist_ok=True)
        print(f"[MarkdownSaver] Checkpoint directory: {CHECKPOINTS_DIR.resolve()}")

    # Read 
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        path = _thread_path(thread_id)
        if not path.exists():
            return None

        raw = path.read_text(encoding="utf-8")
        # Parse the last JSON blob stored in the file
        blocks = [b.strip() for b in raw.split("```json") if "```" in b]
        if not blocks:
            return None

        last_json = blocks[-1].split("```")[0].strip()
        data = json.loads(last_json)
        checkpoint: Checkpoint = data["checkpoint"]
        metadata: CheckpointMetadata = data.get("metadata", {})
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        path = _thread_path(thread_id)
        if not path.exists():
            return

        raw = path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in raw.split("```json") if "```" in b]
        results = []
        for block in blocks:
            json_str = block.split("```")[0].strip()
            try:
                data = json.loads(json_str)
                results.append(
                    CheckpointTuple(
                        config=config,
                        checkpoint=data["checkpoint"],
                        metadata=data.get("metadata", {}),
                        parent_config=None,
                    )
                )
            except json.JSONDecodeError:
                continue

        for item in (results[-limit:] if limit else results):
            yield item

    # Write 
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        path = _thread_path(thread_id)
        checkpoint_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Pretty human-readable section
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        readable = _serialize_messages(messages) if messages else "_no messages yet_"

        section = (
            f"\n## Checkpoint `{checkpoint_id}`\n"
            f"### {timestamp}\n\n"
            f"{readable}\n\n"
            f"```json\n{json.dumps({'checkpoint': checkpoint, 'metadata': metadata}, indent=2, default=str)}\n```\n"
            f"\n---\n"
        )

        if not path.exists():
            path.write_text(f"# Thread: {thread_id}\n{section}", encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(section)

        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}

    def put_writes(self, config: RunnableConfig, writes: Any, task_id: str) -> None:
        # Intermediate writes — not persisted separately for simplicity
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass