import os
from pathlib import Path
from ascend_arena_e2e.agent_base import register_agent, load_agent_profiles
from ascend_arena_e2e.agents.claude_code import ClaudeCodeAgent

@register_agent
class CannbotAgent(ClaudeCodeAgent):
    name = "cannbot"

    def __init__(self, **kwargs):
        profile = load_agent_profiles()["claude"]
        values = {k: v for k, v in profile.items() if k != "harness" and v}
        values.update(kwargs)
        super().__init__(**values)

    def _skill_path(self):
        path = Path(os.environ["CANNBOT_SKILL_PATH"]).resolve()
        if not path.is_file():
            raise ValueError("Cannbot skill file is missing")
        return path

    def preflight(self, *, docker=False):
        super().preflight(docker=docker)
        self._skill_path()

    def setup_agents_md(self, workspace):
        super().setup_agents_md(workspace)
        with (workspace / "AGENTS.md").open("a") as out:
            out.write("\n## Cannbot skill\nRead and follow " + str(self._skill_path()) + " before editing.\n")
