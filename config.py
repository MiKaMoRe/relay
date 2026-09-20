from pathlib import Path

AGENTS_DIR = Path(__file__).parent / "agents"
DEFAULT_CATEGORY = "general"
VISION_CATEGORY = "vision"
MAX_HISTORY_MESSAGES = 12


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_agent_configs(agents_dir: Path) -> dict[str, dict[str, str]]:
    """
    Сканирует agents/*, пропуская служебные папки (начинающиеся с "_"),
    и собирает description/system_prompt/model для каждого агента.
    """
    configs: dict[str, dict[str, str]] = {}
    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        try:
            configs[folder.name] = {
                "description": _read(folder / "description.md"),
                "system_prompt": _read(folder / "system_prompt.md"),
                "model": _read(folder / "model.txt"),
            }
        except FileNotFoundError as e:
            raise RuntimeError(
                f"В папке agents/{folder.name}/ не хватает файла: {e.filename}. "
                "Нужны description.md, system_prompt.md и model.txt."
            ) from e
    return configs


AGENT_CONFIGS = load_agent_configs(AGENTS_DIR)

if DEFAULT_CATEGORY not in AGENT_CONFIGS:
    raise RuntimeError(
        f"Не найдена папка agents/{DEFAULT_CATEGORY}/ — она обязательна как категория по умолчанию."
    )

ROUTER_DIR = AGENTS_DIR / "_router"
ROUTER_MODEL = _read(ROUTER_DIR / "model.txt")
ROUTER_SYSTEM_TEMPLATE = _read(ROUTER_DIR / "system_prompt.md")

# Категории, которые участвуют в текстовой классификации роутера.
# vision туда не входит — она выбирается по факту наличия картинки, а не текстом.
ROUTABLE_CATEGORIES = [name for name in AGENT_CONFIGS if name != VISION_CATEGORY]


def build_router_system_prompt() -> str:
    categories_desc = "\n".join(
        f'- "{name}": {AGENT_CONFIGS[name]["description"]}'
        for name in ROUTABLE_CATEGORIES
    )
    return ROUTER_SYSTEM_TEMPLATE.replace("{categories}", categories_desc)


ROUTER_SYSTEM_PROMPT = build_router_system_prompt()
