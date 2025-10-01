"""Interactive CLI wizard used to generate run configuration blueprints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_BLUEPRINT_DIR = Path("blueprints")


InputFn = Callable[[str], str]


@dataclass(slots=True)
class Blueprint:
    name: str
    payload: Dict[str, object]
    source_path: Optional[Path] = None


@dataclass(slots=True)
class BlueprintStore:
    root: Path = field(default_factory=lambda: DEFAULT_BLUEPRINT_DIR)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def list_blueprints(self) -> List[Blueprint]:
        items: List[Blueprint] = []
        for path in sorted(self.root.glob("*.yml")):
            with path.open("r", encoding="utf8") as handle:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError:
                    data = {}
            items.append(Blueprint(name=path.stem, payload=data, source_path=path))
        return items

    def save(self, blueprint: Blueprint) -> Path:
        name = blueprint.name or "blueprint"
        path = self.root / f"{name}.yml"
        with path.open("w", encoding="utf8") as handle:
            json.dump(blueprint.payload, handle, indent=2)
        return path


@dataclass
class WizardResult:
    blueprint: Blueprint
    runtime_config: Dict[str, object]


class InitWizard:
    """CLI workflow for building or loading strategy blueprints."""

    def __init__(self, *, store: Optional[BlueprintStore] = None, input_fn: InputFn | None = None) -> None:
        self.store = store or BlueprintStore()
        self.input = input_fn or input

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> WizardResult:
        mode = self._prompt_mode()
        if mode == "load":
            blueprint = self._select_existing_blueprint()
            if not blueprint:
                raise RuntimeError("No saved blueprints available")
            payload = self._maybe_edit_blueprint(blueprint)
        else:
            payload = self._collect_from_scratch()
            blueprint_name = self._prompt("Name for the strategy (leave blank to auto-generate): ")
            if not blueprint_name:
                blueprint_name = self._auto_name(payload)
            blueprint = Blueprint(name=blueprint_name, payload=payload)
            self.store.save(blueprint)

        runtime = {
            "strategy": payload["strategy"],
            "symbols": payload["symbols"],
            "timeframe": payload["timeframe"],
            "portfolio": payload["portfolio"],
            "auto_liquidate_on_close": payload.get("auto_liquidate_on_close", False),
        }
        return WizardResult(blueprint=blueprint, runtime_config=runtime)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prompt(self, message: str, *, default: Optional[str] = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        response = self.input(f"{message}{suffix}")
        if not response and default is not None:
            return default
        return response

    def _prompt_mode(self) -> str:
        while True:
            choice = self._prompt("Build from scratch or Load saved strategy? (build/load): ", default="build")
            choice = choice.strip().lower()
            if choice in {"build", "load"}:
                return choice
            print("Please enter 'build' or 'load'.")

    # Build path -------------------------------------------------------
    def _collect_from_scratch(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        payload["symbols"] = self._prompt("Step 1 - Symbols (comma separated): ")
        payload["strategy"] = self._prompt("Step 2 - Strategy FQN: ")
        payload["timeframe"] = self._prompt("Step 3 - Timeframe: ")
        payload["portfolio"] = self._prompt("Step 4 - Starting cash: ")
        payload["auto_liquidate_on_close"] = self._prompt_toggle(
            "Enable auto-liquidation at end of run? (y/N): ",
            default=False,
        )
        payload.update(self._prompt_advanced_options())
        return payload

    def _prompt_toggle(self, message: str, *, default: bool) -> bool:
        while True:
            choice = self._prompt(message, default="Y" if default else "N").strip().lower()
            if choice in {"y", "yes"}:
                return True
            if choice in {"n", "no"}:
                return False
            print("Please answer y or n.")

    def _prompt_advanced_options(self, existing: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        default_value = json.dumps(existing or {}, indent=2)
        payload["fees"] = self._prompt(
            "Step 7 - Advanced fee settings (JSON, optional): ",
            default=default_value,
        )
        try:
            payload["fees"] = json.loads(payload["fees"])
        except json.JSONDecodeError:
            print("Invalid JSON supplied for fees. Falling back to empty object.")
            payload["fees"] = {}
        return payload

    # Load path --------------------------------------------------------
    def _select_existing_blueprint(self) -> Optional[Blueprint]:
        blueprints = self.store.list_blueprints()
        if not blueprints:
            return None

        print("Saved strategies:")
        for idx, blueprint in enumerate(blueprints, start=1):
            print(f"  [{idx}] {blueprint.name}")

        while True:
            choice = self._prompt("Select a saved strategy: ")
            if not choice:
                return None
            try:
                index = int(choice) - 1
            except ValueError:
                print("Enter a number from the list.")
                continue
            if 0 <= index < len(blueprints):
                return blueprints[index]
            print("Invalid selection.")

    def _maybe_edit_blueprint(self, blueprint: Blueprint) -> Dict[str, object]:
        payload = dict(blueprint.payload)
        edit_choice = self._prompt(
            "Loaded saved strategy. Edit configuration before running? (y/N): ",
            default="n",
        ).strip().lower()
        if edit_choice in {"", "n", "no"}:
            return payload

        step_keys = ["symbols", "strategy", "timeframe", "portfolio"]
        for key in step_keys:
            existing = payload.get(key, "")
            prompt = f"Edit {key}?"
            payload[key] = self._prompt(f"{prompt}: ", default=str(existing))
        payload["auto_liquidate_on_close"] = self._prompt_toggle(
            "Enable auto-liquidation at end of run? (y/N): ",
            default=bool(payload.get("auto_liquidate_on_close", False)),
        )
        payload.update(self._prompt_advanced_options(existing=payload.get("fees")))
        return payload

    def _auto_name(self, payload: Dict[str, object]) -> str:
        strategy = str(payload.get("strategy", "strategy")).split(".")[-1]
        timeframe = payload.get("timeframe", "")
        symbols = str(payload.get("symbols", "")).replace(",", "-")
        return f"{strategy}-{symbols}-{timeframe}".strip("-")


__all__ = ["InitWizard", "WizardResult", "BlueprintStore", "Blueprint"]


def main() -> None:
    wizard = InitWizard()
    result = wizard.run()
    print("Generated runtime config:")
    print(json.dumps(result.runtime_config, indent=2))


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    main()
