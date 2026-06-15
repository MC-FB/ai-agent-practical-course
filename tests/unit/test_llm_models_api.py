from __future__ import annotations

from http import HTTPStatus

from app import api
from dagqa.config import AppConfig


def _azure_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "llm": {
                "provider": "azure_openai",
                "model": "azure/gpt-4o-mini",
                "api_key_env": "AZURE_OPENAI_API_KEY",
                "api_base_env": "AZURE_OPENAI_ENDPOINT",
                "api_version_env": "AZURE_OPENAI_API_VERSION",
            }
        }
    )


def test_config_for_cluster_selection_builds_openai_compatible_config(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    config = api.config_for_selection(
        api.LLMSelection(provider="cluster", model="google/gemma-4-31B-it")
    )

    assert config.llm.provider == "cluster"
    assert config.llm.model == "google/gemma-4-31B-it"
    assert config.llm.api_key_env == "CLUSTER_API_KEY"
    assert config.llm.api_base == api.CLUSTER_API_BASE


def test_config_for_selection_applies_partial_planner_override(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)
    base = _azure_config().planner

    override_max_nodes = 5
    config = api.config_for_selection(planner=api.PlannerSelection(max_nodes=override_max_nodes))

    assert config.planner.max_nodes == override_max_nodes
    assert config.planner.max_depth == base.max_depth
    assert config.planner.repair_rounds == base.repair_rounds


def test_config_for_selection_without_planner_keeps_config_defaults(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)
    base = _azure_config().planner

    assert api.config_for_selection().planner.max_nodes == base.max_nodes
    # An override with no set fields is a no-op, same as passing nothing.
    assert api.config_for_selection(planner=api.PlannerSelection()).planner.max_nodes == (
        base.max_nodes
    )


async def test_list_llm_models_combines_azure_and_live_cluster_catalog(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        return ["Qwen/Qwen3.5-122B-A10B", "google/gemma-4-31B-it", "openai/gpt-oss-120b"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.list_llm_models()

    assert result["default"] == {"provider": "cluster", "model": "Qwen/Qwen3.5-122B-A10B"}
    assert [item["model"] for item in result["models"]] == [
        "azure/gpt-4o-mini",
        "Qwen/Qwen3.5-122B-A10B",
        "google/gemma-4-31B-it",
        "openai/gpt-oss-120b",
    ]


async def test_list_llm_models_uses_first_available_cluster_model_when_preferred_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        return ["custom/current-model"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.list_llm_models()

    assert result["default"] == {"provider": "cluster", "model": "custom/current-model"}


async def test_validated_config_rejects_unavailable_cluster_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        return ["Qwen/Qwen3.5-122B-A10B"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    try:
        await api.validated_config_for_selection(
            api.LLMSelection(provider="cluster", model="openai/gpt-oss-120b")
        )
    except api.HTTPException as exc:
        assert exc.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert "Qwen/Qwen3.5-122B-A10B" in exc.detail
    else:
        raise AssertionError("Expected unavailable cluster model to be rejected.")


async def test_validated_config_accepts_current_cluster_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        return ["Qwen/Qwen3.5-122B-A10B"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    config = await api.validated_config_for_selection(
        api.LLMSelection(provider="cluster", model="Qwen/Qwen3.5-122B-A10B")
    )

    assert config.llm.provider == "cluster"
    assert config.llm.model == "Qwen/Qwen3.5-122B-A10B"


async def test_list_llm_models_keeps_azure_available_when_cluster_fails(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        raise api.HTTPException(status_code=503, detail="VPN unavailable")

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.list_llm_models()

    assert len(result["models"]) == 1
    assert result["models"][0]["provider"] == "azure_openai"
    assert result["cluster_error"] == "VPN unavailable"
