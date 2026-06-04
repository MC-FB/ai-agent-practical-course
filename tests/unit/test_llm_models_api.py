from __future__ import annotations

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


async def test_list_llm_models_combines_azure_and_live_cluster_catalog(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        return ["Qwen/Qwen3.5-122B-A10B", "google/gemma-4-31B-it", "openai/gpt-oss-120b"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.list_llm_models()

    assert result["default"] == {"provider": "cluster", "model": "openai/gpt-oss-120b"}
    assert [item["model"] for item in result["models"]] == [
        "azure/gpt-4o-mini",
        "Qwen/Qwen3.5-122B-A10B",
        "google/gemma-4-31B-it",
        "openai/gpt-oss-120b",
    ]


async def test_list_llm_models_keeps_azure_available_when_cluster_fails(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_config", _azure_config)

    async def cluster_models() -> list[str]:
        raise api.HTTPException(status_code=503, detail="VPN unavailable")

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.list_llm_models()

    assert len(result["models"]) == 1
    assert result["models"][0]["provider"] == "azure_openai"
    assert result["cluster_error"] == "VPN unavailable"
