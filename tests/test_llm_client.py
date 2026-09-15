from pydantic import SecretStr

from geoagent.core.config import load_app_config
from geoagent.models.llm_client import LLMClient
from geoagent.models.vlm_client import VLMClient


class _StreamingResponse:
    status_code = 200
    ok = True
    text = ""

    def iter_lines(self, decode_unicode=False):
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}'
        yield b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
        yield b"data: [DONE]"

    def close(self):
        return None


def test_brain_model_sends_deepseek_thinking_payload(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.brain_api_key = SecretStr("test-key")
    config.env.brain_base_url = "https://api.deepseek.com"
    seen = {}

    def fake_post_chat_completion(self, payload, stream=False):
        seen["payload"] = payload
        return {
            "id": "chatcmpl-test",
            "model": payload["model"],
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {},
        }

    monkeypatch.setattr(LLMClient, "_post_chat_completion", fake_post_chat_completion)

    client = LLMClient(app_config=config, model_role="brain")
    response = client.generate("Return JSON.", json_mode=True)

    assert response["text"] == '{"ok": true}'
    # Thinking is disabled for all roles in configs/models.yaml, so the client
    # sends enable_thinking=False instead of a thinking block.
    assert "thinking" not in seen["payload"]
    assert seen["payload"]["enable_thinking"] is False
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["payload"]["response_format"] == {"type": "json_object"}


def test_brain_backed_vlm_sends_both_thinking_compatibility_fields(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.brain_api_key = SecretStr("test-key")
    config.env.brain_base_url = "https://brain.example.com"
    config.env.brain_model = "brain-vision-model"
    seen = {}

    def fake_post_chat_completion(self, payload, stream=False):
        seen["payload"] = payload
        return {
            "id": "chatcmpl-test",
            "model": payload["model"],
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {},
        }

    monkeypatch.setattr(
        VLMClient,
        "image_data_url",
        classmethod(lambda cls, image_path: "data:image/jpeg;base64,eA=="),
    )
    monkeypatch.setattr(VLMClient, "_post_chat_completion", fake_post_chat_completion)

    client = VLMClient(app_config=config)
    response = client.generate("Inspect.", image_path="unused.jpg", json_mode=True)

    assert response["text"] == '{"ok": true}'
    assert client.api_key.get_secret_value() == "test-key"
    assert client.base_url == "https://brain.example.com"
    assert client.model_name == "brain-vision-model"
    assert seen["payload"]["enable_thinking"] is False
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_memory_manager_prefers_role_credentials_and_falls_back_to_brain():
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.brain_api_key = SecretStr("brain-key")
    config.env.brain_base_url = "https://brain.example.com"
    config.env.brain_model = "brain-model"

    fallback_client = LLMClient(app_config=config, model_role="memory_manager")
    assert fallback_client.api_key.get_secret_value() == "brain-key"
    assert fallback_client.base_url == "https://brain.example.com"
    assert fallback_client.model_name == "brain-model"

    config.env.memory_manager_api_key = SecretStr("memory-key")
    config.env.memory_manager_base_url = "https://memory.example.com"
    config.env.memory_manager_model = "memory-model"
    dedicated_client = LLMClient(app_config=config, model_role="memory_manager")
    assert dedicated_client.api_key.get_secret_value() == "memory-key"
    assert dedicated_client.base_url == "https://memory.example.com"
    assert dedicated_client.model_name == "memory-model"


def test_missing_credentials_raise_config_error():
    config = load_app_config(config_dir="configs", env_file=None)
    client = LLMClient(app_config=config, model_role="brain")
    try:
        client.generate("test")
        raise AssertionError("should have raised")
    except Exception as exc:  # noqa: BLE001
        assert "BRAIN_API_KEY" in str(exc)


def test_streaming_clients_request_and_capture_usage(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.brain_api_key = SecretStr("brain-key")
    config.env.brain_base_url = "https://brain.example.com"
    payloads = []

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return _StreamingResponse()

    monkeypatch.setattr("geoagent.models.llm_client.requests.post", fake_post)
    monkeypatch.setattr("geoagent.models.vlm_client.requests.post", fake_post)

    llm_data = LLMClient(app_config=config, model_role="brain")._stream_chat_completion({"model": "brain"})
    vlm_data = VLMClient(app_config=config)._stream_chat_completion({"model": "vlm"})

    assert all(payload["stream_options"] == {"include_usage": True} for payload in payloads)
    assert llm_data["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert vlm_data["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
