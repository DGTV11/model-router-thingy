# from debug import printd
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict, Union, cast

# from config import LLM_CONFIG, VLM_CONFIG
from openai import OpenAI
from pydantic import BaseModel, RootModel, TypeAdapter, validator

logger = logging.getLogger(__name__)

# *Dataclasses

# LLMs and Backends


@dataclass
class LLM:
    name: str
    backend_name: str


@dataclass
class LLMBackend:
    name: str
    models: List[LLM]
    client: OpenAI


# Results


@dataclass
class LLMCallSuccess:
    response: str
    model: LLM


@dataclass
class LLMCallSingleFailure:
    model: LLM
    message: str


@dataclass
class LLMCallFailure:
    failures: List[LLMCallSingleFailure]


# *Config

# Pydantic Models


class LLMBackendConfig(TypedDict):
    name: str
    base_url: str
    api_key: str
    models: List[str]


LLMBackendsConfig = RootModel[Dict[str, List[LLMBackendConfig]]]


# Load

# llm_backends = [
#     (
#         backend["name"],
#         OpenAI(base_url=backend["base_url"], api_key=backend["api_key"], max_retries=0),
#         backend["models"],
#     )
#     for backend in LLM_CONFIG
# ]
# vlm_backends = [
#     (
#         backend["name"],
#         OpenAI(base_url=backend["base_url"], api_key=backend["api_key"], max_retries=0),
#         backend["models"],
#     )
#     for backend in VLM_CONFIG
# ]


def load_backends_config(
    backends_config: Dict[str, Any],
) -> Dict[str, List[LLMBackend]]:
    backends_config_validated = LLMBackendsConfig.model_validate(backends_config)
    return {
        backends_name: [
            LLMBackend(
                name=backend["name"],
                models=[
                    LLM(name=model, backend_name=backend["name"])
                    for model in backend["models"]
                ],
                client=OpenAI(
                    base_url=backend["base_url"],
                    api_key=backend["api_key"],
                    max_retries=0,
                ),
            )
            for backend in backends
        ]
        for backends_name, backends in backends_config_validated.root.items()
    }


# *Router


def call_router(
    messages: List[Dict[str, Union[str, Any]]],
    backends: List[LLMBackend],
    max_tokens=8192,
) -> Union[LLMCallSuccess, LLMCallFailure]:
    failures = []

    # for name, client, models in backends:
    for backend in backends:
        for model in backend.models:
            try:
                # raise Exception("Test error")
                completion = backend.client.chat.completions.create(
                    model=model.name,
                    messages=cast(Any, messages),
                    max_tokens=max_tokens,
                )

                message_content = completion.choices[0].message.content

                assert message_content, "Empty completion"

                logger.debug(f"Using backend {backend.name} model {model.name}")
                return LLMCallSuccess(response=message_content, model=model)
            except Exception as e:
                # print(backend.models, model)
                failures.append(LLMCallSingleFailure(model=model, message=str(e)))
                logger.debug(f"Backend {backend.name} model {model.name} failed: {e}")

    return LLMCallFailure(failures=failures)


# *FOR IMAGES
# {
#     "role": "user",
#     "content": [
#         {"type": "text", "text": prompt.strip()},
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/{img_type};base64,{b64_image}"
#             },
#         },
#     ],
# },


def extract_yaml(resp: str) -> Dict[str, Any]:
    matches = list(re.finditer(r"```(?:ya?ml)?\s*([\s\S]*?)```", resp, re.IGNORECASE))
    if matches:
        yaml_str = matches[-1].group(1).strip()
    else:
        yaml_str = re.sub(r"^<think>.*?</think>", "", resp, flags=re.DOTALL).strip()

    yaml_str_sanitised = re.sub(r"[\ud800-\udfff]", "", yaml_str)
    data = yaml.safe_load(yaml_str_sanitised)

    # recursively sanitize everything that might contain text after yaml loading
    def deep_clean(obj: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(obj, str):
            return re.sub(r"[\ud800-\udfff]", "", obj)
        elif isinstance(obj, list):
            return [deep_clean(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: deep_clean(v) for k, v in obj.items()}
        else:
            return obj

    return deep_clean(data)


# *test


def main():
    import yaml

    config = input("YAML config path: ").strip()

    with open(config, "r") as f:
        backends_config = load_backends_config(yaml.safe_load(f))
    print(backends_config)
    assert "llm_backends" in backends_config  # and "vlm_backends" in backends_config

    DEFAULT_PROMPT = (
        "Why is the sky blue? Explain using SVG diagram enclosed in code fences."
    )
    prompt = (
        input(f"Input test prompt (Default: '{DEFAULT_PROMPT}'): ").strip()
        or DEFAULT_PROMPT
    )
    result = call_router(
        messages=[{"role": "user", "content": prompt}],
        backends=backends_config["llm_backends"],
    )
    if isinstance(result, LLMCallSuccess):
        print(
            f"Successfully called backend {result.model.backend_name} model {result.model.name}:\n{result.response}"
        )
    elif isinstance(result, LLMCallFailure):
        print(
            f"All models failed:\n{'\n'.join(map(lambda f: f'backend {f.model.backend_name} model {f.model.name}: {f.message}', result.failures))}"
        )


if __name__ == "__main__":
    main()
