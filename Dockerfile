# Prompt-injection and safety classifier API: DeBERTa-v3 injection classifier + Qwen3Guard-Gen-0.6B.
# Source: https://github.com/bon5co/promptguard-railway
# CPU only (torch), weights baked in (about 2.3 GB), Bearer API_KEY, stateless (no volume needed).
# upstream: pytorch/pytorch torch 2.15.0 and huggingface/transformers 5.24.0 (newest releases, checked 2026-11-22) are pinned in requirements.txt
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(torch|transformers|tokenizers|safetensors|numpy) "

# Weights, each pinned to one Hugging Face commit and verified by sha256 so a changed or replaced file
# fails the build instead of shipping. A download at boot would sit inside Railway's healthcheck window.
#   protectai/deberta-v3-base-prompt-injection-v2  Apache-2.0 (model card + LICENSE file)
#   Qwen/Qwen3Guard-Gen-0.6B                        Apache-2.0 (model card + LICENSE file)
ARG HF_INJ=https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/resolve/90c9989b1a342275dd0d1a95aad283c04e075671
ARG HF_GUARD=https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B/resolve/fada3b2f655b89601929198343c94cd2f64d93cc
ADD --checksum=sha256:05079f4735092040b780d459027afab413faa6eeb66a548571a58832304b60bb ${HF_INJ}/config.json /models/deberta-injection/config.json
ADD --checksum=sha256:6521cb8d0ac08148c81464899c424e6148fcc62befa371089fa4061d8b6e0424 ${HF_INJ}/model.safetensors /models/deberta-injection/model.safetensors
ADD --checksum=sha256:c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd ${HF_INJ}/spm.model /models/deberta-injection/spm.model
ADD --checksum=sha256:f0a66ad0d735d8dca9ecac4ff50fcdef4bb6adbadd2941a926844844d2c2059b ${HF_INJ}/tokenizer.json /models/deberta-injection/tokenizer.json
ADD --checksum=sha256:557b3d33d3f41b81ad769244e506549e98a1857d41dd58160aacd4d98d710b5a ${HF_INJ}/tokenizer_config.json /models/deberta-injection/tokenizer_config.json
ADD --checksum=sha256:9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9 ${HF_INJ}/special_tokens_map.json /models/deberta-injection/special_tokens_map.json
ADD --checksum=sha256:dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f ${HF_INJ}/added_tokens.json /models/deberta-injection/added_tokens.json
ADD --checksum=sha256:59899c6091b540582ed617e8eeaac4919dc985ccfc35459ee9752b699be5205b ${HF_INJ}/LICENSE /models/deberta-injection/LICENSE
ADD --checksum=sha256:4674816bdf440c113a9ca87aedfa069b67f7cabbd89e71da7d4e4a68adce4af4 ${HF_GUARD}/config.json /models/qwen3guard/config.json
ADD --checksum=sha256:5f0bc42aae9f779d06bf620ce685a02cb0cde802d0f27f05c7331aaf0e1fd2d7 ${HF_GUARD}/generation_config.json /models/qwen3guard/generation_config.json
ADD --checksum=sha256:8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 ${HF_GUARD}/merges.txt /models/qwen3guard/merges.txt
ADD --checksum=sha256:4f3ce47ebd968cddb67de08d8764f8ede7c410a7d1fb9e08145a4c7a2f2e5c0f ${HF_GUARD}/model.safetensors /models/qwen3guard/model.safetensors
ADD --checksum=sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 ${HF_GUARD}/tokenizer.json /models/qwen3guard/tokenizer.json
ADD --checksum=sha256:934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81 ${HF_GUARD}/tokenizer_config.json /models/qwen3guard/tokenizer_config.json
ADD --checksum=sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 ${HF_GUARD}/vocab.json /models/qwen3guard/vocab.json
ADD --checksum=sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e ${HF_GUARD}/LICENSE /models/qwen3guard/LICENSE

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin pg \
    && chmod -R a+rX /models
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
ENV INJECTION_MODEL_DIR=/models/deberta-injection SAFETY_MODEL_DIR=/models/qwen3guard
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
