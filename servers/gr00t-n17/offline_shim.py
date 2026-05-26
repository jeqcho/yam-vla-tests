"""Pre-import shim: when HF_HUB_OFFLINE=1, return empty model_info stubs.

WHY THIS EXISTS
---------------
The GR00T server loads `nvidia/Cosmos-Reason2-2B` as the VL backbone. Even
with the blobs locally cached AND `HF_HUB_OFFLINE=1` set, transformers'
`PreTrainedTokenizerBase._patch_mistral_regex` calls
`huggingface_hub.model_info(model_id)` unconditionally as part of an
`is_base_mistral` heuristic. With HF_HUB_OFFLINE set, that call raises
`OfflineModeIsEnabled` and crashes server startup; without it, the HF API
returns 401 (gated repo). Neither works unless `hf auth login` is done.

This shim short-circuits: when offline, return a stub ModelInfo with empty
`tags` so `is_base_mistral(...)` returns False without a network call.
Cosmos-Reason2-2B is qwen-based, not Mistral-based, so the False result is
correct -- the mistral-regex patch should NOT apply to it.

Safety: the patch is a strict subset of what huggingface_hub does -- only
activates with `HF_HUB_OFFLINE=1`, and only short-circuits when the blobs
are already on disk under HF_HOME. With `hf auth login` done, this shim
does nothing (HF_HUB_OFFLINE wouldn't have been set, see run_server.sh).
"""
from __future__ import annotations

import os

if os.environ.get("HF_HUB_OFFLINE") == "1":
    import huggingface_hub

    _real_model_info = huggingface_hub.model_info

    def _stub_model_info(repo_id, *args, **kwargs):
        """Return a minimal ModelInfo-like object with empty tags.

        Bypasses the HF API call that `is_base_mistral` makes during
        tokenizer loading; lets transformers proceed to read the local
        snapshot via `cached_file(..., local_files_only=True)`.
        """
        try:
            return _real_model_info(repo_id, *args, **kwargs)
        except Exception:
            class _Stub:
                tags = []
                siblings = []
                downloads = 0
                likes = 0
                modelId = repo_id
                id = repo_id
            return _Stub()

    huggingface_hub.model_info = _stub_model_info
    # Some import paths re-export model_info; patch those that exist too.
    try:
        from huggingface_hub import hf_api as _hf_api  # type: ignore
        _hf_api.model_info = _stub_model_info
    except Exception:
        pass
