"""Runtime patches and hooks for Qwen3-4B on TPU v5e-1."""

import builtins
import sys
import urllib.error
import urllib.request


def install_qwen_patches():
    # 1. Bypass GCE metadata polling 404 retries on Colab
    orig_urlopen = urllib.request.urlopen

    def _fast_urlopen(url, *args, **kwargs):
        u_str = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        if "metadata.google.internal" in u_str:
            raise urllib.error.URLError("Bypassed GCE metadata poll on Colab")
        return orig_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _fast_urlopen

    # 2. Token padding pruning hook
    def _wrap_get_token_paddings(orig_fn):
        def _wrapped(*args, **kwargs):
            paddings = orig_fn(*args, **kwargs)
            if isinstance(paddings, list) and len(paddings) > 2:
                pruned = sorted(set([paddings[0], paddings[-1]]))
                print(f"[SUBRUNNER_HOOK] Pruned token paddings {paddings} -> {pruned}", flush=True)
                return pruned
            return paddings

        _wrapped._is_pruned = True
        return _wrapped

    # 3. Suppress redundant clear_caches during weight load & structured decoding precompile
    orig_imp = builtins.__import__

    def custom_imp(name, *args, **kwargs):
        res = orig_imp(name, *args, **kwargs)
        if name.startswith("tpu_inference"):
            mod_w = sys.modules.get("tpu_inference.models.jax.utils.weight_utils")
            if mod_w and hasattr(mod_w, "jax"):
                mod_w.jax.clear_caches = lambda: None
            for m_name, m in list(sys.modules.items()):
                if m and m_name.startswith("tpu_inference"):
                    fn = getattr(m, "get_token_paddings", None)
                    if callable(fn) and not getattr(fn, "_is_pruned", False):
                        setattr(m, "get_token_paddings", _wrap_get_token_paddings(fn))
                    for c_name in ("TPUModelRunner", "CompilationManager"):
                        cls = getattr(m, c_name, None)
                        if cls and hasattr(cls, "_precompile_structured_decoding"):
                            setattr(cls, "_precompile_structured_decoding", lambda self, *a, **kw: None)
        return res

    builtins.__import__ = custom_imp
