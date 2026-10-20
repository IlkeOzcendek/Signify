from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def missing_model_assets(engine, project_root=None):
    root = Path(project_root) if project_root is not None else PROJECT_ROOT

    if engine == "general":
        from verify import MODEL_PATH

        required = [Path(MODEL_PATH), Path("best_threshold.json")]
    elif engine == "archive":
        required = [
            Path("models/archive_v2_binary_head.pth"),
            Path("models/archive_v2_gray_head.pth"),
        ]
    elif engine == "utsig":
        required = [Path("models/utsig_v3_binary_head.pth")]
    else:
        raise ValueError(f"Unknown verification engine: {engine}")

    return [
        str(path) if not path.is_absolute() else path.name
        for path in required
        if not (path if path.is_absolute() else root / path).is_file()
    ]


def ensure_engine_ready(engine):
    missing = missing_model_assets(engine)
    if not missing:
        return True

    import streamlit as st
    from utils.i18n import current_language

    if current_language() == "tr":
        st.warning(
            "Bu motorun model dosyaları henüz kurulmamış. Karşılaştırma için "
            "README içindeki Model setup bölümünü izleyin. Görüntü yükleme, "
            "ön işleme ve kayıt yönetimini kullanmaya devam edebilirsiniz."
        )
        st.caption("Eksik dosyalar:")
    else:
        st.warning(
            "This engine's model files are not installed yet. Follow Model setup "
            "in the README before running a comparison. Image upload, preprocessing "
            "and reference management remain available."
        )
        st.caption("Missing files:")
    st.code("\n".join(missing), language=None)
    return False
