import streamlit as st

from utils.i18n import current_language


def render_about_page():
    is_tr = current_language() == "tr"
    tr = lambda tr_text, en_text: tr_text if is_tr else en_text

    st.subheader(tr("Signify Hakkında", "About Signify"))
    st.caption(tr(
        "Yapay zekâ destekli, çok motorlu imza doğrulama sistemi",
        "AI-powered, multi-engine signature verification system",
    ))

    st.write(tr(
        "Signify; referans imzalar ile kontrol edilen imzayı karşılaştırır, görüntü profiline uygun doğrulama motorunu seçer ve sonucu Gerçek, Belirsiz veya Sahte olarak raporlar.",
        "Signify compares reference signatures with a questioned signature, selects the verification engine suited to the image profile, and reports the result as Genuine, Uncertain, or Forgery.",
    ))

    st.markdown(tr("### Nasıl çalışır?", "### How it works"))
    steps = [
        tr("İmza görüntüleri yüklenir veya uygulama kamerasıyla çekilir.", "Signature images are uploaded or captured with the in-app camera."),
        tr("Görüntüler temizlenir ve modelin gördüğü 224×224 girişe dönüştürülür.", "Images are cleaned and converted into the 224×224 input seen by the model."),
        tr("Otomatik yönlendirme uygun doğrulama motorunu seçer.", "Automatic routing selects the appropriate verification engine."),
        tr("Bir veya birden fazla referans karşılaştırılarak sonuç ve karar güveni üretilir.", "One or more references are compared to produce a result and decision confidence."),
    ]
    for index, step in enumerate(steps, start=1):
        st.markdown(f"**{index}.** {step}")

    st.markdown(tr("### Doğrulama Motorları", "### Verification Engines"))
    engine_rows = [
        {
            tr("Motor", "Engine"): "General v1",
            tr("Kullanım", "Use case"): tr("Genel fotoğraflar ve standart taramalar", "General photos and standard scans"),
            tr("Teknik yapı", "Technical stack"): "Siamese CNN · CEDAR + selected UTSig",
        },
        {
            tr("Motor", "Engine"): "Low-resolution scan v2",
            tr("Kullanım", "Use case"): tr("Kompakt, renkli ve düşük çözünürlüklü imzalar", "Compact, colored, low-resolution signatures"),
            tr("Teknik yapı", "Technical stack"): "MobileNetV3 · dual pair heads · archive collection",
        },
        {
            tr("Motor", "Engine"): "High-resolution mono v3",
            tr("Kullanım", "Use case"): tr("Yüksek çözünürlüklü tek renk taramalar", "High-resolution monochrome scans"),
            tr("Teknik yapı", "Technical stack"): "MobileNetV3 · UTSig pair head",
        },
    ]
    st.dataframe(engine_rows, use_container_width=True, hide_index=True)

    st.markdown(tr("### Veri ve Gizlilik", "### Data & Privacy"))
    st.write(tr(
        "Model çıkarımı uygulamanın çalıştığı bilgisayarda yerel olarak gerçekleştirilir. Kayıtlı kişi bilgileri ve referans imzalar yerel veritabanı ve yönetilen dosya alanında saklanır. Varsayılan başlatıcı uygulamayı yalnızca bu bilgisayardan erişilebilir şekilde açar.",
        "Model inference runs locally on the computer hosting the application. Registered identities and reference signatures are stored in the local database and managed file storage. The default launcher makes the application accessible only from this computer.",
    ))

    st.warning(tr(
        "Signify karar destek amaçlı bir MVP'dir. Belirsiz sonuçlarda daha net veya ek imza örnekleri kullanılmalı; yüksek riskli işlemlerde sonuç uzman incelemesiyle birlikte değerlendirilmelidir.",
        "Signify is a decision-support MVP. Uncertain results should be retried with clearer or additional samples, and high-risk decisions should include expert review.",
    ))

    col1, col2, col3 = st.columns(3)
    col1.metric(tr("Sürüm", "Version"), "MVP")
    col2.metric(tr("Doğrulama motoru", "Verification engines"), "3")
    col3.metric(tr("Altyapı", "Framework"), "PyTorch + Streamlit")
