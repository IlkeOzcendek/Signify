import pandas as pd
import plotly.express as px
import streamlit as st

from utils.history import load_verification_history
from utils.database import load_registered_people
from utils.helpers import THRESHOLD, T_DIFF, T_SAME
from utils.i18n import current_language
from verify_archive_v2 import HIGH_PROBABILITY, LOW_PROBABILITY
from verify_utsig_v3 import MANUAL_DIFFERENT_MAX, MANUAL_SAME_MIN


ENGINE_LABELS = {
    "general": "General v1",
    "archive": "Low-resolution scan v2",
    "utsig": "High-resolution mono v3",
}


def render_analysis_page():
    is_tr = current_language() == "tr"
    tr = lambda tr_text, en_text: tr_text if is_tr else en_text

    st.subheader(tr("Analiz ve Raporlar", "Analysis & Reports"))
    st.caption(tr(
        "Çok motorlu doğrulama sistemi, kayıtlı veriler ve geçmiş sonuçların özeti",
        "Overview of the multi-engine verification system, registered data, and historical results",
    ))

    st.divider()
    st.markdown(tr("### Doğrulama Sistemi", "### Verification System"))
    st.info(tr(
        "Signify yalnızca CEDAR'a bağlı tek bir model kullanmaz. Görüntü profiline göre üç doğrulama motorundan uygun olanı otomatik seçer.",
        "Signify is not a single CEDAR-only model. It automatically selects one of three verification engines based on the image profile.",
    ))

    engine_rows = [
        {
            tr("Motor", "Engine"): "General v1",
            tr("Mimari", "Architecture"): "Siamese CNN",
            tr("Görüntü profili", "Image profile"): tr("Genel fotoğraf ve standart tarama", "General photos and standard scans"),
            tr("Veri kaynağı", "Data source"): "CEDAR + selected UTSig",
            tr("Karar bölgesi", "Decision region"): f"match < {T_SAME:.2f} · uncertain {T_SAME:.2f}–{T_DIFF:.2f}",
        },
        {
            tr("Motor", "Engine"): "Low-resolution scan v2",
            tr("Mimari", "Architecture"): "MobileNetV3 + dual pair heads",
            tr("Görüntü profili", "Image profile"): tr("Kompakt ve renkli imza taramaları", "Compact, colored signature scans"),
            tr("Veri kaynağı", "Data source"): tr("Arşiv imza koleksiyonu", "Archive signature collection"),
            tr("Karar bölgesi", "Decision region"): f"different ≤ {LOW_PROBABILITY:.2f} · match ≥ {HIGH_PROBABILITY:.2f}",
        },
        {
            tr("Motor", "Engine"): "High-resolution mono v3",
            tr("Mimari", "Architecture"): "MobileNetV3 + pair head",
            tr("Görüntü profili", "Image profile"): tr("Yüksek çözünürlüklü tek renk tarama", "High-resolution monochrome scans"),
            tr("Veri kaynağı", "Data source"): "UTSig (person-held-out)",
            tr("Karar bölgesi", "Decision region"): f"different ≤ {MANUAL_DIFFERENT_MAX:.2f} · match ≥ {MANUAL_SAME_MIN:.2f}",
        },
    ]
    st.dataframe(pd.DataFrame(engine_rows), use_container_width=True, hide_index=True)
    st.caption(tr(
        f"General v1 ikili karar eşiği: {THRESHOLD:.3f}. Raporlarda her doğrulamada gerçekten kullanılan motor ayrıca belirtilir.",
        f"General v1 binary threshold: {THRESHOLD:.3f}. Reports also identify the engine actually used for each verification.",
    ))

    st.divider()
    st.markdown(tr("### Kayıtlı Veritabanı", "### Registered Database"))
    people = load_registered_people()
    total_signatures = sum(person["signature_count"] for person in people)

    col1, col2 = st.columns(2)
    col1.metric(tr("Kayıtlı Kişi", "Registered People"), len(people))
    col2.metric(tr("Referans İmza", "Reference Signatures"), total_signatures)

    if people:
        database_df = pd.DataFrame([
            {
                tr("Kişi", "Person"): person["name"],
                tr("Referans sayısı", "Reference count"): person["signature_count"],
                tr("Kayıt tarihi", "Registered at"): person.get("created_at"),
            }
            for person in people
        ])
        st.dataframe(database_df, use_container_width=True, hide_index=True)

        fig_bar = px.bar(
            database_df,
            x=tr("Kişi", "Person"),
            y=tr("Referans sayısı", "Reference count"),
            text=tr("Referans sayısı", "Reference count"),
            title=tr("Kişi Başına Referans İmza", "Reference Signatures per Person"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(tr("Henüz kayıtlı kişi bulunmuyor.", "No registered people found yet."))

    st.divider()
    st.markdown(tr("### Doğrulama Geçmişi", "### Verification History"))
    history_df = load_verification_history()

    if history_df.empty:
        st.info(tr("Henüz doğrulama geçmişi yok.", "No verification history yet."))
        return

    total_verifications = len(history_df)
    genuine_count = int((history_df["result"] == "Genuine").sum())
    uncertain_count = int((history_df["result"] == "Uncertain").sum())
    forgery_count = int((history_df["result"] == "Forgery").sum())

    cols = st.columns(4)
    cols[0].metric(tr("Toplam", "Total"), total_verifications)
    cols[1].metric(tr("Gerçek", "Genuine"), genuine_count)
    cols[2].metric(tr("Belirsiz", "Uncertain"), uncertain_count)
    cols[3].metric(tr("Sahte", "Forgery"), forgery_count)

    result_chart, engine_chart = st.columns(2)
    with result_chart:
        fig_result = px.pie(
            history_df,
            names="result",
            title=tr("Sonuç Dağılımı", "Result Distribution"),
        )
        st.plotly_chart(fig_result, use_container_width=True)

    with engine_chart:
        if "engine" in history_df.columns and history_df["engine"].notna().any():
            engine_history = history_df.copy()
            engine_history["engine_label"] = engine_history["engine"].map(ENGINE_LABELS).fillna(
                tr("Eski kayıt / bilinmiyor", "Legacy / unknown")
            )
            fig_engine = px.histogram(
                engine_history,
                x="engine_label",
                title=tr("Kullanılan Motorlar", "Engines Used"),
                labels={"engine_label": tr("Motor", "Engine")},
            )
            st.plotly_chart(fig_engine, use_container_width=True)
        else:
            st.info(tr(
                "Eski geçmiş kayıtlarında motor bilgisi tutulmamış. Yeni doğrulamalar burada motor bazında görünecek.",
                "Legacy history did not store engine information. New verifications will appear here by engine.",
            ))

    display_history = history_df.sort_values("date", ascending=False).copy()
    if "engine" in display_history.columns:
        display_history["engine"] = display_history["engine"].map(ENGINE_LABELS).fillna("—")
    st.markdown(tr("#### Son Doğrulamalar", "#### Recent Verifications"))
    st.dataframe(display_history, use_container_width=True, hide_index=True)
