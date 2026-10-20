import tempfile
from pathlib import Path

import streamlit as st

from preprocess import preprocess_signature
from utils.image_quality import inspect_uploaded_signature
from utils.i18n import t
from utils.reference_consistency import analyze_reference_consistency
from utils.model_assets import ensure_engine_ready
from utils.camera_capture import clear_camera_collection, render_camera_collection
from utils.helpers import T_DIFF, T_SAME
from verify_archive_v2 import HIGH_PROBABILITY, LOW_PROBABILITY
from verify_utsig_v3 import (
    MANUAL_DIFFERENT_MAX,
    MANUAL_SAME_MIN,
    REGISTERED_DIFFERENT_MAX,
    REGISTERED_SAME_MIN,
)
from utils.database import (
    load_registered_people,
    save_reference_signatures,
    save_profile_image,
    append_reference_signatures,
    delete_person_folder,
    delete_reference_signature,
    get_reference_paths,
    rename_person_folder,
    DATABASE_DIR,
)


def preview_preprocessing(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
        input_file.write(uploaded_file.getbuffer())
        input_path = input_file.name

    output_path = input_path.replace(suffix, f"_processed{suffix}")

    preprocess_signature(input_path, output_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            uploaded_file,
            caption=t("original_signature"),
            use_container_width=True
        )

    with col2:
        st.image(
            output_path,
            caption=t("processed_signature"),
            use_container_width=True
        )


def render_admin_page():
    if "deleted_message" in st.session_state:
        st.toast(st.session_state.deleted_message, icon="🗑️")
        del st.session_state.deleted_message

    st.session_state.people = load_registered_people()

    st.subheader(t("admin_title"))
    st.caption(t("admin_caption"))

    with st.expander("Model routing and thresholds / Motor yönlendirme ve eşikler"):
        st.info(
            "Automatic routing uses only image properties; it never reads the "
            "file name, person name, T/F label, or dataset folder. The verification "
            "screen shows the selected engine and reason. Its engine selector can "
            "be used as an admin/developer override for diagnosis."
        )
        st.dataframe([
            {
                "Engine": "General v1",
                "Profile": "General photo or standard grayscale scan",
                "Different": f"distance > {T_DIFF:.2f}",
                "Same": f"distance < {T_SAME:.2f}",
            },
            {
                "Engine": "Low-resolution scan v2",
                "Profile": "Small colored/blue-ink scan",
                "Different": f"probability ≤ {LOW_PROBABILITY:.2f}",
                "Same": f"probability ≥ {HIGH_PROBABILITY:.2f}",
            },
            {
                "Engine": "High-resolution mono v3 (manual)",
                "Profile": "High-resolution monochrome scan",
                "Different": f"probability ≤ {MANUAL_DIFFERENT_MAX:.2f}",
                "Same": f"probability ≥ {MANUAL_SAME_MIN:.2f}",
            },
            {
                "Engine": "High-resolution mono v3 (registered)",
                "Profile": "High-resolution monochrome scan + multi-reference",
                "Different": f"median ≤ {REGISTERED_DIFFERENT_MAX:.2f}",
                "Same": f"median ≥ {REGISTERED_SAME_MIN:.2f} + 60% support",
            },
        ], use_container_width=True, hide_index=True)
        st.caption(
            "Automatic rules: low-resolution colored majority → v2; "
            "height ≥450 px, width ≥650 px and monochrome majority → v3; otherwise v1."
        )

    st.divider()

    with st.container(border=True):
        st.markdown(f"### {t('add_person')}")
        st.caption(t("enrollment_intro"))
        st.info(t("enrollment_guidance"))

        st.markdown(f"#### 1 · {t('person_information')}")
        person_name = st.text_input(
            t("person_name"),
            placeholder=t("person_name_placeholder"),
        )
        profile_col, profile_help_col = st.columns([1, 3])
        with profile_col:
            profile_image = st.file_uploader(
                t("profile_photo"),
                type=["png", "jpg", "jpeg"],
                key="new_person_profile",
            )
        with profile_help_col:
            st.caption(t("profile_photo_help"))
            if profile_image is not None:
                st.image(profile_image, width=96)

        st.markdown(f"#### 2 · {t('add_reference_signatures')}")
        input_method = st.radio(
            t("reference_input_method"),
            ["Upload Files", "Take Photo"],
            format_func=lambda value: (
                t("upload_files") if value == "Upload Files" else t("take_photo")
            ),
            horizontal=True,
        )

        reference_signatures = []

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                t("upload_reference_signatures"),
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                reference_signatures = uploaded_files

        else:
            reference_signatures = render_camera_collection("new_person_references")

        if reference_signatures:
            st.success(t("admin_references_ready").format(count=len(reference_signatures)))
            with st.expander(t("preprocessing_preview")):
                preview_file = st.selectbox(
                    t("select_preview_signature"),
                    reference_signatures,
                    format_func=lambda file: file.name,
                )

                if st.button(t("preview_preprocessing"), use_container_width=True):
                    preview_preprocessing(preview_file)

        st.markdown(f"#### 3 · {t('save_registration')}")
        if st.button(t("add_person"), use_container_width=True):
            if person_name and reference_signatures:
                if len(reference_signatures) < 5:
                    st.warning(t("few_references_warning"))
                folder_name = save_reference_signatures(person_name, reference_signatures)
                save_profile_image(folder_name, profile_image)
                clear_camera_collection("new_person_references")

                st.session_state.people = load_registered_people()

                st.toast(t("person_saved").format(name=person_name), icon="✅")
                st.toast(
                    t("references_saved").format(count=len(reference_signatures)),
                    icon="📁",
                )
                st.rerun()
            else:
                st.toast(t("registration_missing_fields"), icon="⚠️")

    st.divider()

    st.markdown(f"### {t('registered_people')}")
    st.caption(t("registered_people_caption"))

    if len(st.session_state.people) == 0:
        st.info(t("no_registered_people"))
    else:
        for index, person in enumerate(st.session_state.people):
            person_card = st.container(border=True)
            with person_card:
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.markdown(f"#### 👤 {person['name']}")

                with col2:
                    st.metric(t("reference_count"), person["signature_count"])

                with col3:
                    if st.button(
                        t("delete_person"),
                        key=f"delete_{index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_delete = person["folder_name"]
                        st.rerun()

                if st.session_state.get("pending_delete") == person["folder_name"]:
                    st.warning(t("delete_person_confirmation").format(name=person["name"]))
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            t("confirm_delete"),
                            key=f"confirm_delete_{person['folder_name']}",
                            type="primary",
                            use_container_width=True,
                        ):
                            delete_person_folder(person["folder_name"])
                            st.session_state.people = load_registered_people()
                            st.session_state.deleted_message = t("person_deleted").format(
                                name=person["name"]
                            )
                            st.session_state.pop("pending_delete", None)
                            st.rerun()
                    with cancel_col:
                        if st.button(
                            t("cancel"),
                            key=f"cancel_delete_{person['folder_name']}",
                            use_container_width=True,
                        ):
                            st.session_state.pop("pending_delete", None)
                            st.rerun()

            with person_card.expander(f"{t('manage_refs')} — {person['name']}"):
                person_dir = DATABASE_DIR / person["folder_name"]
                reference_paths = get_reference_paths(person_dir)

                if len(reference_paths) < 5:
                    st.warning(
                        f"Only {len(reference_paths)} reference(s). Add at least "
                        "5 for multi-reference consensus."
                    )
                else:
                    st.success(
                        f"{len(reference_paths)} references available; "
                        "multi-reference consensus is active."
                    )

                if st.button(
                    t("analyze_refs"),
                    key=f"analyze_refs_{person['folder_name']}",
                    use_container_width=True,
                ):
                    if not ensure_engine_ready("general"):
                        return
                    with st.spinner("Analyzing..." if t("language") == "Language" else "Analiz ediliyor..."):
                        st.session_state[
                            f"consistency_{person['folder_name']}"
                        ] = analyze_reference_consistency(reference_paths)

                consistency = st.session_state.get(
                    f"consistency_{person['folder_name']}"
                )
                if consistency:
                    st.markdown(f"#### {t('consistency_title')}")
                    for item in consistency:
                        status = t("outlier") if item["is_outlier"] else t("consistent")
                        message = (
                            f"**{item['reference']}** — {status} · "
                            f"{t('median_distance')}: `{item['median_distance']:.3f}` · "
                            f"{t('matching_refs')}: "
                            f"`{item['matching_count']}/{item['compared_count']}`"
                        )
                        if item["is_outlier"]:
                            st.warning(message)
                        else:
                            st.success(message)

                st.markdown(f"#### {t('existing_refs')}")
                for reference_index, reference_path in enumerate(reference_paths):
                    image_col, name_col, action_col = st.columns([2, 3, 1])
                    with image_col:
                        st.image(reference_path, use_container_width=True)
                    with name_col:
                        st.write(reference_path.name)
                    with action_col:
                        if st.button(
                            t("remove"),
                            key=f"remove_ref_{person['folder_name']}_{reference_index}",
                        ):
                            delete_reference_signature(
                                person["folder_name"], reference_path.name
                            )
                            st.toast("Reference removed.", icon="🗑️")
                            st.rerun()

                st.markdown(f"#### {t('add_refs')}")
                new_references = st.file_uploader(
                    "Upload additional genuine signatures",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key=f"add_refs_{person['folder_name']}",
                )
                additions_usable = True
                if new_references:
                    for uploaded in new_references:
                        quality = inspect_uploaded_signature(uploaded)
                        preview_col, status_col = st.columns([1, 2])
                        with preview_col:
                            st.image(
                                quality.processed_image,
                                caption=f"Model view — {uploaded.name}",
                                use_container_width=True,
                            )
                        with status_col:
                            for message in quality.errors:
                                st.error(message)
                            for message in quality.warnings:
                                st.warning(message)
                            if quality.usable and not quality.warnings:
                                st.success("Quality looks suitable.")
                        additions_usable = additions_usable and quality.usable

                if st.button(
                    "Add selected references",
                    key=f"save_refs_{person['folder_name']}",
                    disabled=not new_references or not additions_usable,
                    use_container_width=True,
                ):
                    append_reference_signatures(
                        person["folder_name"], new_references
                    )
                    st.toast(
                        f"{len(new_references)} reference(s) added.", icon="✅"
                    )
                    st.rerun()

                st.markdown(f"#### {t('rename_person')}")
                new_name = st.text_input(
                    "New person name",
                    value=person["name"],
                    key=f"rename_input_{person['folder_name']}",
                )
                if st.button(
                    t("save_name"),
                    key=f"rename_{person['folder_name']}",
                    use_container_width=True,
                ):
                    try:
                        rename_person_folder(person["folder_name"], new_name)
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.toast("Person renamed.", icon="✅")
                        st.rerun()
