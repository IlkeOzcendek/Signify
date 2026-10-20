import math
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image, UnidentifiedImageError

from utils.history import save_verification_history
from verify import verify_signature
from utils.ui import show_summary_card
from utils.report import create_verification_report
from utils.image_quality import inspect_uploaded_signature
from utils.enrollment import decide_registered_person
from utils.i18n import current_language, t
from utils.engine_detection import resolve_engine_details
from utils.model_assets import ensure_engine_ready
from utils.camera_capture import render_camera_collection, render_single_camera

from verify_archive_v2 import (
    HIGH_PROBABILITY,
    archive_confidence,
    archive_pair_probability,
    archive_verdict,
    decide_archive_registered,
)

from verify_utsig_v3 import (
    MANUAL_SAME_MIN as UTSIG_MANUAL_SAME_MIN,
    REGISTERED_SAME_MIN as UTSIG_REGISTERED_SAME_MIN,
    classify as utsig_classify,
    confidence as utsig_confidence,
    decide_registered as decide_utsig_registered,
    pair_probability as utsig_pair_probability,
)

from utils.database import (
    DATABASE_DIR,
    load_registered_people,
    save_uploaded_file,
    get_reference_paths
)

from utils.helpers import (
    THRESHOLD,
    T_SAME,
    T_DIFF,
    calculate_confidence,
    classify_verdict,
    VERDICT_LABELS,
)

DEFAULT_AVATAR_PATH = Path(__file__).resolve().parents[1] / "assets" / "default-avatar.svg" # Fallback profile image for people without an uploaded photo
PEOPLE_PER_PAGE = 6 # Maximum person cards shown on one selector page


def is_valid_uploaded_image(uploaded_file):
    """Validate an uploaded image without changing its position for callers."""

    try:
        uploaded_file.seek(0) # Moves the file cursor to the beginning before PIL reads it

        with Image.open(uploaded_file) as image:
            image.verify() # Checks image structure without fully decoding all pixels

        return True
    except (UnidentifiedImageError, OSError):
        return False
    finally:
        uploaded_file.seek(0) # Restores the cursor so later preview and save operations can read the same upload


def mode_label(value):
    """Return the shared localized label used by preprocessing mode controls."""

    return t("standard_mode") if value == "default" else t("sharp_mode")


def render_model_preview(uploaded_file, title, preprocess_mode = "sharp"):
    """Show the exact preprocessed input and return whether it is usable."""

    quality = inspect_uploaded_signature(uploaded_file, preprocess_mode = preprocess_mode) # QualityResult contains model input warnings errors and image metrics

    st.image(
        quality.processed_image,
        caption = f"{title} — exact 224×224 image seen by the model",
        use_container_width = True
    )

    for message in quality.errors:
        st.error(message)

    for message in quality.warnings:
        st.warning(message)

    with st.expander(f"{title} quality details"):
        st.write(f"Sharpness score: `{quality.blur_score:.1f}`")
        st.write(f"Ink ratio: `{quality.ink_ratio:.2%}`")

        if quality.usable and not quality.warnings:
            st.success("The image quality looks suitable.")

    return quality.usable


def render_verify_page():
    st.markdown(f"""
    <div class="verify-hero">
        <div class="verify-hero-icon"><span></span></div>
        <div class="verify-hero-copy">
            <h1>{t("verify_title")}</h1>
            <p>{t("verify_intro")}</p>
        </div>
        <div class="signature-illustration">
            <div class="signature-lines"></div>
            <div class="signature-stroke">S</div>
        </div>
    </div>
    """, unsafe_allow_html = True)

    st.markdown(f"### {t('verification_mode')}")

    with st.container(key="verification_mode_cards"):
        verify_mode = st.radio( # verify_mode = Manual comparison or registered-person comparison selected by the user
            t("verification_mode"),
            ["manual", "registered"],
            format_func=lambda value: (
                t("manual_mode_card") if value == "manual"
                else t("registered_mode_card")
            ),
            horizontal=True,
            label_visibility="collapsed",
        )

    with st.expander(t("advanced_settings")):
        st.caption(t("engine_help"))

        engine = st.radio(
            t("verification_engine"),
            ["auto", "general", "archive", "utsig"],
            format_func=lambda value: {
                "auto": t("engine_auto"),
                "general": t("engine_general"),
                "archive": t("engine_archive"),
                "utsig": t("engine_utsig"),
            }[value],
            key="verification_engine",
        )
        if engine == "archive":
            st.info(t("archive_engine_help"))

    # Preprocessing mode belongs to each image ------> references and questioned signature can use their own Standard or Sharp selection
    st.divider()

    if verify_mode == "manual":
        render_manual_compare(engine)
    else:
        render_registered_person_verify(engine)


def render_manual_compare(engine = "general"):
    st.subheader(t("upload_signatures"))
    st.markdown(f"""
    <div class="upload-guidance">
        <div class="guidance-icon">i</div>
        <div>
            <div class="guidance-title">{t("upload_guidance_title")}</div>
            <div class="guidance-copy">{t("reference_guidance")}</div>
        </div>
    </div>
    """, unsafe_allow_html = True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True, key="reference_upload_card"):
            st.markdown(f"### 1 · {t('reference_signature')}")
            st.caption(t("reference_upload_help"))
            reference_input = st.radio(
                "Reference input",
                ["Upload File", "Take Photo"],
                horizontal=True,
                key="manual_reference_input",
            )

            if reference_input == "Upload File":
                reference_files = st.file_uploader(
                    f"{t('reference_signature')} (1–5)",
                    type=["png", "jpg", "jpeg"],
                    key="manual_reference_multi",
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                st.caption(t("supported_files"))
            else:
                reference_files = render_camera_collection("manual_references")

            if len(reference_files) > 5:
                st.warning(t("reference_limit"))
                reference_files = reference_files[:5]

            if reference_files:
                st.success(t("reference_upload_ready").format(count=len(reference_files)))
            else:
                st.caption(t("waiting_for_reference"))

    with col2:
        with st.container(border=True, key="questioned_upload_card"):
            st.markdown(f"### 2 · {t('questioned_signature')}")
            st.caption(t("questioned_upload_help"))
            questioned_input = st.radio(
                "Questioned input",
                ["Upload File", "Take Photo"],
                horizontal=True,
                key="manual_questioned_input",
            )

            if questioned_input == "Upload File":
                questioned_file = st.file_uploader(
                    t("questioned_signature"),
                    type=["png", "jpg", "jpeg"],
                    key="manual_questioned",
                    label_visibility="collapsed",
                )
                st.caption(t("supported_files"))
            else:
                questioned_file = render_single_camera(
                    "manual_questioned",
                    t("take_reference_photo"),
                )

            if questioned_file is not None:
                st.success(t("questioned_upload_ready"))
            else:
                st.caption(t("waiting_for_questioned"))

    st.caption(f"💡 {t('capture_tip')}")

    if reference_files and questioned_file is not None:
        invalid_files = [ # Files PIL cannot recognize as valid images
            uploaded_file
            for uploaded_file in [*reference_files, questioned_file]
            if not is_valid_uploaded_image(uploaded_file)
        ]

        if invalid_files:
            for uploaded_file in invalid_files:
                st.error(t("invalid_image_file").format(name=uploaded_file.name))

            return

        col1, col2 = st.columns(2)

        with col1:
            st.caption(t("references_selected").format(count=len(reference_files)))
            st.caption(t("processing_mode_help"))
            reference_usable = True # Remains True only when every reference passes image-quality checks
            reference_modes = [] # Processing mode selected separately for every reference

            for index, reference_file in enumerate(reference_files, start = 1):
                with st.expander(f"Reference {index}: {reference_file.name}", expanded = index == 1):
                    st.image(reference_file, use_container_width = True)

                    this_mode = st.radio(
                        f"Processing — Reference {index}",
                        ["default", "sharp"],
                        index = 0, # References use Standard by default and Sharp can be selected when needed
                        format_func = mode_label,
                        key = f"manual_ref_mode_{index}",
                        horizontal = True
                    )

                    reference_modes.append(this_mode)

                    reference_usable = (
                        render_model_preview(reference_file, f"Reference {index}", this_mode)
                        and reference_usable
                    ) # Logical and keeps the complete reference group unusable when any one image fails

        with col2:
            st.image(
                questioned_file,
                caption=t("questioned_signature"),
                use_container_width=True
            )
            questioned_mode = st.radio(
                "Processing — Questioned",
                ["default", "sharp"],
                index = 0, # Questioned signature uses Standard by default
                format_func = mode_label,
                key = "manual_questioned_mode_std",
                horizontal = True
            )

            questioned_usable = render_model_preview(questioned_file, "Questioned", questioned_mode)

        st.divider()

        with st.container(key="verification_action"):
            st.markdown(f"""
            <div class="verification-ready">
                <span class="ready-check">✓</span>
                <span>{t("ready_for_verification")}</span>
            </div>
            """, unsafe_allow_html = True)
            verify_clicked = st.button(
                t("verify_button"),
                use_container_width=True,
                disabled=not (reference_usable and questioned_usable),
            )

        if verify_clicked:
            reference_paths = [ # Temporary filesystem paths required by OpenCV and model loaders
                save_uploaded_file(reference_file)
                for reference_file in reference_files
            ]

            questioned_path = save_uploaded_file(questioned_file) # Temporary path of the signature being verified

            effective_engine, engine_reason = resolve_engine_details(
                engine, [*reference_paths, questioned_path]
            ) # effective_engine = Final manual or automatically detected verification engine

            if not ensure_engine_ready(effective_engine):
                for temporary_path in [*reference_paths, questioned_path]:
                    Path(temporary_path).unlink(missing_ok=True)
                return

            st.caption(
                "Selected engine: "
                + {"archive": "Low-resolution scan v2", "utsig": "High-resolution mono v3", "general": "General v1"}[effective_engine]
                + f" — {engine_reason}"
            )

            comparisons = [] # One distance and optional genuine probability for every reference

            for index, reference_path in enumerate(reference_paths, start = 1):
                if effective_engine == "archive":
                    probability = archive_pair_probability(reference_path, questioned_path) # Probability = Model confidence that both signatures belong to the same person
                    distance = 1 - probability # Converts similarity probability into distance ------> smaller still means more similar

                elif effective_engine == "utsig":
                    probability = utsig_pair_probability(reference_path, questioned_path)
                    distance = 1 - probability

                else:
                    _, distance = verify_signature(
                        reference_path,
                        questioned_path,
                        threshold = THRESHOLD,
                        preprocess_mode = reference_modes[index - 1], # index - 1 maps one-based reference numbering back to the zero-based mode list
                        preprocess_mode_2 = questioned_mode
                    )

                    probability = None # General v1 returns embedding distance directly instead of a probability

                comparisons.append({
                    "reference": reference_files[index - 1].name,
                    "distance": distance,
                    "probability": probability,
                })

            best_match = min(
                comparisons,
                key = lambda item: item["distance"]
            ) # min selects the comparison with the lowest distance ------> closest reference

            if effective_engine == "archive":
                if len(comparisons) == 1:
                    probability = comparisons[0]["probability"] # One reference ------> its probability becomes the complete decision score
                    verdict = archive_verdict(probability) # verdict = Internal match uncertain or no-match decision
                    support_count = int(probability >= HIGH_PROBABILITY) # bool becomes one supporting vote or zero
                    required_support = 1
                else:
                    decision = decide_archive_registered( # decision = Group result calculated from all reference probabilities
                        [item["probability"] for item in comparisons]
                    )

                    probability = decision["probability"]
                    verdict = decision["verdict"]
                    support_count = decision["support_count"]
                    required_support = decision["required_support"]

                confidence = archive_confidence(probability)
                distance = 1 - probability
                display_threshold = 1 - HIGH_PROBABILITY # Converts the probability boundary into the distance scale shown in the UI

            elif effective_engine == "utsig":
                if len(comparisons) == 1:
                    probability = comparisons[0]["probability"]
                    verdict = utsig_classify(probability)
                    support_count = int(probability >= UTSIG_MANUAL_SAME_MIN)
                    required_support = 1
                else:
                    decision = decide_utsig_registered(
                        [item["probability"] for item in comparisons]
                    )

                    probability = decision["probability"]
                    verdict = decision["verdict"]
                    support_count = decision["support_count"]
                    required_support = decision["required_support"]

                confidence = utsig_confidence(
                    probability,
                    registered = len(comparisons) > 1 # Registered confidence boundaries are used when several references vote together
                )

                distance = 1 - probability
                display_threshold = 1 - (
                    UTSIG_REGISTERED_SAME_MIN if len(comparisons) > 1
                    else UTSIG_MANUAL_SAME_MIN
                )

            else:
                if len(comparisons) == 1:
                    distance = comparisons[0]["distance"]
                    verdict = classify_verdict(distance)
                    support_count = int(distance < THRESHOLD)
                    required_support = 1
                else:
                    decision = decide_registered_person( # General v1 combines distances instead of probabilities
                        [item["distance"] for item in comparisons],
                        threshold = THRESHOLD,
                        confident_diff = T_DIFF,
                        confident_same = T_SAME
                    )

                    distance = decision["decision_distance"]
                    verdict = decision["verdict"]
                    support_count = decision["support_count"]
                    required_support = decision["required_support"]

                confidence = calculate_confidence(distance)
                display_threshold = THRESHOLD

            result_str = {"match": "Genuine", "uncertain": "Uncertain", "no_match": "Forgery"}[verdict] # Internal verdict maps to the report result label

            st.divider()

            show_summary_card(
                person_name = "Manual Compare",
                result = result_str,
                confidence = confidence,
                distance = distance,
                threshold = display_threshold,
                best_reference = best_match["reference"],
                compared_count = len(comparisons)
            )

            with st.expander(t("technical_details")):
                st.write(f"Distance: `{distance:.4f}`")
                if effective_engine in ("archive", "utsig"):
                    st.write(f"Genuine Probability: `{probability:.4f}`")
                st.write(f"Matching References: `{support_count}/{len(comparisons)}`")
                st.write(f"Required Support: `{required_support}`")
                st.write(f"Best Reference: `{best_match['reference']}`")
                st.write(f"Verdict: `{result_str}`")
                comparison_table = [ # Table rows shown inside the technical details expander
                    {
                        "Reference": item["reference"],
                        "Distance": round(item["distance"], 4),
                        "Genuine Probability": (
                            round(item["probability"], 4)
                            if item["probability"] is not None else None
                        )
                    }
                    for item in comparisons
                ]

                st.dataframe(comparison_table, use_container_width = True)

            save_verification_history(
                person_name="Manual Compare",
                result=result_str,
                distance=distance,
                threshold=display_threshold,
                engine=effective_engine,
            )

            pdf = create_verification_report(
                person_name="Manual Compare",
                result=result_str,
                confidence = confidence,
                distance=distance,
                threshold=display_threshold,
                best_reference=best_match["reference"],
                compared_count=len(comparisons),
                language=current_language(),
                engine=effective_engine,
            )

            st.download_button(
                t("download_report"),
                data=pdf,
                file_name="verification_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    else:
        st.markdown(f"""
        <div class="upload-waiting-bar">
            <span class="waiting-lock"></span>
            <span>{t("upload_both_to_continue")}</span>
            <span class="waiting-arrow">→</span>
        </div>
    """, unsafe_allow_html = True)


def render_registered_person_verify(engine = "general"):
    st.subheader(t("registered_verify"))

    st.session_state.people = load_registered_people() # session_state keeps the registered-person list across Streamlit reruns

    if len(st.session_state.people) == 0:
        st.warning("No registered people found. Please add a person from the Admin Panel first.")
        return

    st.info(
        "Recommendation: Save at least five genuine references from different days and pens for the most reliable result. "
        "Verification can still run with fewer references."
    )

    person_names = [person["name"] for person in st.session_state.people] # Display names used to validate the current selection

    if st.session_state.get("selected_registered_person") not in person_names:
        st.session_state.selected_registered_person = person_names[0] # Falls back to the first person when the previous selection no longer exists

    st.markdown(f"### 1 · {t('select_person')}")
    person_search = st.text_input(
        t("search_person"),
        placeholder=t("search_person_placeholder"),
        label_visibility="collapsed",
        key="registered_person_search",
    )
    normalized_search = person_search.strip().casefold() # casefold creates a strong lowercase form for case-insensitive search

    filtered_people = [ # Registered people whose normalized name contains the search text
        person for person in st.session_state.people
        if normalized_search in person["name"].casefold()
    ]

    total_pages = max(1, math.ceil(len(filtered_people) / PEOPLE_PER_PAGE)) # ceil keeps a partially filled final page
    current_page = min(
        max(1, st.session_state.get("registered_people_page", 1)),
        total_pages
    ) # Clamp keeps the saved page number inside the available range

    st.session_state.registered_people_page = current_page

    page_start = (current_page - 1) * PEOPLE_PER_PAGE # Converts the one-based page number into a zero-based list index
    visible_people = filtered_people[page_start:page_start + PEOPLE_PER_PAGE] # Slice containing only people displayed on this page

    with st.container(key="person_selector"):
        if not filtered_people:
            st.info(t("no_person_search_results"))

        for row_start in range(0, len(visible_people), 2):
            person_columns = st.columns(2) # Two profile cards are placed in every row

            for column, person in zip(
                person_columns, visible_people[row_start:row_start + 2]
            ):
                with column:
                    is_selected = (
                        person["name"] == st.session_state.selected_registered_person
                    ) # is_selected controls button style and the active person

                    with st.container(border=True):
                        photo_col, info_col = st.columns([1, 3])
                        with photo_col:
                            st.image(
                                person.get("profile_path") or str(DEFAULT_AVATAR_PATH),
                                width=72,
                            )
                        with info_col:
                            st.markdown(f"**{person['name']}**")
                            st.caption(
                                t("person_signature_count").format(
                                    count=person["signature_count"]
                                )
                            )
                        if st.button(
                            t("selected_person_button") if is_selected else t("select_person_button"),
                            key=f"select_person_{person['folder_name']}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state.selected_registered_person = person["name"]
                            st.rerun() # Rebuilds the page immediately so the new selection is reflected everywhere

        if filtered_people and total_pages > 1:
            previous_col, page_col, next_col = st.columns([1, 2, 1])
            with previous_col:
                if st.button(
                    t("previous_page"),
                    disabled=current_page == 1,
                    use_container_width=True,
                    key="registered_people_previous",
                ):
                    st.session_state.registered_people_page = current_page - 1
                    st.rerun()
            with page_col:
                st.caption(
                    t("page_status").format(page=current_page, total=total_pages)
                )
            with next_col:
                if st.button(
                    t("next_page"),
                    disabled=current_page == total_pages,
                    use_container_width=True,
                    key="registered_people_next",
                ):
                    st.session_state.registered_people_page = current_page + 1
                    st.rerun()

    selected_person_name = st.session_state.selected_registered_person

    st.caption(t("selected_person_caption").format(name=selected_person_name))

    st.markdown(f"### 2 · {t('questioned_signature')}")
    reference_mode = st.radio(
        "Registered Reference Processing Mode",
        ["default", "sharp"],
        index = 0, # Registered references use Standard by default
        format_func = mode_label,
        key = "registered_ref_mode",
        horizontal = True
    )

    input_method = st.radio(
        "Questioned Signature Input",
        ["Upload File", "Take Photo"],
        horizontal = True
    )

    if input_method == "Upload File":
        questioned_file = st.file_uploader(
            "Upload Questioned Signature",
            type=["png", "jpg", "jpeg"],
            key="registered_questioned_upload"
        )

    else:
        questioned_file = render_single_camera(
            "registered_questioned",
            "Take Questioned Signature Photo",
        )

    if questioned_file is not None:
        if not is_valid_uploaded_image(questioned_file):
            st.error(t("invalid_image_file").format(name=questioned_file.name))
            return

        st.image(
            questioned_file,
            caption="Questioned Signature",
            use_container_width=True
        )
        questioned_mode = st.radio(
            "Processing — Questioned",
            ["default", "sharp"],
            index = 0, # Questioned signature uses Standard by default
            format_func = mode_label,
            key = "registered_questioned_mode_std",
            horizontal = True
        )

        questioned_usable = render_model_preview(questioned_file, "Questioned", questioned_mode)

        if st.button(
            "Verify Against Registered Person",
            use_container_width=True,
            disabled=not questioned_usable,
        ):
            selected_person = None # Full database record matching the selected display name

            for person in st.session_state.people:
                if person["name"] == selected_person_name:
                    selected_person = person
                    break

            person_dir = DATABASE_DIR / selected_person["folder_name"] # Folder containing this person's raw registered signatures
            reference_paths = get_reference_paths(person_dir) # Stable list of reference image paths without the profile photo

            if len(reference_paths) == 0:
                st.error("No reference signatures found for this person.")
                return

            if len(reference_paths) < 5:
                st.warning(
                    f"This person has only {len(reference_paths)} references. "
                    "Verification can continue with the median decision but five or more different genuine samples are recommended."
                )

            questioned_path = save_uploaded_file(questioned_file) # Temporary path required by OpenCV and model loaders

            effective_engine, engine_reason = resolve_engine_details(
                engine, [*reference_paths, questioned_path]
            )

            if not ensure_engine_ready(effective_engine):
                Path(questioned_path).unlink(missing_ok=True)
                return

            st.caption(
                "Selected engine: "
                + {"archive": "Low-resolution scan v2", "utsig": "High-resolution mono v3", "general": "General v1"}[effective_engine]
                + f" — {engine_reason}"
            )

            distances = [] # Comparison details for every registered reference

            for reference_path in reference_paths:
                if effective_engine == "archive":
                    probability = archive_pair_probability(
                        str(reference_path),
                        questioned_path
                    ) # Probability = Estimated chance that this reference and questioned signature match

                    distance = 1 - probability

                elif effective_engine == "utsig":
                    probability = utsig_pair_probability(
                        str(reference_path),
                        questioned_path
                    )

                    distance = 1 - probability

                else:
                    _, distance = verify_signature(
                        str(reference_path),
                        questioned_path,
                        threshold = THRESHOLD,
                        preprocess_mode = reference_mode,
                        preprocess_mode_2 = questioned_mode
                    )

                    probability = None # General v1 works directly with embedding distance

                distances.append(
                    {
                        "reference": reference_path.name,
                        "distance": distance,
                        "probability": probability,
                    }
                )

            best_match = min(distances, key = lambda item: item["distance"]) # Reference with the smallest distance
            best_distance = best_match["distance"]

            average_distance = sum(item["distance"] for item in distances) / len(distances) # Mean distance = Sum of reference distances / Reference count

            if effective_engine == "archive":
                enrollment_decision = decide_archive_registered( # enrollment_decision = Group verdict support counts and stable group probability
                    [item["probability"] for item in distances]
                )

                decision_probability = enrollment_decision["probability"]
                decision_distance = 1 - decision_probability # Converts similarity probability into the distance displayed by the shared UI
                display_threshold = 1 - HIGH_PROBABILITY
                confidence = archive_confidence(decision_probability)

            elif effective_engine == "utsig":
                enrollment_decision = decide_utsig_registered(
                    [item["probability"] for item in distances]
                )

                decision_probability = enrollment_decision["probability"]
                decision_distance = 1 - decision_probability
                display_threshold = 1 - UTSIG_REGISTERED_SAME_MIN
                confidence = utsig_confidence(
                    decision_probability,
                    registered = True
                )

            else:
                enrollment_decision = decide_registered_person(
                    [item["distance"] for item in distances],
                    threshold = THRESHOLD,
                    confident_diff = T_DIFF,
                    confident_same = T_SAME
                )

                decision_distance = enrollment_decision["decision_distance"]
                display_threshold = THRESHOLD
                confidence = calculate_confidence(decision_distance)

            genuine_count = enrollment_decision["support_count"] # References whose score supports a genuine decision

            verdict = enrollment_decision["verdict"] # Group decision ------> match uncertain or no match
            result_str = {"match": "Genuine", "uncertain": "Uncertain", "no_match": "Forgery"}[verdict]

            st.divider()

            show_summary_card(
                person_name = selected_person_name,
                result = result_str,
                confidence = confidence,
                distance = decision_distance,
                threshold = display_threshold,
                best_reference = best_match["reference"],
                compared_count = len(reference_paths)
            )

            with st.expander(t("technical_details")):
                st.write(f"Selected Person: `{selected_person_name}`")
                if effective_engine == "archive":
                    method_label = "Archive v2 60% probability majority + median"
                elif effective_engine == "utsig":
                    method_label = "High-resolution mono v3 60% majority + median"
                else:
                    method_label = (
                        "60% majority across all references + median"
                        if enrollment_decision["method"] == "majority_median"
                        else "Median with fewer than five references"
                    )
                st.write(f"Decision Method: `{method_label}`")
                st.write(f"Decision Distance: `{decision_distance:.4f}`")
                if effective_engine in ("archive", "utsig"):
                    st.write(f"Genuine Probability: `{decision_probability:.4f}`")
                st.write(f"Best Reference: `{best_match['reference']}`")
                st.write(f"Best Distance: `{best_distance:.4f}`")
                st.write(f"Average Distance: `{average_distance:.4f}`")
                st.write(f"Genuine Reference Count: `{genuine_count}/{len(reference_paths)}`")
                st.write(
                    "Required Matching References: "
                    f"`{enrollment_decision['required_support']}`"
                )
                st.write(f"Threshold: `{display_threshold}`")
                st.write(f"Compared Reference Signatures: `{len(reference_paths)}`")

                st.markdown("#### Reference Distance Table")

                distance_table = [] # Rows used by both the technical table and Plotly chart

                for item in distances:
                    distance_table.append(
                        {
                            "Reference Signature": item["reference"],
                            "Distance": round(item["distance"], 4),
                            "Result": "Genuine" if item["distance"] < display_threshold else "Forgery"
                        }
                    )

                st.success(
                    f"Best match: {best_match['reference']} "
                    f"with distance {best_distance:.4f}"
                )

                st.dataframe(distance_table, use_container_width=True)

                distance_df = pd.DataFrame(distance_table) # DataFrame = Table-like structure required by Plotly Express

                fig = px.bar(
                    distance_df,
                    x = "Reference Signature", # Category axis ------> one bar for every registered reference
                    y = "Distance", # Value axis ------> shorter bars represent closer signatures
                    color = "Result", # Bar color represents Genuine or Forgery classification
                    text = "Distance",
                    title = "Reference Signature Distance Comparison",
                    color_discrete_map = {
                        "Genuine": "#22c55e",
                        "Forgery": "#ef4444"
                    }
                )

                fig.update_traces(
                    texttemplate = "%{text:.4f}", # Shows four decimal digits above every bar
                    textposition = "outside"
                )

                max_distance = max(distance_df["Distance"].max(), display_threshold) # Keeps both bars and threshold line inside the chart range

                fig.update_layout(
                    xaxis_title = "Reference Signature",
                    yaxis_title = "Distance",
                    yaxis = dict(range = [0, max_distance + 0.05]),
                    legend_title = "Classification"
                )

                fig.add_hline(
                    y = display_threshold, # Horizontal boundary separating closer and farther reference distances
                    line_dash = "dash",
                    annotation_text = f"Threshold = {display_threshold:.2f}",
                    annotation_position = "top left"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            save_verification_history(
                person_name=selected_person_name,
                result=result_str,
                distance=decision_distance,
                threshold=display_threshold,
                person_id=selected_person["id"],
                engine=effective_engine,
            )

            pdf = create_verification_report(
                person_name=selected_person_name,
                result=result_str,
                confidence = confidence,
                distance=decision_distance,
                threshold=display_threshold,
                best_reference=best_match["reference"],
                compared_count=len(reference_paths),
                language=current_language(),
                engine=effective_engine,
            )

            st.download_button(
                t("download_report"),
                data=pdf,
                file_name=f"{selected_person_name}_verification_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
