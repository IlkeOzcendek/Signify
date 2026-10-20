"""Camera and crop UI helpers used before enrollment storage.

This module intentionally does not call the model or preprocessing pipeline.
It only lets the user inspect and crop the original image before it is saved.
"""

from io import BytesIO
from typing import Dict, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

from utils.i18n import current_language


class PreparedImage(BytesIO):
    """In memory PNG with the UploadedFile interface used by storage code"""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name

    def getbuffer(self):
        return super().getbuffer()


def _tr(tr_text: str, en_text: str) -> str:
    return tr_text if current_language() == "tr" else en_text


def open_upload(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    uploaded_file.seek(0)
    return image


def encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def encode_camera_crop(image: Image.Image) -> bytes:
    """Encode an accepted camera crop at a compact, upload-like resolution.

    Phone cameras produce multi-megapixel files while successful desktop crops
    are typically much smaller.  Keeping the crop inside 600x220 preserves its
    aspect ratio and makes camera captures follow the same downstream routing as
    those compact uploads.  This is intentionally limited to camera captures;
    uploaded files and model preprocessing are left untouched.
    """

    compact = image.convert("RGB").copy()
    compact.thumbnail((600, 220), Image.Resampling.LANCZOS)
    return encode_png(compact)


def image_capture_feedback(image: Image.Image) -> Dict[str, object]:
    """Measure capture conditions without altering the image"""

    sample = image.copy()
    sample.thumbnail((640, 640))
    luminance = np.asarray(sample.convert("L"), dtype=np.float32)
    brightness = float(luminance.mean())
    contrast = float(luminance.std())
    width, height = image.size

    if brightness < 70:
        level = "dark"
        message = _tr(
            "Işık düşük. Daha aydınlık bir yere geç veya kâğıdı gölgelememeye çalış",
            "Low light. Move somewhere brighter and avoid casting a shadow on the paper",
        )
    elif brightness > 247:
        level = "bright"
        message = _tr(
            "Görüntü fazla parlak. Flaş yansımasını azalt ve kamerayı biraz eğ",
            "The image is overexposed. Reduce flash glare or tilt the camera slightly",
        )
    elif contrast < 18:
        level = "contrast"
        message = _tr(
            "İmza ile kâğıt arasındaki kontrast düşük. Netliği ve ışığı kontrol et",
            "Low contrast between ink and paper. Check focus and lighting",
        )
    else:
        level = "good"
        message = _tr(
            "Işık ve kontrast iyi görünüyor",
            "Lighting and contrast look good",
        )

    return {
        "level": level,
        "message": message,
        "brightness": brightness,
        "contrast": contrast,
        "width": width,
        "height": height,
    }


def show_capture_feedback(image: Image.Image) -> None:
    feedback = image_capture_feedback(image)
    if feedback["level"] == "good":
        st.success(feedback["message"], icon="💡")
    else:
        st.warning(feedback["message"], icon="💡")

    st.caption(
        _tr(
            f"Çözünürlük: {feedback['width']}×{feedback['height']} · "
            f"Işık: {feedback['brightness']:.0f}/255 · Kontrast: {feedback['contrast']:.0f}",
            f"Resolution: {feedback['width']}×{feedback['height']} · "
            f"Light: {feedback['brightness']:.0f}/255 · Contrast: {feedback['contrast']:.0f}",
        )
    )


def crop_editor(image: Image.Image, key: str) -> Image.Image:
    """Render a touch-friendly free crop box and return the visible crop."""

    rotation = st.radio(
        _tr("Döndür", "Rotate"),
        [0, 90, 180, 270],
        horizontal=True,
        key=f"{key}_rotation",
        format_func=lambda angle: "0°" if angle == 0 else f"{angle}°",
    )
    working_image = image.rotate(rotation, expand=True) if rotation else image
    st.caption(
        _tr(
            "Köşeleri sürükleyerek yalnızca imza ve çevresinde az miktarda boşluk bırak.",
            "Drag the corners to keep the signature with a small amount of space around it.",
        )
    )
    return st_cropper(
        working_image,
        realtime_update=True,
        box_color="#6c63ff",
        aspect_ratio=None,
        return_type="image",
        key=f"{key}_cropper",
    )


def prepared_image(image: Image.Image, name: str) -> PreparedImage:
    output_name = f"{name.rsplit('.', 1)[0]}_cropped.png"
    return PreparedImage(encode_png(image), output_name)


def render_single_camera(namespace: str, label: Optional[str] = None):
    """Capture, review, confirm, and retain one verification image."""

    open_key = f"{namespace}_camera_open"
    generation_key = f"{namespace}_camera_generation"
    pending_key = f"{namespace}_pending_capture"
    accepted_key = f"{namespace}_accepted_capture"
    st.session_state.setdefault(open_key, True)
    st.session_state.setdefault(generation_key, 0)

    accepted_data = st.session_state.get(accepted_key)
    if accepted_data is not None:
        st.success(_tr("Test fotoğrafı hazır", "Test photo is ready"), icon="✅")
        st.image(
            accepted_data,
            caption=_tr("Seçilen test imzası", "Selected questioned signature"),
            use_container_width=True,
        )
        change_col, delete_col = st.columns(2)
        with change_col:
            if st.button(
                _tr("Yeniden çek", "Retake"),
                key=f"{namespace}_retake_accepted",
                use_container_width=True,
            ):
                st.session_state.pop(accepted_key, None)
                st.session_state[open_key] = True
                st.session_state[generation_key] += 1
                st.rerun()
        with delete_col:
            if st.button(
                _tr("Seçileni sil", "Delete selected"),
                key=f"{namespace}_delete_accepted",
                use_container_width=True,
            ):
                st.session_state.pop(accepted_key, None)
                st.session_state[open_key] = False
                st.rerun()
        return PreparedImage(accepted_data, "questioned_signature.png")

    pending_data = st.session_state.get(pending_key)
    if pending_data is not None:
        pending_file = PreparedImage(pending_data, "camera_signature.jpg")
        image = open_upload(pending_file)
        st.info(_tr("Fotoğrafı kontrol et ve onayla", "Review and confirm the photo"), icon="👀")
        show_capture_feedback(image)
        cropped = crop_editor(
            image,
            f"{namespace}_review_{st.session_state[generation_key]}",
        )
        st.image(
            cropped,
            caption=_tr("Onaylanacak kırpım", "Crop to confirm"),
            use_container_width=True,
        )
        accept_col, retake_col = st.columns(2)
        with accept_col:
            if st.button(
                _tr("Bu fotoğrafı kullan", "Use this photo"),
                type="primary",
                key=f"{namespace}_accept_pending",
                use_container_width=True,
            ):
                st.session_state[accepted_key] = encode_camera_crop(cropped)
                st.session_state.pop(pending_key, None)
                st.session_state[open_key] = False
                st.rerun()
        with retake_col:
            if st.button(
                _tr("Sil ve yeniden çek", "Delete and retake"),
                key=f"{namespace}_retake_pending",
                use_container_width=True,
            ):
                st.session_state.pop(pending_key, None)
                st.session_state[open_key] = True
                st.session_state[generation_key] += 1
                st.rerun()
        return None

    if not st.session_state[open_key]:
        if st.button(
            _tr("Kamerayı aç", "Open camera"),
            key=f"{namespace}_open_camera",
            use_container_width=True,
        ):
            st.session_state[open_key] = True
            st.rerun()
        return None

    if st.button(
        _tr("Kamerayı kapat", "Close camera"),
        key=f"{namespace}_close_camera",
        use_container_width=True,
    ):
        st.session_state[open_key] = False
        st.rerun()

    st.info(
        _tr(
            "Fotoğrafı çekince kamera otomatik kapanır ve onay ekranı açılır.",
            "After capture, the camera closes automatically and opens the review screen.",
        ),
        icon="📷",
    )
    from utils.mobile_camera import mobile_camera

    st.markdown(f"**{label or _tr('Fotoğraf çek', 'Take a photo')}**")
    camera_file = mobile_camera(
        key=f"{namespace}_camera_{st.session_state[generation_key]}"
    )
    if camera_file is not None:
        st.session_state[pending_key] = bytes(camera_file.getbuffer())
        st.session_state[open_key] = False
        st.session_state[generation_key] += 1
        st.rerun()
    return None


def render_camera_collection(namespace: str) -> list:
    """Capture, review, crop, and stage multiple photos on phone or desktop."""

    photos_key = f"{namespace}_photos"
    counter_key = f"{namespace}_camera_counter"
    open_key = f"{namespace}_camera_open"
    st.session_state.setdefault(photos_key, [])
    st.session_state.setdefault(counter_key, 0)
    st.session_state.setdefault(open_key, True)

    st.info(
        _tr(
            "İmzayı düz bir zemine koy, kamerayı kâğıda paralel tut ve gölge düşürme.",
            "Place the signature on a flat surface, keep the camera parallel, and avoid shadows.",
        ),
        icon="📷",
    )
    from utils.mobile_camera import mobile_camera

    camera_file = None
    if st.session_state[open_key]:
        if st.button(
            _tr("Kamerayı kapat", "Close camera"),
            key=f"{namespace}_close_camera",
            use_container_width=True,
        ):
            st.session_state[open_key] = False
            st.rerun()
        camera_file = mobile_camera(
            key=f"{namespace}_camera_{st.session_state[counter_key]}"
        )
    elif st.button(
        _tr("Kamerayı aç", "Open camera"),
        key=f"{namespace}_open_camera",
        use_container_width=True,
    ):
        st.session_state[open_key] = True
        st.rerun()

    if camera_file is not None:
        image = open_upload(camera_file)
        show_capture_feedback(image)
        cropped = crop_editor(image, f"{namespace}_new_{st.session_state[counter_key]}")
        st.image(cropped, caption=_tr("Kaydedilecek kırpım", "Crop to be saved"))

        if st.button(
            _tr("Bu fotoğrafı kullan", "Use this photo"),
            type="primary",
            use_container_width=True,
            key=f"{namespace}_accept_{st.session_state[counter_key]}",
        ):
            st.session_state[photos_key].append(encode_camera_crop(cropped))
            st.session_state[counter_key] += 1
            st.rerun()

    photos = st.session_state[photos_key]
    if photos:
        st.success(
            _tr(
                f"{len(photos)} fotoğraf hazır. Yeni çekim yapmak için kamerayı tekrar açabilirsin.",
                f"{len(photos)} photo(s) ready. Open the camera again to add another.",
            )
        )
        columns = st.columns(min(4, len(photos)))
        for index, data in enumerate(photos):
            with columns[index % len(columns)]:
                st.image(data, caption=f"#{index + 1}", use_container_width=True)
                if st.button("✕", key=f"{namespace}_remove_{index}"):
                    photos.pop(index)
                    st.rerun()

    return [
        PreparedImage(data, f"camera_signature_{index + 1}.png")
        for index, data in enumerate(photos)
    ]


def clear_camera_collection(namespace: str) -> None:
    st.session_state.pop(f"{namespace}_photos", None)
    st.session_state.pop(f"{namespace}_camera_counter", None)
    st.session_state.pop(f"{namespace}_camera_open", None)
