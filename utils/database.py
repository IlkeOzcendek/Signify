"""Application-facing database and managed-file-storage operations.

This module keeps the small API used by the Streamlit pages while routing all
persistent metadata through SQLAlchemy.  Images live in managed storage and the
database stores only portable, project-relative paths.
"""

import tempfile
from pathlib import Path

from db.models import Base
from db.repository import (
    add_signatures,
    create_person,
    delete_person,
    delete_signature,
    get_person,
    list_people,
    list_signatures,
    rename_person,
    update_profile_path,
)
from db.session import engine
from db.storage import (
    PEOPLE_STORAGE_DIR,
    delete_person_storage,
    delete_stored_file,
    resolve_stored_path,
    save_profile_file,
    save_reference_file,
)


# Existing pages build ``DATABASE_DIR / person_id``.  Keep that expression valid
# while making the UUID-backed managed storage directory its actual destination.
DATABASE_DIR = PEOPLE_STORAGE_DIR


def initialize_database() -> None:
    """Create missing tables for a fresh local installation.

    Alembic remains the source of schema migrations; create_all only makes the
    application robust when it is launched before the first explicit upgrade.
    """

    Base.metadata.create_all(engine)


initialize_database()


def _application_person(person):
    """Add compatibility keys expected by the current Streamlit pages."""

    profile_path = person.get("profile_path")
    return {
        **person,
        "folder_name": person["id"],
        "profile_path": (
            str(resolve_stored_path(profile_path)) if profile_path else None
        ),
    }


def load_registered_people():
    return [_application_person(person) for person in list_people()]


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.getbuffer())
        return temp.name


def save_reference_signatures(person_name, reference_signatures):
    """Create a person and atomically register their uploaded references."""

    person = create_person(person_name)
    saved_paths = []
    try:
        rows = []
        for upload in reference_signatures:
            relative_path = save_reference_file(person["id"], upload)
            saved_paths.append(relative_path)
            rows.append((relative_path, upload.name))
        add_signatures(person["id"], rows)
    except Exception:
        delete_person(person["id"])
        delete_person_storage(person["id"])
        raise
    return person["id"]


def save_profile_image(person_id, profile_image):
    if profile_image is None:
        return None

    person = get_person(person_id)
    if person is None:
        raise ValueError("Person not found.")

    previous_path = person.get("profile_path")
    new_path = save_profile_file(person_id, profile_image)
    try:
        update_profile_path(person_id, new_path)
    except Exception:
        delete_stored_file(new_path)
        raise

    if previous_path and previous_path != new_path:
        delete_stored_file(previous_path)
    return resolve_stored_path(new_path)


def append_reference_signatures(person_id, reference_signatures):
    if get_person(person_id) is None:
        raise ValueError("Person not found.")

    saved_paths = []
    try:
        rows = []
        for upload in reference_signatures:
            relative_path = save_reference_file(person_id, upload)
            saved_paths.append(relative_path)
            rows.append((relative_path, upload.name))
        records = add_signatures(person_id, rows)
    except Exception:
        for relative_path in saved_paths:
            delete_stored_file(relative_path)
        raise
    return [resolve_stored_path(record["file_path"]) for record in records]


def get_reference_paths(person_dir):
    """Return existing managed reference paths for a UUID person directory."""

    person_id = Path(person_dir).name
    paths = []
    for signature in list_signatures(person_id):
        path = resolve_stored_path(signature["file_path"])
        if path.is_file():
            paths.append(path)
    return paths


def delete_reference_signature(person_id, file_name):
    safe_name = Path(file_name).name
    for signature in list_signatures(person_id):
        if Path(signature["file_path"]).name == safe_name:
            deleted = delete_signature(signature["id"])
            if deleted:
                delete_stored_file(deleted["file_path"])
            return deleted
    return None


def rename_person_folder(person_id, new_person_name):
    rename_person(person_id, new_person_name)
    return person_id


def delete_person_folder(person_id):
    deleted = delete_person(person_id)
    if deleted is not None:
        delete_person_storage(person_id)
    return deleted
