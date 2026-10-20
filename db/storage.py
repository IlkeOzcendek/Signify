import shutil # For file and folder operations like copy, move, ...
from pathlib import Path
from typing import Optional # It is used for type hinting
from uuid import UUID, uuid4

from db.session import PROJECT_ROOT, STORAGE_DIR

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# All registered person files are kept under this directory
PEOPLE_STORAGE_DIR = STORAGE_DIR / "people"
PEOPLE_STORAGE_DIR.mkdir(parents = True, exist_ok= True)


def normalize_person_id(person_id: str) -> str:
    """Validate and return a canonical UUID string"""

    try:
        return str(UUID(person_id))

    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid person ID")


def get_person_storage_dir(person_id: str) -> Path:
    """Return the safe storage directory belonging to one person"""

    safe_person_id = normalize_person_id(person_id)

    return PEOPLE_STORAGE_DIR / safe_person_id


def get_upload_extension(uploaded_file) -> str:
    """Return a supported lowercase image extension"""

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image file type")

    return extension


def to_relative_path(file_path: Path) -> str:
    """Convert an absolute project file path into a portable relative path"""

    return file_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_stored_path(relative_path: str) -> Path:
    """Resolve a stored path and ensure it stays inside database/storage"""

    resolved_path = (PROJECT_ROOT / relative_path).resolve()
    resolved_storage = STORAGE_DIR.resolve()

    if (
        resolved_path != resolved_storage
        and resolved_storage not in resolved_path.parents
    ):
        raise ValueError("Invalid storage path")

    return resolved_path


def save_reference_file(person_id: str, uploaded_file) -> str:
    """Save one raw reference signature and return its relative path"""

    extension = get_upload_extension(uploaded_file)

    signature_dir = (
        get_person_storage_dir(person_id)
        / "signatures"
    )
    signature_dir.mkdir(parents = True, exist_ok = True)

    destination = signature_dir / f"{uuid4()}{extension}"
    destination.write_bytes(uploaded_file.getbuffer())

    return to_relative_path(destination)


def save_profile_file(
    person_id: str,
    uploaded_file,
) -> Optional[str]:
    """Save a profile image and return its relative path"""

    if uploaded_file is None:
        return None

    extension = get_upload_extension(uploaded_file)

    profile_dir = (
        get_person_storage_dir(person_id)
        / "profile"
    )
    profile_dir.mkdir(parents = True, exist_ok = True)

    destination = profile_dir / f"profile_{uuid4()}{extension}"
    destination.write_bytes(uploaded_file.getbuffer())

    # The new image is written first. Only then are older profiles removed
    for existing_path in profile_dir.iterdir():
        if existing_path != destination and existing_path.is_file():
            existing_path.unlink()

    return to_relative_path(destination)


def delete_stored_file(relative_path: Optional[str]) -> bool:
    """Safely delete one stored file."""

    if not relative_path:
        return False

    file_path = resolve_stored_path(relative_path)

    if not file_path.is_file():
        return False

    file_path.unlink()

    return True


def delete_person_storage(person_id: str) -> bool:
    """Delete one person's complete storage directory"""

    person_dir = get_person_storage_dir(person_id)

    if not person_dir.exists():
        return False

    shutil.rmtree(person_dir)

    return True
