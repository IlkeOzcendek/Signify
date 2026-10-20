from typing import Dict, List, Optional, Tuple # Python 3.9 compatible type descriptions for returned dictionaries optional values and file pairs

from sqlalchemy import select # Builds SELECT queries with Python objects instead of handwritten SQL
from sqlalchemy.orm import selectinload # Loads related signatures before the session closes

from db.models import Person, Signature, VerificationRecord # SQLAlchemy table models used by repository queries
from db.session import SessionLocal # Factory that creates a short lived database session for each operation

def signature_to_dict(signature: Signature) -> Dict:
    """Convert a signature database object into application friendly data"""

    return { # The application receives plain data and does not need to manage a live SQLAlchemy session
        "id": signature.id,
        "person_id": signature.person_id,
        "file_path": signature.file_path,
        "original_filename": signature.original_filename,
        "created_at": signature.created_at,
    }

def person_to_dict(person: Person) -> Dict:
    """Convert a person and loaded signatures into application friendly data"""

    signatures = [ # Converts every related Signature model before the Person session is closed
        signature_to_dict(signature)
        for signature in person.signatures
    ]

    return { # Keeps the keys expected by the Streamlit application independent from database column names
        "id": person.id,
        "name": person.display_name,
        "profile_path": person.profile_path,
        "signature_count": len(signatures),
        "signatures": signatures,
        "created_at": person.created_at,
    }

def verification_to_dict(record: VerificationRecord) -> Dict:
    """Convert a verification record into application-friendly data"""

    return { # Uses the old history column names such as person and date for easier application integration
        "id": record.id,
        "person_id": record.person_id,
        "person": record.person_name,
        "result": record.result,
        "distance": record.distance,
        "threshold": record.threshold,
        "engine": record.engine,
        "date": record.created_at,
    }

def _person_statement():
    """Build the shared person query with signatures eagerly loaded"""

    return select(Person).options(
        selectinload(Person.signatures)
    ) # Eager loading prevents DetachedInstanceError when person_to_dict reads signatures

def create_person(
    display_name: str,
    profile_path: Optional[str] = None,
) -> Dict:
    """Create and return a registered person"""

    cleaned_name = display_name.strip() # Removes accidental spaces from the beginning and end of a person's name

    if not cleaned_name:
        raise ValueError("Person name cannot be empty") # Stops invalid data before it reaches SQLite

    with SessionLocal() as session: # The context manager closes the session even when an exception occurs
        person = Person( # Creates a Python model object but does not write it to SQLite yet
            display_name=cleaned_name,
            profile_path=profile_path,
        )
        session.add(person) # Places the new object in the current unit of work
        session.commit() # Permanently writes the pending INSERT to SQLite

        saved_person = session.scalar( # Reads one Person object and eagerly loads its signatures
            _person_statement().where(Person.id == person.id)
        )

        return person_to_dict(saved_person)

def get_person(person_id: str) -> Optional[Dict]:
    """Return one person by UUID or None when it does not exist"""

    with SessionLocal() as session:
        person = session.scalar( # scalar returns the first model object instead of a complete SQL row
            _person_statement().where(Person.id == person_id)
        )

        return person_to_dict(person) if person is not None else None # Missing UUIDs have a predictable None result

def list_people() -> List[Dict]:
    """Return all registered people with their signatures"""

    with SessionLocal() as session:
        statement = _person_statement().order_by( # Produces stable alphabetical output for the user interface
            Person.display_name,
            Person.created_at,
        )
        people = session.scalars(statement).all() # scalars extracts Person objects and all collects the complete result

        return [person_to_dict(person) for person in people]

def rename_person(person_id: str, new_display_name: str) -> Dict:
    """Change a person's display name without changing its UUID or files."""

    cleaned_name = new_display_name.strip() # Only the visible name changes; the UUID storage folder remains stable

    if not cleaned_name:
        raise ValueError("Person name cannot be empty.")

    with SessionLocal() as session:
        person = session.get(Person, person_id) # Primary-key lookup is the simplest and fastest way to find one person

        if person is None:
            raise ValueError("Person not found.")

        person.display_name = cleaned_name # SQLAlchemy tracks this field change automatically
        session.commit() # Flushes the generated UPDATE and completes the transaction

        saved_person = session.scalar(
            _person_statement().where(Person.id == person.id)
        )

        return person_to_dict(saved_person)

def update_profile_path(
    person_id: str,
    profile_path: Optional[str],
) -> Dict:
    """Set or clear the stored profile image path for a person."""

    with SessionLocal() as session:
        person = session.get(Person, person_id) # Profile ownership is resolved by the person's UUID

        if person is None:
            raise ValueError("Person not found.")

        person.profile_path = profile_path # None clears the profile path without deleting the Person row
        session.commit()

        saved_person = session.scalar(
            _person_statement().where(Person.id == person.id)
        )

        return person_to_dict(saved_person)

def add_signature(
    person_id: str,
    file_path: str,
    original_filename: Optional[str] = None,
) -> Dict:
    """Add one reference signature record to an existing person"""

    with SessionLocal() as session:
        if session.get(Person, person_id) is None: # Gives a clear error before SQLite reports a foreign-key failure
            raise ValueError("Person not found")

        signature = Signature( # Stores metadata and a path; the image bytes remain in database/storage
            person_id=person_id,
            file_path=file_path,
            original_filename=original_filename,
        )
        session.add(signature)
        session.commit()

        return signature_to_dict(signature)

def add_signatures(
    person_id: str,
    files: List[Tuple[str, Optional[str]]],
) -> List[Dict]:
    """Add several reference signature records in one transaction"""

    with SessionLocal() as session:
        if session.get(Person, person_id) is None: # All signatures in this transaction must belong to a real person
            raise ValueError("Person not found")

        signatures = [ # Creates all model objects before issuing a single transaction commit
            Signature(
                person_id = person_id,
                file_path = file_path,
                original_filename = original_filename,
            )
            for file_path, original_filename in files
        ]

        session.add_all(signatures) # Registers the complete collection as pending INSERT operations
        session.commit() # Either the transaction completes or SQLAlchemy rolls it back after an error

        return [signature_to_dict(signature) for signature in signatures]

def list_signatures(person_id: str) -> List[Dict]:
    """Return a person's reference signatures in creation order"""

    with SessionLocal() as session:
        statement = ( # Limits the query to one owner and keeps reference order deterministic
            select(Signature)
            .where(Signature.person_id == person_id)
            .order_by(Signature.created_at, Signature.id)
        )
        signatures = session.scalars(statement).all()

        return [signature_to_dict(signature) for signature in signatures]

def delete_signature(signature_id: str) -> Optional[Dict]:
    """Delete a signature record and return its data for file cleanup"""

    with SessionLocal() as session:
        signature = session.get(Signature, signature_id) # Signature UUID avoids ambiguity between similar filenames

        if signature is None:
            return None

        deleted_signature = signature_to_dict(signature) # Saves its file path before the database row disappears
        session.delete(signature) # Marks the row for deletion in the current transaction
        session.commit()

        return deleted_signature

def delete_person(person_id: str) -> Optional[Dict]:
    """Delete a person and return its data for storage cleanup."""

    with SessionLocal() as session:
        person = session.scalar( # Signatures are loaded so their paths remain available for later file cleanup
            _person_statement().where(Person.id == person_id)
        )

        if person is None:
            return None

        deleted_person = person_to_dict(person) # Copies person and signature information before committing DELETE
        session.delete(person) # Foreign-key CASCADE removes this person's Signature rows
        session.commit()

        return deleted_person

def create_verification_record(
    person_name: str,
    result: str,
    distance: float,
    threshold: Optional[float],
    person_id: Optional[str] = None,
    engine: Optional[str] = None,
) -> Dict:
    """Store and return one verification result"""

    with SessionLocal() as session:
        if person_id is not None and session.get(Person, person_id) is None: # Manual comparisons may leave person_id as None
            raise ValueError("Person not found")

        record = VerificationRecord( # Keeps a historical name snapshot even if the Person is renamed or deleted later
            person_id = person_id,
            person_name = person_name,
            result = result,
            distance = distance,
            threshold = threshold,
            engine = engine,
        )
        session.add(record)
        session.commit()

        return verification_to_dict(record)

def list_verification_records() -> List[Dict]:
    """Return verification history from oldest to newest"""

    with SessionLocal() as session:
        statement = select(VerificationRecord).order_by( # Oldest first ordering matches the existing CSV analysis flow
            VerificationRecord.created_at,
            VerificationRecord.id,
        )
        records = session.scalars(statement).all()

        return [verification_to_dict(record) for record in records]
