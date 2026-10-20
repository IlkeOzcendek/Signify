from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base

def utc_now() -> datetime: # -> Says utc_now() function returns in type datetime
    """Returns the current time in UTC"""
    return datetime.now(timezone.utc)

def generate_uuid() -> str: # The goal is to prevent the same identity from being created twice to avoid filename conflicts and to generate unique identifiers that are difficult to guess
    """Creates a globally unique string identifier"""

    return str(uuid4())

class Person(Base):
    """A person who is registered in Signify"""

    __tablename__ = "people"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key = True,
        default = generate_uuid,
    )

    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable = False,
        index = True,
    )

    profile_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable = True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone = True),
        nullable = False,
        default = utc_now,
    )

    signatures: Mapped[List["Signature"]] = relationship(
        back_populates =  "person",
        cascade = "all, delete-orphan",
        passive_deletes = True,
    )

class Signature(Base):
    """A genuine reference signature belonging to a registered person"""

    __tablename__ = "signatures"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key = True,
        default = generate_uuid,
    )

    person_id: Mapped[str] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable = False,
        index = True,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable = False,
        unique = True,
    )

    original_filename: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable = True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone = True),
        nullable = False,
        default = utc_now,
    )

    person: Mapped["Person"] = relationship(
        back_populates = "signatures",
    )

class VerificationRecord(Base):
    """The result of one signature verification operation."""

    __tablename__ = "verification_history"

    id: Mapped[int] = mapped_column(
        primary_key = True,
        autoincrement = True,
    )

    person_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL"),
        nullable = True,
        index = True,
    )

    person_name: Mapped[str] = mapped_column(
        String(200),
        nullable = False,
    )

    result: Mapped[str] = mapped_column(
        String(50),
        nullable =False,
    )

    distance: Mapped[float] = mapped_column(
        Float,
        nullable = False,
    )

    threshold: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable = True,
    )

    engine: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable = True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone = True),
        nullable = False,
        default = utc_now,
        index = True,
    )
