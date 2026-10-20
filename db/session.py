from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Gives the root of the project automatically

# SQLite database is going to be kept as one folder
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "signify.db"

STORAGE_DIR = DATABASE_DIR / "storage" # Profiles and the signatures are not going to take place at SQLite but here

DATABASE_DIR.mkdir(parents=True, exist_ok=True) # Parents creates the needed folders if needed, if exists therefore do not give error
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

class Base(DeclarativeBase): # DeclarativeBase is SQLAlchemy's table creation engine
  """The common base class of all SQLAlchemy table models"""

  pass # There is nothing yet

engine = create_engine( # Engine is the connection manager that enables access to the database
  DATABASE_URL,
  connect_args = {"check_same_thread": False}, # Different threads are going to be able to connect at the same time to the engine, especially Streamlit can use different threads at the same time
)
#If you want to make a session first create an engine
@event.listens_for(engine, "connect") # It automatically runs a process whenever a new SQLite connection is opened
def enable_sqlite_foreign_keys(dbapi_connection,connection_Record):
  """Enables foreign key checking on every SQLite connection"""

  cursor = dbapi_connection.cursor() # Cursor is a pen or terminal that you use to type commands into the database
  cursor.execute("PRAGMA foreign_keys = ON") # PRAGMA only changes the settings of SQLite
  cursor.close() # Work done on cursor, close it

SessionLocal = sessionmaker (
  bind = engine, # This session should be in usage in SQLite through the engine reaching to signify.db
  autoflush = False, # Without commitment after each query it will not be sent automatically, flush() or commit() usage is needed
  expire_on_commit = False, # After commit, the information that is used is not expired; it is still valid and can be used later on
)
