#!/usr/bin/env python3
"""Fix memory database schema"""

import sqlite3
from pathlib import Path


def fix_database():
    memory_dir = Path("memory_store")
    db_path = memory_dir / "memory.db"

    if not db_path.exists():
        print("No database found, nothing to fix")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if conversations table exists
    cursor.execute(
        """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='conversations'
    """
    )

    if not cursor.fetchone():
        print("Creating conversations table...")
        cursor.execute(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                metadata TEXT,
                timestamp DATETIME
            )
        """
        )
        conn.commit()
        print("Conversations table created")
    else:
        print("Conversations table already exists")

    conn.close()
    print("Database fixed!")


if __name__ == "__main__":
    fix_database()
