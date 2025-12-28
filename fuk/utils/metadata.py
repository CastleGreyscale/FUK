"""
SQLite database for tracking image generations
Stores prompts, settings, LoRAs used, and output paths
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import fuk.config as config


class MetadataDB:
    """Manage generation metadata in SQLite"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.METADATA_DB
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.conn.cursor()
        
        # Main generations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                cfg_scale REAL NOT NULL,
                shift REAL,
                seed INTEGER NOT NULL,
                output_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # LoRAs used in each generation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_loras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                lora_name TEXT NOT NULL,
                strength REAL NOT NULL,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)
        
        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON generations(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_seed 
            ON generations(seed)
        """)
        
        self.conn.commit()
    
    def store_generation(self, metadata: Dict[str, Any]) -> int:
        """
        Store generation metadata and return the generation ID
        
        Args:
            metadata: Dict from QwenGenerator.generate()
        
        Returns:
            generation_id for referencing this generation
        """
        cursor = self.conn.cursor()
        
        # Insert main generation record
        cursor.execute("""
            INSERT INTO generations (
                timestamp, prompt, negative_prompt, width, height,
                steps, cfg_scale, shift, seed, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata["timestamp"],
            metadata["prompt"],
            metadata["negative_prompt"],
            metadata["width"],
            metadata["height"],
            metadata["steps"],
            metadata["cfg_scale"],
            metadata.get("shift"),
            metadata["seed"],
            metadata["output_path"],
        ))
        
        generation_id = cursor.lastrowid
        
        # Insert LoRA records if any
        for lora_name, strength in metadata.get("loras", []):
            cursor.execute("""
                INSERT INTO generation_loras (generation_id, lora_name, strength)
                VALUES (?, ?, ?)
            """, (generation_id, lora_name, strength))
        
        self.conn.commit()
        
        return generation_id
    
    def get_generation(self, generation_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a generation by ID with its LoRAs"""
        cursor = self.conn.cursor()
        
        # Get main record
        cursor.execute("""
            SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Convert to dict
        gen = dict(row)
        
        # Get associated LoRAs
        cursor.execute("""
            SELECT lora_name, strength FROM generation_loras
            WHERE generation_id = ?
        """, (generation_id,))
        
        gen["loras"] = [(r["lora_name"], r["strength"]) for r in cursor.fetchall()]
        
        return gen
    
    def search_by_prompt(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search generations by prompt text"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generations
            WHERE prompt LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{search_term}%", limit))
        
        results = []
        for row in cursor.fetchall():
            gen = dict(row)
            
            # Get LoRAs for this generation
            cursor.execute("""
                SELECT lora_name, strength FROM generation_loras
                WHERE generation_id = ?
            """, (gen["id"],))
            
            gen["loras"] = [(r["lora_name"], r["strength"]) for r in cursor.fetchall()]
            results.append(gen)
        
        return results
    
    def search_by_seed(self, seed: int) -> Optional[Dict[str, Any]]:
        """Find generation by exact seed"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generations WHERE seed = ?
        """, (seed,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        gen = dict(row)
        
        # Get LoRAs
        cursor.execute("""
            SELECT lora_name, strength FROM generation_loras
            WHERE generation_id = ?
        """, (gen["id"],))
        
        gen["loras"] = [(r["lora_name"], r["strength"]) for r in cursor.fetchall()]
        
        return gen
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent generations"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generations
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            gen = dict(row)
            
            cursor.execute("""
                SELECT lora_name, strength FROM generation_loras
                WHERE generation_id = ?
            """, (gen["id"],))
            
            gen["loras"] = [(r["lora_name"], r["strength"]) for r in cursor.fetchall()]
            results.append(gen)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM generations")
        total = cursor.fetchone()["total"]
        
        cursor.execute("""
            SELECT lora_name, COUNT(*) as usage_count
            FROM generation_loras
            GROUP BY lora_name
            ORDER BY usage_count DESC
            LIMIT 10
        """)
        top_loras = [(r["lora_name"], r["usage_count"]) for r in cursor.fetchall()]
        
        cursor.execute("""
            SELECT AVG(steps) as avg_steps, AVG(cfg_scale) as avg_cfg
            FROM generations
        """)
        averages = cursor.fetchone()
        
        return {
            "total_generations": total,
            "top_loras": top_loras,
            "avg_steps": averages["avg_steps"],
            "avg_cfg": averages["avg_cfg"],
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function to integrate with generator
def store_generation_metadata(metadata: Dict[str, Any]) -> int:
    """Store metadata and return generation ID"""
    with MetadataDB() as db:
        return db.store_generation(metadata)
