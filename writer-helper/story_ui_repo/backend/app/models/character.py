from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class CharacterCreate(BaseModel):
    story_id: Optional[str] = ""
    name: str
    alias: Optional[str] = ""
    aliases: Optional[List[str]] = Field(default_factory=list)
    age: Optional[str] = ""
    gender: Optional[str] = ""
    role: Optional[str] = "Protagonist"
    personality_traits: Optional[List[str]] = Field(default_factory=list)
    backstory: Optional[str] = ""
    physical_description: Optional[str] = ""
    relationships: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    motivations: Optional[str] = ""
    goals: Optional[str] = ""
    fears: Optional[str] = ""
    secrets: Optional[str] = ""
    current_status: Optional[str] = "Alive"
    first_appearance: Optional[str] = "Chapter 1"
    last_appearance: Optional[str] = "Chapter 1"
    portrait_url: Optional[str] = None
    character_arc: Optional[str] = ""
    status_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    evidence: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    confidence: Optional[float] = 1.0

class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    alias: Optional[str] = None
    aliases: Optional[List[str]] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    role: Optional[str] = None
    personality_traits: Optional[List[str]] = None
    backstory: Optional[str] = None
    physical_description: Optional[str] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    motivations: Optional[str] = None
    goals: Optional[str] = None
    fears: Optional[str] = None
    secrets: Optional[str] = None
    current_status: Optional[str] = None
    first_appearance: Optional[str] = None
    last_appearance: Optional[str] = None
    portrait_url: Optional[str] = None
    character_arc: Optional[str] = None
    status_history: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None

class Character(BaseModel):
    id: str = Field(default_factory=lambda: f"char_{uuid.uuid4().hex[:10]}")
    story_id: str
    name: str
    alias: str = ""
    aliases: List[str] = Field(default_factory=list)
    age: str = ""
    gender: Optional[str] = ""
    role: str = "Protagonist"
    personality_traits: List[str] = Field(default_factory=list)
    backstory: str = ""
    physical_description: str = ""
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    motivations: str = ""
    goals: str = ""
    fears: str = ""
    secrets: str = ""
    current_status: str = "Alive"
    first_appearance: str = "Chapter 1"
    last_appearance: str = "Chapter 1"
    portrait_url: Optional[str] = None
    character_arc: Optional[str] = ""
    status_history: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
