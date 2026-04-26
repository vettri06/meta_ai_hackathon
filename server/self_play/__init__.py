"""Self-play and adaptive curriculum system for the AI Firewall environment.

Provides recursive skill amplification through:
  - WeaknessAnalyzer: 8-dimension regret-based performance profiling
  - CurriculumEngine: ADR parameter generation targeting agent weaknesses
  - SelfPlayArena: Elo-rated training loop with mastery gating
"""
from server.self_play.weakness_analyzer import WeaknessAnalyzer, WeaknessProfile
from server.self_play.curriculum_engine import CurriculumEngine
from server.self_play.arena import SelfPlayArena

__all__ = [
    "WeaknessAnalyzer",
    "WeaknessProfile",
    "CurriculumEngine",
    "SelfPlayArena",
]
