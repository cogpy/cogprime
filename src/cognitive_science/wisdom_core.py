from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Optional

class WisdomTheoryType(Enum):
    PRODUCT = auto()  # What wisdom is
    PROCESS = auto()  # How one becomes wise

class KnowledgeType(Enum):
    DESCRIPTIVE = auto()  # Knowledge that (e.g., facts)
    INTERPRETIVE = auto()  # Grasping significance
    SOPHIA = auto()       # Theoretical wisdom
    PHRONESIS = auto()    # Practical wisdom

class WisdomDimension(Enum):
    COGNITIVE = auto()    # Understanding significance
    REFLECTIVE = auto()   # Multi-perspectival, self-examination
    AFFECTIVE = auto()    # Compassion, agape

@dataclass
class WisdomState:
    cognitive_understanding: float  # 0-1 scale
    reflective_capacity: float     # 0-1 scale
    affective_development: float   # 0-1 scale
    current_knowledge: Dict[KnowledgeType, float]
    active_perspectives: List[str]
    self_transcendence_level: float

@dataclass
class TransformationProcess:
    from_state: WisdomState
    to_state: WisdomState
    insight_gained: str
    illusion_overcome: str
    reality_accessed: str

class WisdomCore:
    def __init__(self):
        self.current_state = WisdomState(
            cognitive_understanding=0.0,
            reflective_capacity=0.0,
            affective_development=0.0,
            current_knowledge={k: 0.0 for k in KnowledgeType},
            active_perspectives=[],
            self_transcendence_level=0.0
        )
        self.transformation_history: List[TransformationProcess] = []
    
    def evaluate_wisdom(self) -> Dict[WisdomDimension, float]:
        """Evaluate current wisdom level across dimensions."""
        return {
            WisdomDimension.COGNITIVE: self.current_state.cognitive_understanding,
            WisdomDimension.REFLECTIVE: self.current_state.reflective_capacity,
            WisdomDimension.AFFECTIVE: self.current_state.affective_development
        }
    
    def process_transformation(self, insight: str, illusion: str, reality: str) -> None:
        """Process a transformative experience that increases wisdom."""
        old_state = self.current_state
        
        # Update state based on transformation
        self.current_state.cognitive_understanding += 0.1
        self.current_state.reflective_capacity += 0.1
        self.current_state.affective_development += 0.1
        self.current_state.self_transcendence_level += 0.1
        
        # Record transformation
        self.transformation_history.append(
            TransformationProcess(
                from_state=old_state,
                to_state=self.current_state,
                insight_gained=insight,
                illusion_overcome=illusion,
                reality_accessed=reality
            )
        )
    
    def add_perspective(self, perspective: str) -> None:
        """Add a new perspective to active perspectives."""
        if perspective not in self.current_state.active_perspectives:
            self.current_state.active_perspectives.append(perspective)
            self.current_state.reflective_capacity += 0.05
    
    def develop_knowledge(self, knowledge_type: KnowledgeType, amount: float) -> None:
        """Develop a specific type of knowledge."""
        current = self.current_state.current_knowledge.get(knowledge_type, 0.0)
        self.current_state.current_knowledge[knowledge_type] = min(1.0, current + amount)
        
        # Theoretical and practical wisdom enhance cognitive understanding
        if knowledge_type in [KnowledgeType.SOPHIA, KnowledgeType.PHRONESIS]:
            self.current_state.cognitive_understanding += amount * 0.1
    
    def cultivate_agape(self) -> None:
        """Cultivate agape to enhance affective development."""
        self.current_state.affective_development += 0.1
        self.current_state.self_transcendence_level += 0.05
    
    def get_development_recommendations(self) -> List[str]:
        """Get recommendations for further wisdom development."""
        recommendations = []
        
        if self.current_state.cognitive_understanding < 0.7:
            recommendations.append("Deepen understanding through study and reflection")
            
        if self.current_state.reflective_capacity < 0.7:
            recommendations.append("Practice perspective-taking and self-examination")
            
        if self.current_state.affective_development < 0.7:
            recommendations.append("Cultivate compassion and agape")
            
        if self.current_state.self_transcendence_level < 0.7:
            recommendations.append("Engage in transformative practices")
            
        return recommendations 