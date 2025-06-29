"""
CogPrime Memory Module - Integration with mem0

This module provides memory management capabilities for CogPrime by integrating
mem0's persistence, vector search, and knowledge graph features.

The Memory class extends CogPrime's working memory with:
- Persistent storage for cognitive states
- Vector-based semantic search
- Knowledge graph representation and queries
- LLM-based fact extraction for learning

Basic usage:
    from cogprime.memory import Memory
    
    # Create a memory instance
    memory = Memory()
    
    # Store a cognitive state
    memory.store_cognitive_state("state_1", cognitive_state)
    
    # Retrieve a cognitive state
    state = memory.retrieve_cognitive_state("state_1")
    
    # Store a fact with semantic embedding
    memory.add_fact("Cats are mammals", tags=["biology", "animals"])
    
    # Search for semantically similar facts
    results = memory.semantic_search("What animals are pets?", limit=5)
    
    # Extract facts from text using LLM
    facts = memory.extract_facts("The brain contains approximately 86 billion neurons.")
"""

import os
import json
import uuid
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Import mem0 components
try:
    from mem0.memory.main import Memory as Mem0Memory
    from mem0.memory.graph_memory import MemoryGraph
    from mem0.configs.base import MemoryConfig
    from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    
# Import CogPrime components
from ..core.cognitive_core import CognitiveState

# Configure logging
logger = logging.getLogger(__name__)


class MemoryBackend:
    """Base class for memory backends."""
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value associated with the specified key.
        
        Returns:
            bool: True if the value was stored successfully, otherwise False.
        """
        raise NotImplementedError
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the specified key.
        
        Returns:
            The value corresponding to the key if found, or None if the key does not exist.
        """
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """
        Delete the value associated with the specified key.
        
        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """
        Searches for values matching the given query string.
        
        Parameters:
            query (str): The search query.
            limit (int): Maximum number of results to return.
        
        Returns:
            List of tuples containing the key, value, and a relevance score for each matching entry.
        """
        raise NotImplementedError


class DictMemoryBackend(MemoryBackend):
    """Simple in-memory dictionary backend."""
    
    def __init__(self):
        """
        Initialize the in-memory dictionary backend for storing key-value pairs.
        """
        self.data = {}
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value in memory under the specified key.
        
        Returns:
            bool: True if the value was stored successfully.
        """
        self.data[key] = value
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the given key from memory.
        
        Returns:
            The value stored under the specified key, or None if the key does not exist.
        """
        return self.data.get(key)
    
    def delete(self, key: str) -> bool:
        """
        Remove a value from memory by its key.
        
        Returns:
            bool: True if the key was found and deleted; False if the key did not exist.
        """
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """
        Searches for entries whose values contain the query string as a substring.
        
        Performs a case-insensitive substring search over all stored values. Returns a list of tuples containing the key, value, and a relevance score (1.0 for direct string matches, 0.8 for matches within dictionary values).
        
        Parameters:
            query (str): The substring to search for within stored values.
            limit (int): Maximum number of results to return.
        
        Returns:
            List[Tuple[str, Any, float]]: A list of (key, value, score) tuples for matching entries.
        """
        results = []
        for key, value in self.data.items():
            if isinstance(value, str) and query.lower() in value.lower():
                results.append((key, value, 1.0))  # Score 1.0 for exact match
            elif isinstance(value, dict) and any(
                isinstance(v, str) and query.lower() in v.lower() 
                for v in value.values()
            ):
                results.append((key, value, 0.8))  # Score 0.8 for partial match
        
        return results[:limit]


class Mem0MemoryBackend(MemoryBackend):
    """Memory backend using mem0 for persistence, vector search, and graph memory."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Mem0MemoryBackend with the provided configuration.
        
        Creates a mem0-based memory backend for persistent storage, semantic search, and knowledge graph operations. Raises ImportError if mem0 is not installed.
        
        Parameters:
            config (Dict, optional): Configuration dictionary for mem0 and backend options.
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0 is not available. Please install it using: "
                "pip install mem0"
            )
        
        self.config = config or {}
        
        # Create mem0 Memory instance
        memory_config = MemoryConfig(
            **self.config.get("memory", {})
        )
        self.memory = Mem0Memory(config=memory_config)
        
        # Enable graph memory if configured
        self.graph_enabled = self.config.get("enable_graph", True)
        
        logger.info("Initialized mem0 memory backend")
    
    def store(self, key: str, value: Any) -> bool:
        """
        Stores a value in the mem0 backend under the specified key.
        
        If the value is a CognitiveState, it is converted to a serializable dictionary before storage. The value is stored as a JSON string (unless already a string) along with metadata including the key, timestamp, and value type.
        
        Returns:
            bool: True if the value was stored successfully, False otherwise.
        """
        try:
            # Convert to JSON serializable format if needed
            if isinstance(value, CognitiveState):
                value = self._cognitive_state_to_dict(value)
            
            # Store in mem0
            content = json.dumps(value) if not isinstance(value, str) else value
            metadata = {
                "key": key,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": type(value).__name__
            }
            
            self.memory.add(
                content=content,
                metadata=metadata,
                id=key
            )
            return True
        except Exception as e:
            logger.error(f"Error storing value with key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from persistent storage by key, deserializing it to its original type if possible.
        
        Returns:
            The retrieved value, reconstructed as a `CognitiveState` object, dictionary, or raw content string, or `None` if the key is not found or an error occurs.
        """
        try:
            # Get from mem0
            result = self.memory.get(key)
            if not result:
                return None
            
            # Parse the content
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # Convert back to appropriate type
            value_type = metadata.get("type", "")
            if value_type == "CognitiveState":
                return self._dict_to_cognitive_state(json.loads(content))
            elif value_type == "dict":
                return json.loads(content)
            else:
                return content
        except Exception as e:
            logger.error(f"Error retrieving value with key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from memory by its key.
        
        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        try:
            self.memory.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting value with key {key}: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """
        Performs a semantic vector search for entries matching the query and returns a list of results with their keys, deserialized values, and similarity scores.
        
        Parameters:
            query (str): The search query string.
            limit (int): Maximum number of results to return.
        
        Returns:
            List of tuples containing the entry key, the deserialized value (CognitiveState, dict, or string), and the similarity score.
        """
        try:
            # Use mem0's vector search
            results = self.memory.search(
                text=query,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                key = result.get("id", "")
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                score = result.get("score", 0.0)
                
                # Convert content based on type
                value_type = metadata.get("type", "")
                if value_type == "CognitiveState":
                    value = self._dict_to_cognitive_state(json.loads(content))
                elif value_type == "dict":
                    value = json.loads(content)
                else:
                    value = content
                
                formatted_results.append((key, value, score))
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching with query {query}: {e}")
            return []
    
    def add_fact(self, fact: str, tags: List[str] = None, 
                metadata: Dict = None) -> str:
        """
                Adds a fact string to memory with associated tags and metadata, generating a unique ID if not provided.
                
                Parameters:
                    fact (str): The fact to store.
                    tags (List[str], optional): Tags for categorization.
                    metadata (Dict, optional): Additional metadata for the fact.
                
                Returns:
                    str: The unique ID assigned to the added fact, or None if the operation fails.
                """
        try:
            # Prepare metadata
            meta = metadata or {}
            meta["type"] = "fact"
            meta["tags"] = tags or []
            meta["timestamp"] = datetime.datetime.now().isoformat()
            
            # Generate ID if not provided
            fact_id = meta.get("id", f"fact_{uuid.uuid4()}")
            meta["id"] = fact_id
            
            # Add to mem0
            self.memory.add(
                content=fact,
                metadata=meta,
                id=fact_id
            )
            
            return fact_id
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            return None
    
    def extract_facts(self, text: str) -> List[Dict]:
        """
        Extracts facts from the provided text using an LLM-based extraction method.
        
        High-confidence facts (confidence > 0.7) are automatically added to memory with associated metadata.
        
        Parameters:
            text (str): The input text from which to extract facts.
        
        Returns:
            List[Dict]: A list of extracted facts, each represented as a dictionary with metadata.
        """
        try:
            # Use mem0's LLM-based fact extraction
            facts = self.memory.extract_facts(text)
            
            # Add extracted facts to memory
            for fact in facts:
                fact_text = fact.get("fact", "")
                confidence = fact.get("confidence", 0.0)
                
                if fact_text and confidence > 0.7:  # Only add high-confidence facts
                    self.add_fact(
                        fact=fact_text,
                        metadata={"confidence": confidence, "source": "extraction"}
                    )
            
            return facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []
    
    def add_relation(self, subject: str, predicate: str, object: str, 
                    metadata: Dict = None) -> bool:
        """
                    Adds a triple (subject, predicate, object) to the knowledge graph if graph memory is enabled.
                    
                    Parameters:
                        subject (str): The subject entity of the relation.
                        predicate (str): The type of relation.
                        object (str): The object entity of the relation.
                        metadata (Dict, optional): Additional metadata for the relation.
                    
                    Returns:
                        bool: True if the relation was successfully added; False if graph memory is disabled or an error occurs.
                    """
        if not self.graph_enabled:
            logger.warning("Graph memory not enabled")
            return False
        
        try:
            # Use mem0's graph memory
            self.memory.graph.add_relation(
                subject=subject,
                predicate=predicate,
                object=object,
                metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return False
    
    def query_graph(self, query: str) -> List[Dict]:
        """
        Executes a Cypher query on the knowledge graph and returns the results.
        
        Parameters:
            query (str): The Cypher query string to execute.
        
        Returns:
            List[Dict]: A list of dictionaries representing the query results. Returns an empty list if graph memory is not enabled or if an error occurs.
        """
        if not self.graph_enabled:
            logger.warning("Graph memory not enabled")
            return []
        
        try:
            # Use mem0's graph memory
            return self.memory.graph.query(query)
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return []
    
    def _cognitive_state_to_dict(self, state: CognitiveState) -> Dict:
        """
        Convert a CognitiveState object into a JSON-serializable dictionary for storage.
        
        Non-serializable objects in working memory are stringified, and tensor fields are converted to lists to ensure compatibility with JSON serialization.
        """
        # Convert PyTorch tensors to lists
        attention_focus = state.attention_focus.tolist() if hasattr(state.attention_focus, 'tolist') else state.attention_focus
        
        # Convert working memory (may contain non-serializable objects)
        working_memory = {}
        for k, v in state.working_memory.items():
            try:
                # Try to serialize, skip if not possible
                json.dumps({k: v})
                working_memory[k] = v
            except (TypeError, OverflowError):
                working_memory[k] = str(v)
        
        # Convert sensory buffer tensors to lists
        sensory_buffer = {}
        for k, v in state.sensory_buffer.items():
            if hasattr(v, 'tolist'):
                sensory_buffer[k] = v.tolist()
            else:
                sensory_buffer[k] = v
        
        # Convert current thought and last action to string representations
        current_thought = str(state.current_thought) if state.current_thought else None
        last_action = str(state.last_action) if state.last_action else None
        
        return {
            "attention_focus": attention_focus,
            "working_memory": working_memory,
            "emotional_valence": state.emotional_valence,
            "goal_stack": state.goal_stack,
            "sensory_buffer": sensory_buffer,
            "current_thought": current_thought,
            "last_action": last_action,
            "last_reward": state.last_reward,
            "total_reward": state.total_reward,
            "_type": "CognitiveState"
        }
    
    def _dict_to_cognitive_state(self, data: Dict) -> CognitiveState:
        """
        Reconstruct a CognitiveState object from a dictionary, converting list fields back to tensors and restoring basic attributes.
        
        Parameters:
            data (Dict): Dictionary containing serialized CognitiveState fields.
        
        Returns:
            CognitiveState: The reconstructed CognitiveState instance with tensor and attribute restoration.
        """
        import torch
        from ..modules.reasoning import Thought
        from ..modules.action import Action
        
        # Convert lists back to tensors
        attention_focus = torch.tensor(data.get("attention_focus", [0] * 512))
        
        # Create a new CognitiveState
        state = CognitiveState(
            attention_focus=attention_focus,
            working_memory=data.get("working_memory", {}),
            emotional_valence=data.get("emotional_valence", 0.0),
            goal_stack=data.get("goal_stack", []),
            sensory_buffer=data.get("sensory_buffer", {}),
            last_reward=data.get("last_reward", 0.0),
            total_reward=data.get("total_reward", 0.0)
        )
        
        # Note: current_thought and last_action are stored as strings
        # and would need proper reconstruction from the original classes
        # This is a simplified version that just stores the string representation
        
        return state


class Memory:
    """Memory management for CogPrime, integrating mem0 capabilities."""
    
    def __init__(self, backend_type: str = "mem0", config: Dict = None):
        """
        Initialize the Memory interface with the specified backend and configuration.
        
        Selects either the mem0-based backend (if available) or a simple in-memory dictionary backend for memory operations, based on the provided backend type and configuration.
        """
        self.config = config or {}
        
        # Create the appropriate backend
        if backend_type == "mem0" and MEM0_AVAILABLE:
            self.backend = Mem0MemoryBackend(config=self.config)
        else:
            if backend_type == "mem0" and not MEM0_AVAILABLE:
                logger.warning("mem0 not available, falling back to dictionary backend")
            self.backend = DictMemoryBackend()
        
        self.backend_type = backend_type
    
    def store_cognitive_state(self, key: str, state: CognitiveState) -> bool:
        """
        Store a CognitiveState object in memory under the specified key.
        
        Parameters:
            key (str): Identifier for storing the cognitive state.
            state (CognitiveState): The cognitive state to be stored.
        
        Returns:
            bool: True if the state was stored successfully, False otherwise.
        """
        return self.backend.store(key, state)
    
    def retrieve_cognitive_state(self, key: str) -> Optional[CognitiveState]:
        """
        Retrieve a stored CognitiveState object by its key.
        
        Parameters:
        	key (str): The unique identifier for the cognitive state.
        
        Returns:
        	CognitiveState or None: The retrieved cognitive state if found; otherwise, None.
        """
        return self.backend.retrieve(key)
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value associated with the specified key in memory.
        
        Parameters:
            key (str): The key under which to store the value.
            value (Any): The value to be stored.
        
        Returns:
            bool: True if the value was stored successfully, False otherwise.
        """
        return self.backend.store(key, value)
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the given key from memory.
        
        Parameters:
            key (str): The key whose value should be retrieved.
        
        Returns:
            The value associated with the key, or None if the key does not exist.
        """
        return self.backend.retrieve(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete the entry associated with the given key from memory.
        
        Returns:
            bool: True if the entry was successfully deleted, False otherwise.
        """
        return self.backend.delete(key)
    
    def add_fact(self, fact: str, tags: List[str] = None, 
                metadata: Dict = None) -> str:
        """
                Add a fact to memory, optionally with tags and metadata, and return its unique identifier.
                
                If the backend supports vector embeddings, the fact is stored with an embedding; otherwise, it is stored as a simple entry.
                """
        if hasattr(self.backend, "add_fact"):
            return self.backend.add_fact(fact, tags, metadata)
        else:
            # Fallback for simple backend
            fact_id = f"fact_{uuid.uuid4()}"
            self.backend.store(fact_id, {
                "content": fact,
                "tags": tags or [],
                "metadata": metadata or {},
                "timestamp": datetime.datetime.now().isoformat()
            })
            return fact_id
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """
        Performs a semantic search for content similar to the given query.
        
        Parameters:
            query (str): The text to search for semantically similar entries.
            limit (int): The maximum number of results to return.
        
        Returns:
            List[Tuple[str, Any, float]]: A list of tuples containing the key, value, and similarity score for each matching entry.
        """
        return self.backend.search(query, limit)
    
    def extract_facts(self, text: str) -> List[Dict]:
        """
        Extracts factual information from the provided text using an LLM-based extraction method if supported by the backend.
        
        Parameters:
            text (str): The input text from which to extract facts.
        
        Returns:
            List[Dict]: A list of extracted facts, each represented as a dictionary with associated metadata. Returns an empty list if fact extraction is not supported by the backend.
        """
        if hasattr(self.backend, "extract_facts"):
            return self.backend.extract_facts(text)
        else:
            logger.warning("Fact extraction not supported by this backend")
            return []
    
    def add_relation(self, subject: str, predicate: str, object: str, 
                    metadata: Dict = None) -> bool:
        """
                    Add a relation (triple) to the knowledge graph if supported by the backend.
                    
                    Parameters:
                        subject (str): The subject entity of the relation.
                        predicate (str): The type of relation.
                        object (str): The object entity of the relation.
                        metadata (Dict, optional): Additional metadata for the relation.
                    
                    Returns:
                        bool: True if the relation was added successfully; False if the backend does not support graph operations.
                    """
        if hasattr(self.backend, "add_relation"):
            return self.backend.add_relation(subject, predicate, object, metadata)
        else:
            logger.warning("Graph operations not supported by this backend")
            return False
    
    def query_graph(self, query: str) -> List[Dict]:
        """
        Executes a Cypher query on the knowledge graph and returns the results.
        
        Parameters:
            query (str): The Cypher query string to execute.
        
        Returns:
            List[Dict]: A list of dictionaries representing the query results, or an empty list if the backend does not support graph operations.
        """
        if hasattr(self.backend, "query_graph"):
            return self.backend.query_graph(query)
        else:
            logger.warning("Graph operations not supported by this backend")
            return []
    
    def update_working_memory(self, cognitive_state: CognitiveState, 
                             key: str, value: Any) -> CognitiveState:
        """
                             Update a CognitiveState's working memory with a new key-value pair.
                             
                             Parameters:
                                 cognitive_state (CognitiveState): The cognitive state to modify.
                                 key (str): The key to set or update in working memory.
                                 value (Any): The value to assign to the specified key.
                             
                             Returns:
                                 CognitiveState: The updated cognitive state with modified working memory.
                             """
        cognitive_state.working_memory[key] = value
        return cognitive_state
    
    def save_experience(self, state: CognitiveState, action: Any, 
                       reward: float, next_state: CognitiveState) -> str:
        """
                       Store a reinforcement learning experience tuple in memory and return its unique identifier.
                       
                       The experience includes the initial state, action taken (as a string), reward received, resulting state, timestamp, and is tagged as type "experience".
                       
                       Returns:
                           str: Unique identifier of the saved experience.
                       """
        experience_id = f"exp_{uuid.uuid4()}"
        
        # Store the experience
        self.store(experience_id, {
            "state": state,
            "action": str(action),  # Convert action to string for storage
            "reward": reward,
            "next_state": next_state,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "experience"
        })
        
        return experience_id
    
    def get_recent_experiences(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve a list of recent reinforcement learning experiences stored in memory.
        
        Parameters:
            limit (int): The maximum number of experiences to return.
        
        Returns:
            List[Dict]: A list of experience records, each represented as a dictionary. Returns an empty list if the backend does not support search.
        """
        if hasattr(self.backend, "search"):
            results = self.backend.search("type:experience", limit=limit)
            return [value for _, value, _ in results]
        else:
            # Simple backend doesn't support search
            return []


# Factory function for creating memory instances

def create_memory(backend_type: str = "mem0", config: Dict = None) -> Memory:
    """
    Instantiate and return a Memory object with the specified backend and configuration.
    
    Parameters:
    	backend_type (str): The type of backend to use ("mem0" for persistent, feature-rich memory or "dict" for in-memory storage).
    	config (Dict, optional): Configuration dictionary for the selected backend.
    
    Returns:
    	Memory: A Memory instance using the chosen backend and configuration.
    """
    return Memory(backend_type=backend_type, config=config)


# Integration with CogPrime modules

def integrate_with_cognitive_core(memory: Memory, core) -> None:
    """
    Integrates a Memory instance with a CogPrime cognitive core by registering a callback that stores the current cognitive state after each cognitive cycle.
    """
    # Store the current cognitive state after each cycle
    def store_state_after_cycle(sensory_input, reward, action):
        """
        Stores the current cognitive state in memory after each cognitive cycle.
        
        This function retrieves the latest cognitive state from the core and saves it in memory with a unique key.
        """
        state = core.get_cognitive_state()
        memory.store_cognitive_state(f"state_{uuid.uuid4()}", state)
    
    # Register the callback
    if hasattr(core, "register_cycle_callback"):
        core.register_cycle_callback(store_state_after_cycle)
