"""
CogPrime AtomSpace - Distributed Hypergraph Knowledge Base

This module implements a Python interface to the AtomSpace hypergraph database,
following OpenCog AtomSpace patterns but adapted for Python and extended with
distributed capabilities via node9 namespace and mem0 persistence.

The AtomSpace is a hypergraph database designed for knowledge representation,
reasoning, and distributed cognition.

Basic usage:
    from cogprime.atomspace import AtomSpace, Node, Link
    
    # Create a local atomspace
    atomspace = AtomSpace()
    
    # Create atoms
    concept_cat = Node("ConceptNode", "cat")
    concept_animal = Node("ConceptNode", "animal")
    
    # Create a link between them
    inheritance = Link("InheritanceLink", [concept_cat, concept_animal])
    
    # Add to atomspace
    atomspace.add(concept_cat)
    atomspace.add(concept_animal)
    atomspace.add(inheritance)
    
    # Pattern matching
    pattern = Link("InheritanceLink", [Node("ConceptNode", "cat"), None])
    results = atomspace.query(pattern)
"""

import uuid
import logging
import threading
import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
AtomType = str
AtomValue = Any
AtomID = str  # UUID string representation


class TruthValue:
    """Truth value representation for atoms."""
    
    def __init__(self, strength: float = 1.0, confidence: float = 1.0):
        """Initialize with strength and confidence values.
        
        Args:
            strength: Truth value strength (0.0 to 1.0)
            confidence: Confidence in the truth value (0.0 to 1.0)
        """
        self.strength = max(0.0, min(1.0, strength))
        self.confidence = max(0.0, min(1.0, confidence))
    
    def __repr__(self) -> str:
        return f"TruthValue(strength={self.strength:.3f}, confidence={self.confidence:.3f})"


class AttentionValue:
    """Attention value representation for atoms."""
    
    def __init__(self, sti: float = 0.0, lti: float = 0.0, vlti: bool = False):
        """Initialize with attention values.
        
        Args:
            sti: Short-Term Importance
            lti: Long-Term Importance
            vlti: Very Long-Term Importance flag
        """
        self.sti = sti  # Short-Term Importance
        self.lti = lti  # Long-Term Importance
        self.vlti = vlti  # Very Long-Term Importance flag
    
    def __repr__(self) -> str:
        return f"AttentionValue(sti={self.sti:.3f}, lti={self.lti:.3f}, vlti={self.vlti})"


class Atom(ABC):
    """Base class for all atoms in the AtomSpace."""
    
    def __init__(self, atom_type: AtomType, name: str = None):
        """Initialize an atom with a type and optional name.
        
        Args:
            atom_type: The type of the atom (e.g., "ConceptNode")
            name: Optional name for the atom
        """
        self.atom_type = atom_type
        self.name = name
        self.id = str(uuid.uuid4())
        self.tv = TruthValue()
        self.av = AttentionValue()
        self.incoming_set = set()  # Links that contain this atom
        self.atomspace = None  # Reference to containing atomspace
        self.values = {}  # Additional key-value pairs
    
    @abstractmethod
    def is_node(self) -> bool:
        """Return True if this is a node, False if it's a link."""
        pass
    
    @abstractmethod
    def is_link(self) -> bool:
        """Return True if this is a link, False if it's a node."""
        pass
    
    @abstractmethod
    def get_hash(self) -> int:
        """Get a hash value for this atom."""
        pass
    
    def get_value(self, key: str) -> Any:
        """Get a value by key."""
        return self.values.get(key)
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a value for a key."""
        self.values[key] = value
    
    def set_truth_value(self, tv: TruthValue) -> None:
        """Set the truth value for this atom."""
        self.tv = tv
    
    def get_truth_value(self) -> TruthValue:
        """Get the truth value for this atom."""
        return self.tv
    
    def set_attention_value(self, av: AttentionValue) -> None:
        """Set the attention value for this atom."""
        self.av = av
    
    def get_attention_value(self) -> AttentionValue:
        """Get the attention value for this atom."""
        return self.av
    
    def add_to_incoming_set(self, link) -> None:
        """Add a link to this atom's incoming set."""
        self.incoming_set.add(link)
    
    def remove_from_incoming_set(self, link) -> None:
        """Remove a link from this atom's incoming set."""
        self.incoming_set.discard(link)
    
    def get_incoming_set(self) -> Set:
        """Get the set of links that contain this atom."""
        return self.incoming_set
    
    def __eq__(self, other) -> bool:
        """Check if two atoms are equal."""
        if not isinstance(other, Atom):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash function for atoms."""
        return hash(self.id)


class Node(Atom):
    """Node class representing vertices in the AtomSpace hypergraph."""
    
    def __init__(self, atom_type: AtomType, name: str):
        """Initialize a node with type and name.
        
        Args:
            atom_type: The type of the node (e.g., "ConceptNode")
            name: The name of the node
        """
        super().__init__(atom_type, name)
        if name is None:
            raise ValueError("Node must have a name")
    
    def is_node(self) -> bool:
        """Return True as this is a node."""
        return True
    
    def is_link(self) -> bool:
        """Return False as this is not a link."""
        return False
    
    def get_hash(self) -> int:
        """Get a hash value for this node."""
        return hash((self.atom_type, self.name))
    
    def __repr__(self) -> str:
        return f"{self.atom_type}('{self.name}')"


class Link(Atom):
    """Link class representing hyperedges in the AtomSpace hypergraph."""
    
    def __init__(self, atom_type: AtomType, outgoing_set: List[Atom]):
        """Initialize a link with type and outgoing set.
        
        Args:
            atom_type: The type of the link (e.g., "InheritanceLink")
            outgoing_set: List of atoms that this link connects
        """
        super().__init__(atom_type)
        self.outgoing_set = outgoing_set if outgoing_set else []
    
    def is_node(self) -> bool:
        """Return False as this is not a node."""
        return False
    
    def is_link(self) -> bool:
        """Return True as this is a link."""
        return True
    
    def get_hash(self) -> int:
        """Get a hash value for this link."""
        return hash((self.atom_type, tuple(atom.id for atom in self.outgoing_set)))
    
    def get_arity(self) -> int:
        """Get the arity (number of atoms in outgoing set) of this link."""
        return len(self.outgoing_set)
    
    def get_outgoing_set(self) -> List[Atom]:
        """Get the outgoing set of this link."""
        return self.outgoing_set
    
    def __repr__(self) -> str:
        outgoing_repr = ", ".join(repr(atom) for atom in self.outgoing_set)
        return f"{self.atom_type}({outgoing_repr})"


class AtomSpaceBackend(ABC):
    """Abstract base class for AtomSpace backends."""
    
    @abstractmethod
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the backend storage."""
        pass
    
    @abstractmethod
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from the backend storage."""
        pass
    
    @abstractmethod
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get an atom by ID."""
        pass
    
    @abstractmethod
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name."""
        pass
    
    @abstractmethod
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a given type."""
        pass
    
    @abstractmethod
    def query(self, pattern: Atom) -> List[Atom]:
        """Query atoms matching a pattern."""
        pass
    
    @abstractmethod
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching."""
        pass


class LocalAtomSpaceBackend(AtomSpaceBackend):
    """Local in-memory implementation of the AtomSpace backend."""
    
    def __init__(self):
        """Initialize a local AtomSpace backend."""
        self.atoms_by_id = {}  # id -> atom
        self.nodes_by_type_name = {}  # (type, name) -> node
        self.atoms_by_type = {}  # type -> set of atoms
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the local storage."""
        self.atoms_by_id[atom.id] = atom
        
        # Index by type
        if atom.atom_type not in self.atoms_by_type:
            self.atoms_by_type[atom.atom_type] = set()
        self.atoms_by_type[atom.atom_type].add(atom)
        
        # Index nodes by type and name
        if atom.is_node():
            key = (atom.atom_type, atom.name)
            self.nodes_by_type_name[key] = atom
        
        # Update incoming sets for atoms in the outgoing set
        if atom.is_link():
            for outgoing_atom in atom.outgoing_set:
                outgoing_atom.add_to_incoming_set(atom)
        
        return atom
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from the local storage."""
        if atom.id not in self.atoms_by_id:
            return False
        
        # Remove from type index
        if atom.atom_type in self.atoms_by_type:
            self.atoms_by_type[atom.atom_type].discard(atom)
            if not self.atoms_by_type[atom.atom_type]:
                del self.atoms_by_type[atom.atom_type]
        
        # Remove from type-name index if it's a node
        if atom.is_node():
            key = (atom.atom_type, atom.name)
            if key in self.nodes_by_type_name:
                del self.nodes_by_type_name[key]
        
        # Update incoming sets for atoms in the outgoing set
        if atom.is_link():
            for outgoing_atom in atom.outgoing_set:
                outgoing_atom.remove_from_incoming_set(atom)
        
        # Remove from main index
        del self.atoms_by_id[atom.id]
        
        return True
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get an atom by ID."""
        return self.atoms_by_id.get(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name."""
        key = (atom_type, name)
        return self.nodes_by_type_name.get(key)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a given type."""
        return list(self.atoms_by_type.get(atom_type, set()))
    
    def query(self, pattern: Atom) -> List[Atom]:
        """Query atoms matching a pattern."""
        results = []
        
        # If pattern is a node, look it up directly
        if pattern.is_node():
            atom = self.get_atom_by_type_name(pattern.atom_type, pattern.name)
            if atom:
                results.append(atom)
            return results
        
        # If pattern is a link, find matching links
        if pattern.is_link():
            # Get all links of the same type
            candidates = self.get_atoms_by_type(pattern.atom_type)
            
            for candidate in candidates:
                if not candidate.is_link():
                    continue
                
                # Check if outgoing sets match
                if len(candidate.outgoing_set) != len(pattern.outgoing_set):
                    continue
                
                match = True
                for i, pattern_atom in enumerate(pattern.outgoing_set):
                    candidate_atom = candidate.outgoing_set[i]
                    
                    # None is a wildcard
                    if pattern_atom is None:
                        continue
                    
                    # Check if atoms match
                    if pattern_atom.is_node() and candidate_atom.is_node():
                        if (pattern_atom.atom_type != candidate_atom.atom_type or
                                pattern_atom.name != candidate_atom.name):
                            match = False
                            break
                    elif pattern_atom.is_link() and candidate_atom.is_link():
                        # Recursive match for nested links
                        if pattern_atom.atom_type != candidate_atom.atom_type:
                            match = False
                            break
                        # TODO: Implement recursive matching for nested links
                    else:
                        match = False
                        break
                
                if match:
                    results.append(candidate)
        
        return results
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching."""
        # TODO: Implement more sophisticated pattern matching
        # This is a placeholder for advanced pattern matching functionality
        return []


class Node9AtomSpaceBackend(AtomSpaceBackend):
    """AtomSpace backend that uses node9 namespace for distributed operation."""
    
    def __init__(self, namespace_path: str = "/cog/space"):
        """Initialize a node9 namespace-based AtomSpace backend.
        
        Args:
            namespace_path: Path in the node9 namespace for AtomSpace
        """
        self.namespace_path = namespace_path
        # TODO: Implement node9 namespace integration
        logger.warning("Node9AtomSpaceBackend is not fully implemented yet")
        
        # Temporary fallback to local backend
        self._local_backend = LocalAtomSpaceBackend()
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.add_atom(atom)
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.remove_atom(atom)
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get an atom by ID from the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atom(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name from the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a given type from the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """Query atoms matching a pattern in the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching in the node9 namespace."""
        # TODO: Implement node9 namespace integration
        return self._local_backend.pattern_match(pattern)


class Mem0AtomSpaceBackend(AtomSpaceBackend):
    """AtomSpace backend that uses mem0 for persistence and vector search."""
    
    def __init__(self, config: Dict = None):
        """Initialize a mem0-based AtomSpace backend.
        
        Args:
            config: Configuration for mem0 integration
        """
        self.config = config or {}
        # TODO: Implement mem0 integration
        logger.warning("Mem0AtomSpaceBackend is not fully implemented yet")
        
        # Temporary fallback to local backend
        self._local_backend = LocalAtomSpaceBackend()
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.add_atom(atom)
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.remove_atom(atom)
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get an atom by ID from mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.get_atom(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name from mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a given type from mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """Query atoms matching a pattern in mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching in mem0 storage."""
        # TODO: Implement mem0 integration
        return self._local_backend.pattern_match(pattern)
    
    def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple[Atom, float]]:
        """Perform vector similarity search using mem0.
        
        Args:
            vector: The query vector
            limit: Maximum number of results to return
            
        Returns:
            List of (atom, similarity_score) tuples
        """
        # TODO: Implement mem0 vector search integration
        return []


class BackendType(Enum):
    """Enum for AtomSpace backend types."""
    LOCAL = "local"
    NODE9 = "node9"
    MEM0 = "mem0"
    DISTRIBUTED = "distributed"  # Uses both node9 and mem0


class AtomSpace:
    """Main AtomSpace class for managing atoms and performing operations."""
    
    def __init__(self, backend_type: Union[BackendType, str] = BackendType.LOCAL, 
                 config: Dict = None):
        """Initialize an AtomSpace with the specified backend.
        
        Args:
            backend_type: Type of backend to use (local, node9, mem0, distributed)
            config: Configuration for the backend
        """
        self.config = config or {}
        
        # Convert string to enum if needed
        if isinstance(backend_type, str):
            backend_type = BackendType(backend_type)
        
        # Create the appropriate backend
        if backend_type == BackendType.LOCAL:
            self.backend = LocalAtomSpaceBackend()
        elif backend_type == BackendType.NODE9:
            self.backend = Node9AtomSpaceBackend(
                namespace_path=self.config.get('namespace_path', '/cog/space')
            )
        elif backend_type == BackendType.MEM0:
            self.backend = Mem0AtomSpaceBackend(config=self.config)
        elif backend_type == BackendType.DISTRIBUTED:
            # TODO: Implement a combined backend that uses both node9 and mem0
            self.backend = LocalAtomSpaceBackend()
            logger.warning("Distributed backend not fully implemented yet, using local backend")
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        self.backend_type = backend_type
        self._event_handlers = {}  # Event name -> list of handlers
    
    def add(self, atom: Atom) -> Atom:
        """Add an atom to the AtomSpace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The added atom (may be a different instance if already exists)
        """
        # Set the atomspace reference
        atom.atomspace = weakref.ref(self)
        
        # Add to backend
        result = self.backend.add_atom(atom)
        
        # Trigger events
        self._trigger_event('atom_added', result)
        
        return result
    
    def remove(self, atom: Atom) -> bool:
        """Remove an atom from the AtomSpace.
        
        Args:
            atom: The atom to remove
            
        Returns:
            True if the atom was removed, False if it wasn't in the AtomSpace
        """
        result = self.backend.remove_atom(atom)
        
        if result:
            # Clear the atomspace reference
            atom.atomspace = None
            
            # Trigger events
            self._trigger_event('atom_removed', atom)
        
        return result
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get an atom by ID.
        
        Args:
            atom_id: The ID of the atom to get
            
        Returns:
            The atom, or None if not found
        """
        return self.backend.get_atom(atom_id)
    
    def get_node(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name.
        
        Args:
            atom_type: The type of the node
            name: The name of the node
            
        Returns:
            The node, or None if not found
        """
        return self.backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a given type.
        
        Args:
            atom_type: The type of atoms to get
            
        Returns:
            List of atoms of the specified type
        """
        return self.backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """Query atoms matching a pattern.
        
        Args:
            pattern: The pattern to match (can contain None as wildcards)
            
        Returns:
            List of matching atoms
        """
        return self.backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching.
        
        Args:
            pattern: A dictionary describing the pattern to match
            
        Returns:
            List of dictionaries containing matches
        """
        return self.backend.pattern_match(pattern)
    
    def get_all_atoms(self) -> List[Atom]:
        """Get all atoms in the AtomSpace.
        
        Returns:
            List of all atoms
        """
        return list(self.backend.atoms_by_id.values())
    
    def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple[Atom, float]]:
        """Perform vector similarity search (requires mem0 backend).
        
        Args:
            vector: The query vector
            limit: Maximum number of results to return
            
        Returns:
            List of (atom, similarity_score) tuples
        """
        if isinstance(self.backend, Mem0AtomSpaceBackend):
            return self.backend.vector_search(vector, limit)
        else:
            logger.warning("Vector search requires mem0 backend")
            return []
    
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """Register a handler for an event.
        
        Args:
            event_name: The name of the event
            handler: The handler function
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
    
    def unregister_event_handler(self, event_name: str, handler: Callable) -> bool:
        """Unregister a handler for an event.
        
        Args:
            event_name: The name of the event
            handler: The handler function
            
        Returns:
            True if the handler was removed, False if it wasn't registered
        """
        if event_name in self._event_handlers:
            if handler in self._event_handlers[event_name]:
                self._event_handlers[event_name].remove(handler)
                return True
        return False
    
    def _trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """Trigger an event, calling all registered handlers.
        
        Args:
            event_name: The name of the event
            *args, **kwargs: Arguments to pass to the handlers
        """
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")


# Factory functions for creating atoms

def create_node(atom_type: AtomType, name: str) -> Node:
    """Create a new node.
    
    Args:
        atom_type: The type of the node
        name: The name of the node
        
    Returns:
        A new Node instance
    """
    return Node(atom_type, name)


def create_link(atom_type: AtomType, outgoing_set: List[Atom]) -> Link:
    """Create a new link.
    
    Args:
        atom_type: The type of the link
        outgoing_set: List of atoms that this link connects
        
    Returns:
        A new Link instance
    """
    return Link(atom_type, outgoing_set)


# Integration with CogPrime modules

def register_cognitive_module(atomspace: AtomSpace, module_name: str, 
                             handler: Callable) -> None:
    """Register a cognitive module with the AtomSpace.
    
    This allows cognitive modules to receive notifications about AtomSpace events.
    
    Args:
        atomspace: The AtomSpace to register with
        module_name: The name of the cognitive module
        handler: The handler function for AtomSpace events
    """
    atomspace.register_event_handler('atom_added', handler)
    atomspace.register_event_handler('atom_removed', handler)


def create_cognitive_binding(atomspace: AtomSpace, perception_module, 
                            reasoning_module, action_module) -> None:
    """Create bindings between cognitive modules and the AtomSpace.
    
    This sets up the necessary connections for the CogPrime cognitive cycle.
    
    Args:
        atomspace: The AtomSpace to bind to
        perception_module: The perception module
        reasoning_module: The reasoning module
        action_module: The action module
    """
    # Register modules
    register_cognitive_module(atomspace, 'perception', 
                             lambda atom: perception_module.process_atom(atom))
    register_cognitive_module(atomspace, 'reasoning', 
                             lambda atom: reasoning_module.process_atom(atom))
    register_cognitive_module(atomspace, 'action', 
                             lambda atom: action_module.process_atom(atom))
    
    # TODO: Implement more sophisticated bindings
