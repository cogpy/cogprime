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
    
    def _match_atoms_recursive(self, pattern: Atom, candidate: Atom) -> bool:
        """Recursively match atoms including nested links.
        
        Args:
            pattern: The pattern atom to match
            candidate: The candidate atom to check
            
        Returns:
            True if atoms match, False otherwise
        """
        # Check if both are nodes
        if pattern.is_node() and candidate.is_node():
            return (pattern.atom_type == candidate.atom_type and 
                    pattern.name == candidate.name)
        
        # Check if both are links
        if pattern.is_link() and candidate.is_link():
            # Type must match
            if pattern.atom_type != candidate.atom_type:
                return False
            
            # Arity must match
            if len(pattern.outgoing_set) != len(candidate.outgoing_set):
                return False
            
            # Recursively check all outgoing atoms
            for pattern_atom, candidate_atom in zip(pattern.outgoing_set, candidate.outgoing_set):
                # None is a wildcard
                if pattern_atom is None:
                    continue
                
                # Recursively match
                if not self._match_atoms_recursive(pattern_atom, candidate_atom):
                    return False
            
            return True
        
        # Different types don't match
        return False
    
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
                        # Recursively match nested link outgoing sets
                        nested_matches = self._match_atoms_recursive(pattern_atom, candidate_atom)
                        if not nested_matches:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    results.append(candidate)
        
        return results
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching using dictionary patterns.
        
        Args:
            pattern: Dictionary pattern with keys:
                - 'type': Atom type to match (optional)
                - 'name': Node name to match (optional)
                - 'outgoing': List of outgoing atom patterns (for links)
                - 'variables': Dict of variable names to bind
                
        Returns:
            List of dictionaries containing matched atoms and variable bindings
        """
        results = []
        
        # Extract pattern components
        pattern_type = pattern.get('type')
        pattern_name = pattern.get('name')
        pattern_outgoing = pattern.get('outgoing', [])
        variables = pattern.get('variables', {})
        
        # If no type specified, search all atoms
        if pattern_type:
            candidates = self.get_atoms_by_type(pattern_type)
        else:
            candidates = list(self.atoms_by_id.values())
        
        # Match each candidate
        for candidate in candidates:
            bindings = {}
            
            # Check type match
            if pattern_type and candidate.atom_type != pattern_type:
                continue
            
            # Check name match for nodes
            if pattern_name and candidate.is_node():
                if candidate.name != pattern_name:
                    continue
            
            # Check outgoing set for links
            if pattern_outgoing and candidate.is_link():
                if len(pattern_outgoing) != len(candidate.outgoing_set):
                    continue
                
                # Match each outgoing atom
                match = True
                for i, out_pattern in enumerate(pattern_outgoing):
                    out_candidate = candidate.outgoing_set[i]
                    
                    # Variable binding
                    if isinstance(out_pattern, str) and out_pattern.startswith('$'):
                        bindings[out_pattern] = out_candidate
                    # Recursive pattern matching
                    elif isinstance(out_pattern, dict):
                        sub_results = self.pattern_match(out_pattern)
                        if not sub_results:
                            match = False
                            break
                        bindings.update(sub_results[0])
                    # Direct atom comparison
                    elif isinstance(out_pattern, Atom):
                        if not self._match_atoms_recursive(out_pattern, out_candidate):
                            match = False
                            break
                
                if not match:
                    continue
            
            # Add matched result
            result = {
                'atom': candidate,
                'bindings': bindings
            }
            results.append(result)
        
        return results


# Import the full Node9 backend implementation
try:
    from .node9_backend import Node9AtomSpaceBackend
except ImportError:
    # Fallback implementation if node9_backend is not available
    class Node9AtomSpaceBackend(AtomSpaceBackend):
        """AtomSpace backend that uses node9 namespace for distributed storage."""
        
        def __init__(self, namespace_path: str = '/cog/space'):
            self.namespace_path = namespace_path
            logger.warning("Node9AtomSpaceBackend module not found, using local backend")
            self._local_backend = LocalAtomSpaceBackend()
        
        def add_atom(self, atom: Atom) -> Atom:
            return self._local_backend.add_atom(atom)
        
        def remove_atom(self, atom: Atom) -> bool:
            return self._local_backend.remove_atom(atom)
        
        def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
            return self._local_backend.get_atom(atom_id)
        
        def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
            return self._local_backend.get_atom_by_type_name(atom_type, name)
        
        def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
            return self._local_backend.get_atoms_by_type(atom_type)
        
        def query(self, pattern: Atom) -> List[Atom]:
            return self._local_backend.query(pattern)
        
        def pattern_match(self, pattern: Dict) -> List[Dict]:
            return self._local_backend.pattern_match(pattern)


# Import the full Mem0 backend implementation
try:
    from .mem0_backend import Mem0AtomSpaceBackend
except ImportError:
    # Fallback implementation if mem0_backend is not available
    class Mem0AtomSpaceBackend(AtomSpaceBackend):
        """AtomSpace backend that uses mem0 for persistence and vector search."""
        
        def __init__(self, config: Dict = None):
            self.config = config or {}
            logger.warning("Mem0AtomSpaceBackend module not found, using local backend")
            self._local_backend = LocalAtomSpaceBackend()
        
        def add_atom(self, atom: Atom) -> Atom:
            return self._local_backend.add_atom(atom)
        
        def remove_atom(self, atom: Atom) -> bool:
            return self._local_backend.remove_atom(atom)
        
        def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
            return self._local_backend.get_atom(atom_id)
        
        def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
            return self._local_backend.get_atom_by_type_name(atom_type, name)
        
        def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
            return self._local_backend.get_atoms_by_type(atom_type)
        
        def query(self, pattern: Atom) -> List[Atom]:
            return self._local_backend.query(pattern)
        
        def pattern_match(self, pattern: Dict) -> List[Dict]:
            return self._local_backend.pattern_match(pattern)
        
        def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple[Atom, float]]:
            return []


class DistributedAtomSpaceBackend(AtomSpaceBackend):
    """Distributed AtomSpace backend combining node9 and mem0.
    
    This backend uses:
    - node9 for namespace management and graph structure
    - mem0 for persistence, vector search, and semantic queries
    """
    
    def __init__(self, node9_config: Dict = None, mem0_config: Dict = None):
        """Initialize distributed backend with both node9 and mem0.
        
        Args:
            node9_config: Configuration for node9 backend
            mem0_config: Configuration for mem0 backend
        """
        self.node9_config = node9_config or {}
        self.mem0_config = mem0_config or {}
        
        # Initialize both backends
        try:
            self.node9_backend = Node9AtomSpaceBackend(**self.node9_config)
        except Exception as e:
            logger.warning(f"Failed to initialize node9 backend: {e}, using local")
            self.node9_backend = LocalAtomSpaceBackend()
        
        try:
            self.mem0_backend = Mem0AtomSpaceBackend(config=self.mem0_config)
        except Exception as e:
            logger.warning(f"Failed to initialize mem0 backend: {e}, using local")
            self.mem0_backend = LocalAtomSpaceBackend()
        
        # Use node9 as primary for structure, mem0 for search
        self.primary_backend = self.node9_backend
        self.search_backend = self.mem0_backend
        
        logger.info("Initialized distributed backend with node9 and mem0")
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add atom to both backends."""
        # Add to node9 for structure
        result = self.primary_backend.add_atom(atom)
        
        # Also add to mem0 for search
        try:
            self.search_backend.add_atom(atom)
        except Exception as e:
            logger.warning(f"Failed to add atom to mem0: {e}")
        
        return result
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove atom from both backends."""
        success = self.primary_backend.remove_atom(atom)
        
        try:
            self.search_backend.remove_atom(atom)
        except Exception as e:
            logger.warning(f"Failed to remove atom from mem0: {e}")
        
        return success
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """Get atom from primary backend."""
        return self.primary_backend.get_atom(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get atom by type and name from primary backend."""
        return self.primary_backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a type from primary backend."""
        return self.primary_backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """Query using primary backend."""
        return self.primary_backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Pattern match using primary backend."""
        return self.primary_backend.pattern_match(pattern)
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[Atom, float]]:
        """Perform semantic search using mem0 backend.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of (atom, score) tuples
        """
        try:
            # Use mem0 for semantic search
            if hasattr(self.search_backend, 'semantic_search'):
                return self.search_backend.semantic_search(query, limit)
            else:
                logger.warning("Semantic search not available, using pattern match")
                return []
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
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
            # Implement a combined backend that uses both node9 and mem0
            # node9 handles the namespace and graph structure
            # mem0 handles persistence and semantic search
            try:
                self.backend = DistributedAtomSpaceBackend(
                    node9_config={
                        'namespace_path': self.config.get('namespace_path', '/cog/space')
                    },
                    mem0_config=self.config.get('mem0', {})
                )
                logger.info("Initialized distributed backend with node9 and mem0")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed backend: {e}, using local backend")
                self.backend = LocalAtomSpaceBackend()
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
