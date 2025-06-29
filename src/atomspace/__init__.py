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
        """
        Initialize a TruthValue instance with specified strength and confidence, clamped between 0.0 and 1.0.
        
        Parameters:
            strength (float): Degree of truth, where 0.0 is completely false and 1.0 is completely true.
            confidence (float): Certainty of the truth value, from 0.0 (no confidence) to 1.0 (full confidence).
        """
        self.strength = max(0.0, min(1.0, strength))
        self.confidence = max(0.0, min(1.0, confidence))
    
    def __repr__(self) -> str:
        """
        Return a string representation of the TruthValue with formatted strength and confidence.
        """
        return f"TruthValue(strength={self.strength:.3f}, confidence={self.confidence:.3f})"


class AttentionValue:
    """Attention value representation for atoms."""
    
    def __init__(self, sti: float = 0.0, lti: float = 0.0, vlti: bool = False):
        """
        Initialize an AttentionValue instance with specified short-term, long-term, and very long-term importance values.
        
        Parameters:
        	sti (float): Short-term importance value.
        	lti (float): Long-term importance value.
        	vlti (bool): Indicates if the atom has very long-term importance.
        """
        self.sti = sti  # Short-Term Importance
        self.lti = lti  # Long-Term Importance
        self.vlti = vlti  # Very Long-Term Importance flag
    
    def __repr__(self) -> str:
        """
        Return a string representation of the AttentionValue, showing short-term, long-term, and very long-term importance values.
        """
        return f"AttentionValue(sti={self.sti:.3f}, lti={self.lti:.3f}, vlti={self.vlti})"


class Atom(ABC):
    """Base class for all atoms in the AtomSpace."""
    
    def __init__(self, atom_type: AtomType, name: str = None):
        """
        Create a new atom with the specified type and optional name, assigning a unique identifier and initializing truth and attention values.
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
        """
        Determine whether the atom is a node.
        
        Returns:
            bool: True if the atom is a node, False otherwise.
        """
        pass
    
    @abstractmethod
    def is_link(self) -> bool:
        """
        Indicates whether the atom is a link.
        
        Returns:
            bool: True if the atom is a link; False otherwise.
        """
        pass
    
    @abstractmethod
    def get_hash(self) -> int:
        """
        Return a hash value uniquely identifying this atom.
        
        Intended to be implemented by subclasses to provide a consistent hash based on atom properties.
        """
        pass
    
    def get_value(self, key: str) -> Any:
        """
        Retrieve the value associated with the specified key from the atom's key-value store.
        
        Parameters:
            key (str): The key for which to retrieve the value.
        
        Returns:
            Any: The value associated with the key, or None if the key does not exist.
        """
        return self.values.get(key)
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Assigns a value to the specified key in the atom's key-value store.
        
        Parameters:
            key (str): The key under which the value will be stored.
            value (Any): The value to associate with the key.
        """
        self.values[key] = value
    
    def set_truth_value(self, tv: TruthValue) -> None:
        """
        Assigns a new truth value to this atom.
        
        Parameters:
        	tv (TruthValue): The truth value to set for the atom.
        """
        self.tv = tv
    
    def get_truth_value(self) -> TruthValue:
        """
        Returns the truth value associated with this atom.
        
        Returns:
            TruthValue: The strength and confidence values representing the atom's truth assessment.
        """
        return self.tv
    
    def set_attention_value(self, av: AttentionValue) -> None:
        """
        Assigns a new attention value to this atom.
        
        Parameters:
        	av (AttentionValue): The attention value to set for the atom.
        """
        self.av = av
    
    def get_attention_value(self) -> AttentionValue:
        """
        Returns the attention value associated with this atom.
        
        Returns:
            AttentionValue: The attention value object containing short-term, long-term, and very long-term importance metrics.
        """
        return self.av
    
    def add_to_incoming_set(self, link) -> None:
        """
        Adds the specified link to the set of incoming links for this atom.
        
        Parameters:
            link (Link): The link to be added as an incoming connection.
        """
        self.incoming_set.add(link)
    
    def remove_from_incoming_set(self, link) -> None:
        """
        Removes the specified link from the set of incoming links to this atom.
        """
        self.incoming_set.discard(link)
    
    def get_incoming_set(self) -> Set:
        """
        Return the set of links in which this atom appears as an outgoing atom.
        
        Returns:
            Set: A set of Link instances referencing this atom.
        """
        return self.incoming_set
    
    def __eq__(self, other) -> bool:
        """
        Determine whether this atom is equal to another atom based on their unique IDs.
        
        Returns:
            bool: True if the other object is an Atom with the same ID, otherwise False.
        """
        if not isinstance(other, Atom):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """
        Returns the hash value of the atom based on its unique identifier.
        """
        return hash(self.id)


class Node(Atom):
    """Node class representing vertices in the AtomSpace hypergraph."""
    
    def __init__(self, atom_type: AtomType, name: str):
        """
        Initialize a Node with the specified type and name.
        
        Raises:
            ValueError: If the name is None.
        """
        super().__init__(atom_type, name)
        if name is None:
            raise ValueError("Node must have a name")
    
    def is_node(self) -> bool:
        """
        Indicates that this atom is a node.
        
        Returns:
            bool: True, since this object represents a node.
        """
        return True
    
    def is_link(self) -> bool:
        """
        Indicates that this atom is not a link.
        
        Returns:
            bool: Always returns False.
        """
        return False
    
    def get_hash(self) -> int:
        """
        Return a hash value for the node based on its type and name.
        """
        return hash((self.atom_type, self.name))
    
    def __repr__(self) -> str:
        """
        Return a string representation of the node showing its type and name.
        """
        return f"{self.atom_type}('{self.name}')"


class Link(Atom):
    """Link class representing hyperedges in the AtomSpace hypergraph."""
    
    def __init__(self, atom_type: AtomType, outgoing_set: List[Atom]):
        """
        Initialize a Link atom with a specified type and a list of connected atoms.
        
        Parameters:
        	atom_type (AtomType): The type of the link atom.
        	outgoing_set (List[Atom]): Atoms that this link connects as its outgoing set.
        """
        super().__init__(atom_type)
        self.outgoing_set = outgoing_set if outgoing_set else []
    
    def is_node(self) -> bool:
        """
        Indicates that this atom is not a node.
        
        Returns:
            bool: Always returns False.
        """
        return False
    
    def is_link(self) -> bool:
        """
        Indicates that this atom is a link.
        
        Returns:
            bool: Always True, as this object represents a link.
        """
        return True
    
    def get_hash(self) -> int:
        """
        Return a hash value based on the link's type and the IDs of its outgoing atoms.
        """
        return hash((self.atom_type, tuple(atom.id for atom in self.outgoing_set)))
    
    def get_arity(self) -> int:
        """
        Return the number of atoms connected by this link.
        
        Returns:
            int: The arity, representing how many atoms are in the outgoing set of the link.
        """
        return len(self.outgoing_set)
    
    def get_outgoing_set(self) -> List[Atom]:
        """
        Return the list of atoms connected by this link.
        
        Returns:
            List[Atom]: The outgoing set of atoms associated with this link.
        """
        return self.outgoing_set
    
    def __repr__(self) -> str:
        """
        Return a string representation of the link, showing its type and connected atoms.
        """
        outgoing_repr = ", ".join(repr(atom) for atom in self.outgoing_set)
        return f"{self.atom_type}({outgoing_repr})"


class AtomSpaceBackend(ABC):
    """Abstract base class for AtomSpace backends."""
    
    @abstractmethod
    def add_atom(self, atom: Atom) -> Atom:
        """
        Adds an atom to the backend storage and updates relevant indexes.
        
        Returns:
            Atom: The atom that was added.
        """
        pass
    
    @abstractmethod
    def remove_atom(self, atom: Atom) -> bool:
        """
        Removes the specified atom from the backend storage.
        
        Returns:
            bool: True if the atom was successfully removed, False if the atom was not found.
        """
        pass
    
    @abstractmethod
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """
        Retrieve an atom from the AtomSpace by its unique identifier.
        
        Parameters:
        	atom_id: The unique identifier of the atom to retrieve.
        
        Returns:
        	The Atom instance with the specified ID, or None if not found.
        """
        pass
    
    @abstractmethod
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """
        Retrieve a node from the AtomSpace by its type and name.
        
        Parameters:
        	atom_type (AtomType): The type of the node to retrieve.
        	name (str): The name of the node to retrieve.
        
        Returns:
        	Node or None: The node matching the specified type and name, or None if not found.
        """
        pass
    
    @abstractmethod
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """
        Return a list of all atoms with the specified type.
        
        Parameters:
        	atom_type (AtomType): The type of atoms to retrieve.
        
        Returns:
        	List[Atom]: Atoms matching the given type.
        """
        pass
    
    @abstractmethod
    def query(self, pattern: Atom) -> List[Atom]:
        """
        Return a list of atoms that match the given pattern atom.
        
        Parameters:
            pattern (Atom): The atom pattern to match against atoms in the AtomSpace.
        
        Returns:
            List[Atom]: Atoms matching the specified pattern.
        """
        pass
    
    @abstractmethod
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """
        Performs advanced pattern matching against atoms in the AtomSpace.
        
        Parameters:
            pattern (Dict): A dictionary specifying the pattern to match.
        
        Returns:
            List[Dict]: A list of match results, each represented as a dictionary.
        """
        pass


class LocalAtomSpaceBackend(AtomSpaceBackend):
    """Local in-memory implementation of the AtomSpace backend."""
    
    def __init__(self):
        """
        Initialize the in-memory data structures for storing and indexing atoms in the local AtomSpace backend.
        """
        self.atoms_by_id = {}  # id -> atom
        self.nodes_by_type_name = {}  # (type, name) -> node
        self.atoms_by_type = {}  # type -> set of atoms
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Adds an atom to the local in-memory backend, updating all relevant indexes and incoming link sets.
        
        Returns:
            The atom that was added.
        """
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
        """
        Removes the specified atom from local storage and all associated indexes.
        
        Returns:
            bool: True if the atom was removed, False if it was not found.
        """
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
        """
        Retrieve an atom from the backend by its unique identifier.
        
        Parameters:
        	atom_id: The unique identifier of the atom to retrieve.
        
        Returns:
        	The Atom instance with the specified ID, or None if not found.
        """
        return self.atoms_by_id.get(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """
        Retrieve a node from the AtomSpace by its type and name.
        
        Parameters:
            atom_type (AtomType): The type of the node to retrieve.
            name (str): The name of the node to retrieve.
        
        Returns:
            Optional[Node]: The node matching the given type and name, or None if not found.
        """
        key = (atom_type, name)
        return self.nodes_by_type_name.get(key)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """
        Return a list of all atoms with the specified type.
        
        Parameters:
            atom_type (AtomType): The type of atoms to retrieve.
        
        Returns:
            List[Atom]: Atoms matching the given type.
        """
        return list(self.atoms_by_type.get(atom_type, set()))
    
    def query(self, pattern: Atom) -> List[Atom]:
        """
        Return a list of atoms that match the given pattern atom.
        
        If the pattern is a node, returns the node with the same type and name.  
        If the pattern is a link, returns all links of the same type whose outgoing set matches the pattern's outgoing set, supporting None as a wildcard.  
        Recursive matching for nested links is not implemented.
         
        Parameters:
            pattern (Atom): The atom pattern to match against.
        
        Returns:
            List[Atom]: Atoms matching the pattern.
        """
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
        """
        Performs advanced pattern matching against atoms in the AtomSpace.
        
        Parameters:
            pattern (Dict): A dictionary specifying the pattern to match.
        
        Returns:
            List[Dict]: A list of match results, currently always empty as this is a placeholder.
        """
        # TODO: Implement more sophisticated pattern matching
        # This is a placeholder for advanced pattern matching functionality
        return []


class Node9AtomSpaceBackend(AtomSpaceBackend):
    """AtomSpace backend that uses node9 namespace for distributed operation."""
    
    def __init__(self, namespace_path: str = "/cog/space"):
        """
        Initialize a Node9AtomSpaceBackend for distributed AtomSpace operations using a specified node9 namespace path.
        
        Currently, this backend is not fully implemented and falls back to a local in-memory backend.
        """
        self.namespace_path = namespace_path
        # TODO: Implement node9 namespace integration
        logger.warning("Node9AtomSpaceBackend is not fully implemented yet")
        
        # Temporary fallback to local backend
        self._local_backend = LocalAtomSpaceBackend()
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Adds an atom to the AtomSpace using the node9 backend.
        
        Currently, this method delegates to the local backend as node9 integration is not implemented.
        
        Returns:
            The atom that was added.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.add_atom(atom)
    
    def remove_atom(self, atom: Atom) -> bool:
        """
        Removes an atom from the AtomSpace using the node9 backend.
        
        Returns:
            bool: True if the atom was removed successfully, False otherwise.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.remove_atom(atom)
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """
        Retrieve an atom by its unique ID, using the local backend as a fallback.
        
        Returns:
            The Atom instance with the specified ID, or None if not found.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atom(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """
        Retrieve a node by its type and name using the local backend as a fallback.
        
        Returns:
            The Node instance matching the specified type and name, or None if not found.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """
        Return all atoms of the specified type from the AtomSpace.
        
        Parameters:
        	atom_type (AtomType): The type of atoms to retrieve.
        
        Returns:
        	List[Atom]: A list of atoms matching the given type.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """
        Return atoms matching the given pattern using the local backend.
        
        Parameters:
        	pattern (Atom): The atom pattern to match against existing atoms.
        
        Returns:
        	List[Atom]: Atoms that match the specified pattern.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """
        Performs advanced pattern matching using the local backend as a fallback.
        
        Parameters:
            pattern (Dict): A dictionary representing the pattern to match against atoms.
        
        Returns:
            List[Dict]: A list of match results, each represented as a dictionary.
        """
        # TODO: Implement node9 namespace integration
        return self._local_backend.pattern_match(pattern)


class Mem0AtomSpaceBackend(AtomSpaceBackend):
    """AtomSpace backend that uses mem0 for persistence and vector search."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Mem0AtomSpaceBackend with optional configuration for mem0 integration.
        
        Currently, this backend is not fully implemented and falls back to using a local in-memory backend.
        """
        self.config = config or {}
        # TODO: Implement mem0 integration
        logger.warning("Mem0AtomSpaceBackend is not fully implemented yet")
        
        # Temporary fallback to local backend
        self._local_backend = LocalAtomSpaceBackend()
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Adds an atom to the mem0 backend storage.
        
        Currently, this method delegates to the local backend as mem0 integration is not yet implemented.
        
        Returns:
            The atom that was added.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.add_atom(atom)
    
    def remove_atom(self, atom: Atom) -> bool:
        """
        Removes an atom from mem0 storage, delegating to the local backend.
        
        Returns:
            bool: True if the atom was removed, False otherwise.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.remove_atom(atom)
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """
        Retrieve an atom by its unique ID from the backend.
        
        Returns:
            The Atom instance with the specified ID, or None if not found.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.get_atom(atom_id)
    
    def get_atom_by_type_name(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """
        Retrieve a node from mem0 storage by its type and name.
        
        Returns:
            The Node instance matching the specified type and name, or None if not found.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """
        Retrieve all atoms of the specified type from the backend.
        
        Returns:
            List[Atom]: A list of atoms matching the given type.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """
        Query atoms matching a given pattern using the local backend.
        
        Parameters:
        	pattern (Atom): The atom pattern to match against stored atoms.
        
        Returns:
        	List[Atom]: A list of atoms that match the specified pattern.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """
        Performs advanced pattern matching using the mem0 backend.
        
        Parameters:
            pattern (Dict): A dictionary representing the pattern to match against atoms.
        
        Returns:
            List[Dict]: A list of dictionaries representing matched patterns.
        """
        # TODO: Implement mem0 integration
        return self._local_backend.pattern_match(pattern)
    
    def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple[Atom, float]]:
        """
        Performs a vector similarity search for atoms using the mem0 backend.
        
        Parameters:
            vector (List[float]): The query vector for similarity search.
            limit (int): The maximum number of results to return.
        
        Returns:
            List[Tuple[Atom, float]]: A list of tuples containing atoms and their similarity scores.
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
        """
                 Initialize an AtomSpace instance with the specified backend type and configuration.
                 
                 Parameters:
                     backend_type (BackendType or str): The backend to use for atom storage and operations. Supported values are 'local', 'node9', 'mem0', or 'distributed'.
                     config (dict, optional): Configuration options for the selected backend.
                 
                 Raises:
                     ValueError: If an unknown backend type is provided.
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
        """
        Adds an atom to the AtomSpace and triggers the 'atom_added' event.
        
        If the atom already exists in the backend, the existing instance may be returned.
        
        Returns:
            Atom: The atom instance stored in the AtomSpace.
        """
        # Set the atomspace reference
        atom.atomspace = weakref.ref(self)
        
        # Add to backend
        result = self.backend.add_atom(atom)
        
        # Trigger events
        self._trigger_event('atom_added', result)
        
        return result
    
    def remove(self, atom: Atom) -> bool:
        """
        Removes the specified atom from the AtomSpace.
        
        Returns:
            bool: True if the atom was successfully removed; False if the atom was not present.
        """
        result = self.backend.remove_atom(atom)
        
        if result:
            # Clear the atomspace reference
            atom.atomspace = None
            
            # Trigger events
            self._trigger_event('atom_removed', atom)
        
        return result
    
    def get_atom(self, atom_id: AtomID) -> Optional[Atom]:
        """
        Retrieve an atom from the AtomSpace by its unique identifier.
        
        Parameters:
        	atom_id: The unique identifier of the atom to retrieve.
        
        Returns:
        	The corresponding Atom instance if found, otherwise None.
        """
        return self.backend.get_atom(atom_id)
    
    def get_node(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """
        Retrieve a node from the AtomSpace by its type and name.
        
        Parameters:
        	atom_type (AtomType): The type of the node to retrieve.
        	name (str): The name of the node to retrieve.
        
        Returns:
        	Node or None: The matching node if found, otherwise None.
        """
        return self.backend.get_atom_by_type_name(atom_type, name)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """
        Return all atoms of the specified type from the AtomSpace.
        
        Parameters:
            atom_type (AtomType): The type of atoms to retrieve.
        
        Returns:
            List[Atom]: Atoms matching the given type.
        """
        return self.backend.get_atoms_by_type(atom_type)
    
    def query(self, pattern: Atom) -> List[Atom]:
        """
        Return a list of atoms that match the given pattern, supporting wildcards.
        
        Parameters:
            pattern (Atom): An atom pattern to match against, where fields may be set to None to act as wildcards.
        
        Returns:
            List[Atom]: Atoms from the AtomSpace that match the specified pattern.
        """
        return self.backend.query(pattern)
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """
        Perform advanced pattern matching over the AtomSpace using the specified pattern.
        
        Parameters:
            pattern (dict): A dictionary describing the structure and constraints of the pattern to match.
        
        Returns:
            List[dict]: A list of dictionaries, each representing a successful match of the pattern.
        """
        return self.backend.pattern_match(pattern)
    
    def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple[Atom, float]]:
        """
        Performs a vector similarity search for atoms using the mem0 backend.
        
        Parameters:
            vector (List[float]): The query vector for similarity search.
            limit (int): The maximum number of results to return.
        
        Returns:
            List[Tuple[Atom, float]]: A list of tuples containing atoms and their similarity scores. Returns an empty list if the mem0 backend is not used.
        """
        if isinstance(self.backend, Mem0AtomSpaceBackend):
            return self.backend.vector_search(vector, limit)
        else:
            logger.warning("Vector search requires mem0 backend")
            return []
    
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """
        Registers a handler function to be called when the specified event is triggered.
        
        Parameters:
        	event_name (str): The name of the event to listen for.
        	handler (Callable): The function to be called when the event occurs.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
    
    def unregister_event_handler(self, event_name: str, handler: Callable) -> bool:
        """
        Remove a previously registered handler from the specified event.
        
        Parameters:
        	event_name (str): The name of the event.
        	handler (Callable): The handler function to remove.
        
        Returns:
        	bool: True if the handler was successfully removed; False if it was not registered.
        """
        if event_name in self._event_handlers:
            if handler in self._event_handlers[event_name]:
                self._event_handlers[event_name].remove(handler)
                return True
        return False
    
    def _trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """
        Invokes all handlers registered for a given event name, passing any provided arguments.
        
        If a handler raises an exception, the error is logged and event processing continues for remaining handlers.
        """
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")


# Factory functions for creating atoms

def create_node(atom_type: AtomType, name: str) -> Node:
    """
    Create and return a new Node atom with the specified type and name.
    
    Parameters:
        atom_type (AtomType): The type of the node to create.
        name (str): The name assigned to the node.
    
    Returns:
        Node: The newly created Node instance.
    """
    return Node(atom_type, name)


def create_link(atom_type: AtomType, outgoing_set: List[Atom]) -> Link:
    """
    Create and return a new Link atom connecting the specified set of atoms.
    
    Parameters:
        atom_type (AtomType): The type of the link to create.
        outgoing_set (List[Atom]): Atoms that the link will connect.
    
    Returns:
        Link: The newly created Link instance.
    """
    return Link(atom_type, outgoing_set)


# Integration with CogPrime modules

def register_cognitive_module(atomspace: AtomSpace, module_name: str, 
                             handler: Callable) -> None:
    """
                             Registers a cognitive module's handler to receive notifications for atom addition and removal events in the AtomSpace.
                             
                             Parameters:
                                 module_name (str): Name of the cognitive module.
                                 handler (Callable): Function to be called when atoms are added or removed.
                             """
    atomspace.register_event_handler('atom_added', handler)
    atomspace.register_event_handler('atom_removed', handler)


def create_cognitive_binding(atomspace: AtomSpace, perception_module, 
                            reasoning_module, action_module) -> None:
    """
                            Bind perception, reasoning, and action modules to the AtomSpace, enabling them to process atoms on relevant events.
                            
                            This function registers each module's `process_atom` method as an event handler in the AtomSpace, facilitating integration of cognitive cycles.
                            """
    # Register modules
    register_cognitive_module(atomspace, 'perception', 
                             lambda atom: perception_module.process_atom(atom))
    register_cognitive_module(atomspace, 'reasoning', 
                             lambda atom: reasoning_module.process_atom(atom))
    register_cognitive_module(atomspace, 'action', 
                             lambda atom: action_module.process_atom(atom))
    
    # TODO: Implement more sophisticated bindings
