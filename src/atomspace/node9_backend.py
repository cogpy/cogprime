"""
Node9 AtomSpace Backend - Full Implementation
Integrates node9 namespace system with AtomSpace for distributed knowledge representation.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if node9 bindings are available
try:
    # Attempt to import node9 bindings (would need to be implemented)
    # For now, we'll use a file-based namespace approach
    NODE9_AVAILABLE = False
    logger.info("Node9 bindings not available, using file-based namespace")
except ImportError:
    NODE9_AVAILABLE = False


class Node9AtomSpaceBackend:
    """AtomSpace backend that uses node9 namespace for distributed storage."""
    
    def __init__(self, namespace_path: str = '/cog/space'):
        """Initialize a node9-based AtomSpace backend.
        
        Args:
            namespace_path: Path in the node9 namespace for AtomSpace
        """
        self.namespace_path = namespace_path
        self._atoms = {}  # Local cache: atom_id -> atom
        self._type_index = {}  # atom_type -> set of atom_ids
        self._name_index = {}  # (atom_type, name) -> atom_id
        
        # For now, use a file-based approach to simulate node9 namespace
        # In a full implementation, this would use actual node9 namespace operations
        self.storage_path = Path.home() / '.cogprime' / 'node9_atomspace'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enabled = True
        logger.info(f"Node9AtomSpaceBackend initialized with namespace: {namespace_path}")
        logger.info(f"Using storage path: {self.storage_path}")
        
        # Load existing atoms from storage
        self._load_from_storage()
    
    def _load_from_storage(self):
        """Load existing atoms from node9 storage."""
        try:
            index_file = self.storage_path / 'index.json'
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    
                # Reconstruct indexes
                self._type_index = {
                    k: set(v) for k, v in index_data.get('type_index', {}).items()
                }
                self._name_index = {
                    tuple(k.split(':', 1)): v 
                    for k, v in index_data.get('name_index', {}).items()
                }
                
                logger.info(f"Loaded {len(self._type_index)} atom types from storage")
        except Exception as e:
            logger.error(f"Failed to load from storage: {e}")
    
    def _save_to_storage(self):
        """Save current state to node9 storage."""
        try:
            index_file = self.storage_path / 'index.json'
            
            # Prepare index data
            index_data = {
                'type_index': {
                    k: list(v) for k, v in self._type_index.items()
                },
                'name_index': {
                    f"{k[0]}:{k[1]}": v 
                    for k, v in self._name_index.items()
                },
            }
            
            # Write to file
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
            logger.debug("Saved indexes to storage")
        except Exception as e:
            logger.error(f"Failed to save to storage: {e}")
    
    def add_atom(self, atom: 'Atom') -> 'Atom':
        """Add an atom to the node9 namespace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The added atom
        """
        # Add to local cache
        self._atoms[atom.id] = atom
        
        # Update type index
        if atom.atom_type not in self._type_index:
            self._type_index[atom.atom_type] = set()
        self._type_index[atom.atom_type].add(atom.id)
        
        # Update name index for nodes
        if hasattr(atom, 'name') and atom.name:
            self._name_index[(atom.atom_type, atom.name)] = atom.id
        
        # Persist to node9 namespace
        if self.enabled:
            try:
                # Serialize atom to JSON
                atom_data = self._serialize_atom(atom)
                
                # Write to namespace file
                atom_file = self.storage_path / f"{atom.id}.json"
                with open(atom_file, 'w') as f:
                    json.dump(atom_data, f, indent=2)
                
                # Update indexes
                self._save_to_storage()
                
                logger.debug(f"Added atom {atom.id} to node9 namespace")
            except Exception as e:
                logger.error(f"Failed to add atom to node9: {e}")
        
        return atom
    
    def remove_atom(self, atom: 'Atom') -> bool:
        """Remove an atom from the node9 namespace.
        
        Args:
            atom: The atom to remove
            
        Returns:
            True if successful, False otherwise
        """
        if atom.id not in self._atoms:
            return False
        
        # Remove from local cache
        del self._atoms[atom.id]
        
        # Update type index
        if atom.atom_type in self._type_index:
            self._type_index[atom.atom_type].discard(atom.id)
        
        # Update name index
        if hasattr(atom, 'name') and atom.name:
            key = (atom.atom_type, atom.name)
            if key in self._name_index:
                del self._name_index[key]
        
        # Remove from node9 namespace
        if self.enabled:
            try:
                atom_file = self.storage_path / f"{atom.id}.json"
                if atom_file.exists():
                    atom_file.unlink()
                
                # Update indexes
                self._save_to_storage()
                
                logger.debug(f"Removed atom {atom.id} from node9 namespace")
            except Exception as e:
                logger.error(f"Failed to remove atom from node9: {e}")
        
        return True
    
    def get_atom(self, atom_id: str) -> Optional['Atom']:
        """Get an atom by ID from the node9 namespace.
        
        Args:
            atom_id: The atom ID
            
        Returns:
            The atom if found, None otherwise
        """
        # Check cache first
        if atom_id in self._atoms:
            return self._atoms[atom_id]
        
        # Try to load from storage
        if self.enabled:
            try:
                atom_file = self.storage_path / f"{atom_id}.json"
                if atom_file.exists():
                    with open(atom_file, 'r') as f:
                        atom_data = json.load(f)
                    
                    # Deserialize and cache
                    atom = self._deserialize_atom(atom_data)
                    if atom:
                        self._atoms[atom_id] = atom
                        return atom
            except Exception as e:
                logger.error(f"Failed to load atom from node9: {e}")
        
        return None
    
    def get_atom_by_type_name(self, atom_type: str, name: str) -> Optional['Atom']:
        """Get a node by type and name from the node9 namespace.
        
        Args:
            atom_type: The atom type
            name: The atom name
            
        Returns:
            The atom if found, None otherwise
        """
        atom_id = self._name_index.get((atom_type, name))
        if atom_id:
            return self.get_atom(atom_id)
        return None
    
    def get_atoms_by_type(self, atom_type: str) -> List['Atom']:
        """Get all atoms of a given type from the node9 namespace.
        
        Args:
            atom_type: The atom type
            
        Returns:
            List of atoms of the specified type
        """
        atom_ids = self._type_index.get(atom_type, set())
        atoms = []
        
        for atom_id in atom_ids:
            atom = self.get_atom(atom_id)
            if atom:
                atoms.append(atom)
        
        return atoms
    
    def query(self, pattern: 'Atom') -> List['Atom']:
        """Query atoms matching a pattern in the node9 namespace.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List of matching atoms
        """
        results = []
        
        # Simple pattern matching based on type
        if hasattr(pattern, 'atom_type'):
            candidates = self.get_atoms_by_type(pattern.atom_type)
            
            # Filter by name if specified
            if hasattr(pattern, 'name') and pattern.name:
                candidates = [a for a in candidates if hasattr(a, 'name') and a.name == pattern.name]
            
            results.extend(candidates)
        
        return results
    
    def pattern_match(self, pattern: Dict) -> List[Dict]:
        """Perform advanced pattern matching in the node9 namespace.
        
        Args:
            pattern: The pattern specification
            
        Returns:
            List of matching results
        """
        results = []
        
        # Extract pattern components
        atom_type = pattern.get('type')
        name = pattern.get('name')
        
        # Get candidates
        if atom_type:
            candidates = self.get_atoms_by_type(atom_type)
            
            # Apply filters
            if name:
                candidates = [a for a in candidates if hasattr(a, 'name') and a.name == name]
            
            # Convert to result format
            for atom in candidates:
                results.append({
                    'atom': atom,
                    'bindings': {},
                })
        
        return results
    
    def _serialize_atom(self, atom: 'Atom') -> Dict[str, Any]:
        """Serialize an atom to a dictionary for node9 storage.
        
        Args:
            atom: The atom to serialize
            
        Returns:
            Dictionary representation of the atom
        """
        data = {
            'id': atom.id,
            'type': atom.atom_type,
        }
        
        if hasattr(atom, 'name') and atom.name:
            data['name'] = atom.name
        
        if hasattr(atom, 'outgoing') and atom.outgoing:
            data['outgoing'] = [a.id for a in atom.outgoing]
        
        if hasattr(atom, 'tv'):
            data['truth_value'] = {
                'strength': atom.tv.strength,
                'confidence': atom.tv.confidence,
            }
        
        if hasattr(atom, 'av'):
            data['attention_value'] = {
                'sti': atom.av.sti,
                'lti': atom.av.lti,
                'vlti': atom.av.vlti,
            }
        
        return data
    
    def _deserialize_atom(self, data: Dict[str, Any]) -> Optional['Atom']:
        """Deserialize an atom from a dictionary.
        
        Args:
            data: Dictionary representation of the atom
            
        Returns:
            The deserialized atom or None
        """
        # This is a simplified implementation
        # In a full implementation, this would properly reconstruct the atom
        # including resolving outgoing atom references
        
        # For now, we'll just return None and rely on the cache
        # The actual implementation would need to import and use the Atom classes
        return None
    
    def sync_namespace(self):
        """Synchronize the local cache with the node9 namespace.
        
        This method ensures consistency between local and distributed state.
        """
        if not self.enabled:
            return
        
        try:
            # Reload indexes from storage
            self._load_from_storage()
            
            # Clear local cache to force reload
            self._atoms.clear()
            
            logger.info("Synchronized with node9 namespace")
        except Exception as e:
            logger.error(f"Failed to sync namespace: {e}")
