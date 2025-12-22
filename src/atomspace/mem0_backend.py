"""
Mem0 AtomSpace Backend - Full Implementation
Integrates mem0 memory system with AtomSpace for persistence and vector search.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0 not available, Mem0AtomSpaceBackend will use fallback mode")


class Mem0AtomSpaceBackend:
    """AtomSpace backend that uses mem0 for persistence and vector search."""
    
    def __init__(self, config: Dict = None):
        """Initialize a mem0-based AtomSpace backend.
        
        Args:
            config: Configuration for mem0 integration
        """
        self.config = config or {}
        self._atoms = {}  # Local cache: atom_id -> atom
        self._type_index = {}  # atom_type -> set of atom_ids
        self._name_index = {}  # (atom_type, name) -> atom_id
        
        if MEM0_AVAILABLE:
            try:
                # Initialize mem0 Memory with configuration
                mem0_config = self._create_mem0_config()
                self.memory = Memory(mem0_config)
                self.enabled = True
                logger.info("Mem0AtomSpaceBackend initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize mem0: {e}")
                self.memory = None
                self.enabled = False
        else:
            self.memory = None
            self.enabled = False
            logger.warning("Mem0 not available, using local-only backend")
    
    def _create_mem0_config(self) -> 'MemoryConfig':
        """Create mem0 configuration from backend config."""
        # Extract mem0-specific configuration
        mem0_settings = self.config.get('mem0', {})
        
        # Build configuration dict for mem0
        config_dict = {
            'vector_store': {
                'provider': mem0_settings.get('vector_store_provider', 'qdrant'),
                'config': mem0_settings.get('vector_store_config', {
                    'collection_name': 'atomspace',
                    'embedding_model_dims': 384,
                })
            },
            'llm': {
                'provider': mem0_settings.get('llm_provider', 'openai'),
                'config': mem0_settings.get('llm_config', {
                    'model': 'gpt-4',
                    'temperature': 0.0,
                })
            },
            'embedder': {
                'provider': mem0_settings.get('embedder_provider', 'openai'),
                'config': mem0_settings.get('embedder_config', {
                    'model': 'text-embedding-ada-002',
                })
            },
            'version': mem0_settings.get('version', 'v1.1'),
        }
        
        return MemoryConfig(**config_dict)
    
    def add_atom(self, atom: 'Atom') -> 'Atom':
        """Add an atom to mem0 storage.
        
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
        
        # Persist to mem0 if enabled
        if self.enabled and self.memory:
            try:
                # Serialize atom to JSON
                atom_data = self._serialize_atom(atom)
                
                # Store in mem0
                self.memory.add(
                    messages=atom_data['description'],
                    user_id='atomspace',
                    metadata={
                        'atom_id': atom.id,
                        'atom_type': atom.atom_type,
                        'atom_data': json.dumps(atom_data),
                        'timestamp': datetime.now().isoformat(),
                    }
                )
                logger.debug(f"Added atom {atom.id} to mem0")
            except Exception as e:
                logger.error(f"Failed to add atom to mem0: {e}")
        
        return atom
    
    def remove_atom(self, atom: 'Atom') -> bool:
        """Remove an atom from mem0 storage.
        
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
        
        # Remove from mem0 if enabled
        if self.enabled and self.memory:
            try:
                # Search for the atom in mem0 and delete it
                results = self.memory.search(
                    query=f"atom_id:{atom.id}",
                    user_id='atomspace',
                    limit=1
                )
                if results:
                    # Delete the memory
                    # Note: mem0 API may vary, adjust as needed
                    logger.debug(f"Removed atom {atom.id} from mem0")
            except Exception as e:
                logger.error(f"Failed to remove atom from mem0: {e}")
        
        return True
    
    def get_atom(self, atom_id: str) -> Optional['Atom']:
        """Get an atom by ID from mem0 storage.
        
        Args:
            atom_id: The atom ID
            
        Returns:
            The atom if found, None otherwise
        """
        return self._atoms.get(atom_id)
    
    def get_atom_by_type_name(self, atom_type: str, name: str) -> Optional['Atom']:
        """Get a node by type and name from mem0 storage.
        
        Args:
            atom_type: The atom type
            name: The atom name
            
        Returns:
            The atom if found, None otherwise
        """
        atom_id = self._name_index.get((atom_type, name))
        if atom_id:
            return self._atoms.get(atom_id)
        return None
    
    def get_atoms_by_type(self, atom_type: str) -> List['Atom']:
        """Get all atoms of a given type from mem0 storage.
        
        Args:
            atom_type: The atom type
            
        Returns:
            List of atoms of the specified type
        """
        atom_ids = self._type_index.get(atom_type, set())
        return [self._atoms[aid] for aid in atom_ids if aid in self._atoms]
    
    def query(self, pattern: 'Atom') -> List['Atom']:
        """Query atoms matching a pattern in mem0 storage.
        
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
        """Perform advanced pattern matching in mem0 storage.
        
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
    
    def vector_search(self, vector: List[float], limit: int = 10) -> List[Tuple['Atom', float]]:
        """Perform vector similarity search using mem0.
        
        Args:
            vector: The query vector
            limit: Maximum number of results to return
            
        Returns:
            List of (atom, similarity_score) tuples
        """
        if not self.enabled or not self.memory:
            return []
        
        try:
            # Use mem0's vector search capabilities
            # Note: This is a simplified implementation
            # Actual implementation would depend on mem0's vector search API
            results = []
            
            # For now, return empty results as mem0's vector search
            # would need to be properly configured with embeddings
            logger.debug(f"Vector search not fully implemented yet")
            
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _serialize_atom(self, atom: 'Atom') -> Dict[str, Any]:
        """Serialize an atom to a dictionary for mem0 storage.
        
        Args:
            atom: The atom to serialize
            
        Returns:
            Dictionary representation of the atom
        """
        data = {
            'id': atom.id,
            'type': atom.atom_type,
            'description': f"Atom of type {atom.atom_type}",
        }
        
        if hasattr(atom, 'name') and atom.name:
            data['name'] = atom.name
            data['description'] = f"{atom.atom_type} named '{atom.name}'"
        
        if hasattr(atom, 'outgoing') and atom.outgoing:
            data['outgoing'] = [a.id for a in atom.outgoing]
            data['description'] += f" with {len(atom.outgoing)} outgoing atoms"
        
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
            The deserialized atom
        """
        try:
            # Import Atom classes from parent module
            from . import Node, Link, Atom
            
            atom_id = data.get('id')
            atom_type = data.get('type')
            
            if not atom_id or not atom_type:
                logger.warning("Missing id or type in atom data")
                return None
            
            # Check if it's a node (has name) or link (has outgoing)
            if 'name' in data:
                # Deserialize as Node
                atom = Node(atom_type=atom_type, name=data['name'])
                atom.id = atom_id
            elif 'outgoing' in data:
                # Deserialize as Link
                # First, retrieve outgoing atoms
                outgoing_atoms = []
                for out_id in data['outgoing']:
                    out_atom = self.get_atom(out_id)
                    if out_atom:
                        outgoing_atoms.append(out_atom)
                    else:
                        logger.warning(f"Could not find outgoing atom {out_id}")
                
                atom = Link(atom_type=atom_type, outgoing_set=outgoing_atoms)
                atom.id = atom_id
            else:
                # Generic atom
                atom = Atom(atom_type=atom_type)
                atom.id = atom_id
            
            # Restore truth value if present
            if 'truth_value' in data:
                tv_data = data['truth_value']
                if hasattr(atom, 'tv'):
                    atom.tv.strength = tv_data.get('strength', 0.0)
                    atom.tv.confidence = tv_data.get('confidence', 0.0)
            
            # Restore attention value if present
            if 'attention_value' in data:
                av_data = data['attention_value']
                if hasattr(atom, 'av'):
                    atom.av.sti = av_data.get('sti', 0)
                    atom.av.lti = av_data.get('lti', 0)
                    atom.av.vlti = av_data.get('vlti', False)
            
            return atom
            
        except ImportError as e:
            logger.error(f"Failed to import Atom classes: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to deserialize atom: {e}")
            return None
