"""
Node9 AtomSpace Bridge - Connect AtomSpace to node9 namespace

This module creates a bridge between the AtomSpace hypergraph database and the
node9 distributed operating system namespace. It exposes AtomSpace operations
through the node9 filesystem interface using the Styx protocol, allowing atoms
to be manipulated as files in a Plan 9-style namespace.

Key features:
- Exposes atoms as files in the /cog/atoms/ namespace
- Provides special files for querying and pattern matching
- Maps cognitive modules to node9 virtual processes
- Handles serialization between Python and Lua data structures
- Supports distributed operation through node9's network transparency

Basic usage:
    # Start the bridge
    from cogprime.bridges.node9_atomspace_bridge import Node9AtomSpaceBridge
    
    bridge = Node9AtomSpaceBridge(atomspace=my_atomspace)
    bridge.start()
    
    # Now atoms can be accessed through node9 filesystem at /cog/atoms/
"""

import os
import sys
import json
import uuid
import time
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Import AtomSpace components
from ..atomspace import AtomSpace, Node, Link, Atom, TruthValue, AttentionValue

# Configure logging
logger = logging.getLogger(__name__)

# Check if node9 is available
try:
    import ctypes
    NODE9_AVAILABLE = True
except ImportError:
    NODE9_AVAILABLE = False
    logger.warning("node9 FFI libraries not available, falling back to subprocess mode")


class AtomSerializer:
    """Handles serialization and deserialization of atoms for node9 namespace."""
    
    @staticmethod
    def atom_to_dict(atom: Atom) -> Dict:
        """Convert an atom to a dictionary representation.
        
        Args:
            atom: The atom to convert
            
        Returns:
            Dictionary representation of the atom
        """
        base_dict = {
            "id": atom.id,
            "type": atom.atom_type,
            "tv": {
                "strength": atom.tv.strength,
                "confidence": atom.tv.confidence
            },
            "av": {
                "sti": atom.av.sti,
                "lti": atom.av.lti,
                "vlti": atom.av.vlti
            }
        }
        
        if atom.is_node():
            base_dict["name"] = atom.name
            base_dict["is_node"] = True
        else:  # Link
            base_dict["outgoing"] = [AtomSerializer.atom_to_dict(a) for a in atom.outgoing_set]
            base_dict["is_node"] = False
        
        return base_dict
    
    @staticmethod
    def dict_to_atom(data: Dict) -> Atom:
        """Convert a dictionary representation to an atom.
        
        Args:
            data: Dictionary representation of an atom
            
        Returns:
            The reconstructed atom
        """
        # Create truth and attention values
        tv = TruthValue(
            strength=data.get("tv", {}).get("strength", 1.0),
            confidence=data.get("tv", {}).get("confidence", 1.0)
        )
        
        av = AttentionValue(
            sti=data.get("av", {}).get("sti", 0.0),
            lti=data.get("av", {}).get("lti", 0.0),
            vlti=data.get("av", {}).get("vlti", False)
        )
        
        # Create the atom
        if data.get("is_node", True):
            atom = Node(data["type"], data["name"])
        else:
            # Recursively create outgoing atoms
            outgoing = [
                AtomSerializer.dict_to_atom(a) for a in data.get("outgoing", [])
            ]
            atom = Link(data["type"], outgoing)
        
        # Set values
        atom.id = data.get("id", str(uuid.uuid4()))
        atom.tv = tv
        atom.av = av
        
        return atom
    
    @staticmethod
    def atom_to_json(atom: Atom) -> str:
        """Convert an atom to a JSON string.
        
        Args:
            atom: The atom to convert
            
        Returns:
            JSON string representation of the atom
        """
        return json.dumps(AtomSerializer.atom_to_dict(atom))
    
    @staticmethod
    def json_to_atom(json_str: str) -> Atom:
        """Convert a JSON string to an atom.
        
        Args:
            json_str: JSON string representation of an atom
            
        Returns:
            The reconstructed atom
        """
        data = json.loads(json_str)
        return AtomSerializer.dict_to_atom(data)
    
    @staticmethod
    def atom_to_lua(atom: Atom) -> str:
        """Convert an atom to a Lua table representation.
        
        Args:
            atom: The atom to convert
            
        Returns:
            Lua table representation of the atom as a string
        """
        # Convert to dictionary first
        atom_dict = AtomSerializer.atom_to_dict(atom)
        
        # Convert dictionary to Lua table syntax
        return AtomSerializer._dict_to_lua_table(atom_dict)
    
    @staticmethod
    def _dict_to_lua_table(d: Dict) -> str:
        """Convert a Python dictionary to a Lua table string.
        
        Args:
            d: The dictionary to convert
            
        Returns:
            Lua table representation as a string
        """
        parts = []
        for k, v in d.items():
            # Format the key
            if isinstance(k, str):
                key = f'["{k}"]'
            else:
                key = f'[{k}]'
            
            # Format the value
            if v is None:
                value = "nil"
            elif isinstance(v, bool):
                value = "true" if v else "false"
            elif isinstance(v, (int, float)):
                value = str(v)
            elif isinstance(v, str):
                value = f'"{v}"'
            elif isinstance(v, dict):
                value = AtomSerializer._dict_to_lua_table(v)
            elif isinstance(v, list):
                value = AtomSerializer._list_to_lua_table(v)
            else:
                value = f'"{str(v)}"'
            
            parts.append(f"{key} = {value}")
        
        return "{" + ", ".join(parts) + "}"
    
    @staticmethod
    def _list_to_lua_table(lst: List) -> str:
        """Convert a Python list to a Lua table string.
        
        Args:
            lst: The list to convert
            
        Returns:
            Lua table representation as a string
        """
        parts = []
        for i, v in enumerate(lst, 1):  # Lua tables are 1-indexed
            if v is None:
                parts.append("nil")
            elif isinstance(v, bool):
                parts.append("true" if v else "false")
            elif isinstance(v, (int, float)):
                parts.append(str(v))
            elif isinstance(v, str):
                parts.append(f'"{v}"')
            elif isinstance(v, dict):
                parts.append(AtomSerializer._dict_to_lua_table(v))
            elif isinstance(v, list):
                parts.append(AtomSerializer._list_to_lua_table(v))
            else:
                parts.append(f'"{str(v)}"')
        
        return "{" + ", ".join(parts) + "}"


class Node9File:
    """Represents a file in the node9 namespace."""
    
    def __init__(self, path: str, content: str = "", is_dir: bool = False, 
                 read_fn: Callable = None, write_fn: Callable = None):
        """Initialize a node9 file.
        
        Args:
            path: Path in the namespace
            content: Initial content
            is_dir: Whether this is a directory
            read_fn: Function to call when the file is read
            write_fn: Function to call when the file is written
        """
        self.path = path
        self.content = content
        self.is_dir = is_dir
        self.read_fn = read_fn
        self.write_fn = write_fn
        self.children = {}  # For directories
    
    def read(self) -> str:
        """Read the file content.
        
        Returns:
            The file content
        """
        if self.read_fn:
            return self.read_fn()
        return self.content
    
    def write(self, content: str) -> bool:
        """Write content to the file.
        
        Args:
            content: The content to write
            
        Returns:
            True if successful, False otherwise
        """
        if self.write_fn:
            return self.write_fn(content)
        self.content = content
        return True
    
    def add_child(self, name: str, file) -> None:
        """Add a child file to this directory.
        
        Args:
            name: Name of the child
            file: The child file
        """
        if not self.is_dir:
            raise ValueError("Cannot add child to a non-directory")
        self.children[name] = file


class Node9Namespace:
    """Manages the node9 namespace for AtomSpace."""
    
    def __init__(self, root_path: str = "/cog"):
        """Initialize the namespace.
        
        Args:
            root_path: Root path in the node9 namespace
        """
        self.root_path = root_path
        self.files = {}
        
        # Create root directories
        self._create_directory(root_path)
        self._create_directory(f"{root_path}/atoms")
        self._create_directory(f"{root_path}/queries")
        self._create_directory(f"{root_path}/modules")
        self._create_directory(f"{root_path}/types")
    
    def _create_directory(self, path: str) -> None:
        """Create a directory in the namespace.
        
        Args:
            path: Path to create
        """
        self.files[path] = Node9File(path, is_dir=True)
    
    def create_file(self, path: str, content: str = "", 
                   read_fn: Callable = None, write_fn: Callable = None) -> None:
        """Create a file in the namespace.
        
        Args:
            path: Path to create
            content: Initial content
            read_fn: Function to call when the file is read
            write_fn: Function to call when the file is written
        """
        self.files[path] = Node9File(path, content, False, read_fn, write_fn)
        
        # Add to parent directory
        parent_path = os.path.dirname(path)
        if parent_path in self.files:
            parent = self.files[parent_path]
            name = os.path.basename(path)
            parent.add_child(name, self.files[path])
    
    def get_file(self, path: str) -> Optional[Node9File]:
        """Get a file from the namespace.
        
        Args:
            path: Path to get
            
        Returns:
            The file, or None if not found
        """
        return self.files.get(path)
    
    def list_directory(self, path: str) -> List[str]:
        """List the contents of a directory.
        
        Args:
            path: Path to list
            
        Returns:
            List of child names
        """
        file = self.get_file(path)
        if file and file.is_dir:
            return list(file.children.keys())
        return []


class Node9FFI:
    """Foreign Function Interface for node9 integration."""
    
    def __init__(self):
        """Initialize the FFI interface."""
        self.lib = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the FFI interface.
        
        Returns:
            True if successful, False otherwise
        """
        if not NODE9_AVAILABLE:
            logger.error("node9 FFI libraries not available")
            return False
        
        try:
            # Load the node9 library
            self.lib = ctypes.CDLL("libnode9.so")
            
            # Define function prototypes
            self.lib.node9_init.argtypes = [ctypes.c_char_p]
            self.lib.node9_init.restype = ctypes.c_int
            
            self.lib.node9_mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            self.lib.node9_mount.restype = ctypes.c_int
            
            self.lib.node9_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            self.lib.node9_create.restype = ctypes.c_int
            
            self.lib.node9_write.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            self.lib.node9_write.restype = ctypes.c_int
            
            self.lib.node9_read.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            self.lib.node9_read.restype = ctypes.c_int
            
            self.lib.node9_close.argtypes = [ctypes.c_int]
            self.lib.node9_close.restype = ctypes.c_int
            
            # Initialize node9
            result = self.lib.node9_init(b"/cog")
            if result != 0:
                logger.error(f"Failed to initialize node9: {result}")
                return False
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing node9 FFI: {e}")
            return False
    
    def mount_namespace(self, namespace_path: str, local_path: str) -> bool:
        """Mount a namespace path.
        
        Args:
            namespace_path: Path in the namespace
            local_path: Local path to mount
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            result = self.lib.node9_mount(
                namespace_path.encode('utf-8'),
                local_path.encode('utf-8'),
                0  # Flags
            )
            return result == 0
        except Exception as e:
            logger.error(f"Error mounting namespace: {e}")
            return False
    
    def create_file(self, path: str, content: str) -> int:
        """Create a file in the namespace.
        
        Args:
            path: Path in the namespace
            content: Initial content
            
        Returns:
            File descriptor, or -1 on error
        """
        if not self.initialized:
            return -1
        
        try:
            fd = self.lib.node9_create(
                path.encode('utf-8'),
                b"",  # Mode
                0o644  # Permissions
            )
            
            if fd < 0:
                logger.error(f"Failed to create file {path}: {fd}")
                return -1
            
            if content:
                result = self.lib.node9_write(
                    fd,
                    content.encode('utf-8'),
                    len(content)
                )
                
                if result != len(content):
                    logger.error(f"Failed to write to file {path}: {result}")
                    self.lib.node9_close(fd)
                    return -1
            
            return fd
        except Exception as e:
            logger.error(f"Error creating file: {e}")
            return -1
    
    def read_file(self, path: str, max_size: int = 4096) -> str:
        """Read a file from the namespace.
        
        Args:
            path: Path in the namespace
            max_size: Maximum size to read
            
        Returns:
            The file content, or empty string on error
        """
        if not self.initialized:
            return ""
        
        try:
            fd = self.lib.node9_open(
                path.encode('utf-8'),
                b"r"  # Mode
            )
            
            if fd < 0:
                logger.error(f"Failed to open file {path}: {fd}")
                return ""
            
            buffer = ctypes.create_string_buffer(max_size)
            result = self.lib.node9_read(fd, buffer, max_size)
            
            self.lib.node9_close(fd)
            
            if result < 0:
                logger.error(f"Failed to read file {path}: {result}")
                return ""
            
            return buffer.value.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return ""
    
    def write_file(self, path: str, content: str) -> bool:
        """Write to a file in the namespace.
        
        Args:
            path: Path in the namespace
            content: Content to write
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            fd = self.lib.node9_open(
                path.encode('utf-8'),
                b"w"  # Mode
            )
            
            if fd < 0:
                logger.error(f"Failed to open file {path}: {fd}")
                return False
            
            result = self.lib.node9_write(
                fd,
                content.encode('utf-8'),
                len(content)
            )
            
            self.lib.node9_close(fd)
            
            if result != len(content):
                logger.error(f"Failed to write to file {path}: {result}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return False


class LuaFFI:
    """FFI interface for Lua integration."""
    
    def __init__(self):
        """Initialize the Lua FFI interface."""
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the Lua FFI interface.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if LuaJIT is available
            subprocess.run(["luajit", "-v"], check=True, stdout=subprocess.PIPE)
            self.initialized = True
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("LuaJIT not found")
            return False
    
    def generate_ffi_bindings(self, output_path: str) -> bool:
        """Generate FFI bindings for Lua.
        
        Args:
            output_path: Path to write the bindings to
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            # Generate FFI bindings
            bindings = """
-- AtomSpace FFI bindings for Lua
local ffi = require("ffi")
local atomspace = {}

-- Define the C interface
ffi.cdef[[
    typedef struct {
        const char* id;
        const char* type;
        const char* name;
        double tv_strength;
        double tv_confidence;
        double av_sti;
        double av_lti;
        int av_vlti;
    } AtomInfo;
    
    int atomspace_initialize(const char* config);
    AtomInfo* atomspace_add_node(const char* type, const char* name);
    AtomInfo* atomspace_add_link(const char* type, const char** outgoing_ids, int outgoing_count);
    AtomInfo* atomspace_get_atom(const char* id);
    int atomspace_remove_atom(const char* id);
    char* atomspace_query(const char* pattern_json);
    void atomspace_free_string(char* str);
]]

-- Load the library
local lib = ffi.load("/cog/lib/libatomspace_bridge.so")

-- Initialize the AtomSpace
function atomspace.initialize(config)
    return lib.atomspace_initialize(config or "")
end

-- Add a node to the AtomSpace
function atomspace.add_node(type, name)
    local info = lib.atomspace_add_node(type, name)
    if info == nil then
        return nil
    end
    
    local result = {
        id = ffi.string(info.id),
        type = ffi.string(info.type),
        name = ffi.string(info.name),
        tv = {
            strength = info.tv_strength,
            confidence = info.tv_confidence
        },
        av = {
            sti = info.av_sti,
            lti = info.av_lti,
            vlti = info.av_vlti ~= 0
        }
    }
    
    return result
end

-- Add a link to the AtomSpace
function atomspace.add_link(type, outgoing)
    local count = #outgoing
    local ids = ffi.new("const char*[?]", count)
    
    for i = 1, count do
        ids[i-1] = outgoing[i]
    end
    
    local info = lib.atomspace_add_link(type, ids, count)
    if info == nil then
        return nil
    end
    
    local result = {
        id = ffi.string(info.id),
        type = ffi.string(info.type),
        tv = {
            strength = info.tv_strength,
            confidence = info.tv_confidence
        },
        av = {
            sti = info.av_sti,
            lti = info.av_lti,
            vlti = info.av_vlti ~= 0
        }
    }
    
    return result
end

-- Get an atom from the AtomSpace
function atomspace.get_atom(id)
    local info = lib.atomspace_get_atom(id)
    if info == nil then
        return nil
    end
    
    local result = {
        id = ffi.string(info.id),
        type = ffi.string(info.type),
        tv = {
            strength = info.tv_strength,
            confidence = info.tv_confidence
        },
        av = {
            sti = info.av_sti,
            lti = info.av_lti,
            vlti = info.av_vlti ~= 0
        }
    }
    
    if info.name ~= nil then
        result.name = ffi.string(info.name)
        result.is_node = true
    else
        result.is_node = false
    end
    
    return result
end

-- Remove an atom from the AtomSpace
function atomspace.remove_atom(id)
    return lib.atomspace_remove_atom(id) ~= 0
end

-- Query the AtomSpace
function atomspace.query(pattern)
    local pattern_json
    if type(pattern) == "string" then
        pattern_json = pattern
    else
        -- Convert table to JSON
        pattern_json = json.encode(pattern)
    end
    
    local result_ptr = lib.atomspace_query(pattern_json)
    if result_ptr == nil then
        return {}
    end
    
    local result_str = ffi.string(result_ptr)
    lib.atomspace_free_string(result_ptr)
    
    -- Parse JSON result
    local results = json.decode(result_str)
    return results
end

return atomspace
"""
            
            # Write to file
            with open(output_path, "w") as f:
                f.write(bindings)
            
            return True
        except Exception as e:
            logger.error(f"Error generating FFI bindings: {e}")
            return False


class VirtualProcess:
    """Represents a virtual process in the node9 namespace."""
    
    def __init__(self, name: str, module_path: str, namespace_path: str):
        """Initialize a virtual process.
        
        Args:
            name: Name of the process
            module_path: Path to the Python module
            namespace_path: Path in the node9 namespace
        """
        self.name = name
        self.module_path = module_path
        self.namespace_path = namespace_path
        self.process = None
        self.running = False
    
    def start(self) -> bool:
        """Start the virtual process.
        
        Returns:
            True if successful, False otherwise
        """
        if self.running:
            return True
        
        try:
            # Import the module
            module_name = self.module_path.replace("/", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            
            __import__(module_name)
            
            # Start in a separate thread
            self.process = threading.Thread(
                target=self._run_process,
                daemon=True
            )
            self.process.start()
            
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Error starting virtual process {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the virtual process.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.running:
            return True
        
        try:
            # Set flag to stop
            self.running = False
            
            # Wait for thread to terminate
            if self.process:
                self.process.join(timeout=5.0)
            
            return True
        except Exception as e:
            logger.error(f"Error stopping virtual process {self.name}: {e}")
            return False
    
    def _run_process(self) -> None:
        """Run the virtual process."""
        try:
            # Import the module
            module_name = self.module_path.replace("/", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            
            module = sys.modules[module_name]
            
            # Call the main function if available
            if hasattr(module, "main"):
                module.main()
            elif hasattr(module, "run"):
                module.run()
            else:
                logger.warning(f"No main or run function found in {module_name}")
        except Exception as e:
            logger.error(f"Error in virtual process {self.name}: {e}")


class Node9AtomSpaceBridge:
    """Bridge between AtomSpace and node9 namespace."""
    
    def __init__(self, atomspace: AtomSpace, namespace_path: str = "/cog"):
        """Initialize the bridge.
        
        Args:
            atomspace: The AtomSpace to bridge
            namespace_path: Path in the node9 namespace
        """
        self.atomspace = atomspace
        self.namespace_path = namespace_path
        self.namespace = Node9Namespace(namespace_path)
        self.ffi = Node9FFI()
        self.lua_ffi = LuaFFI()
        self.running = False
        self.virtual_processes = {}
    
    def start(self) -> bool:
        """Start the bridge.
        
        Returns:
            True if successful, False otherwise
        """
        if self.running:
            return True
        
        try:
            # Initialize FFI
            if not self.ffi.initialize():
                logger.warning("Failed to initialize node9 FFI, continuing in subprocess mode")
            
            # Initialize Lua FFI
            if not self.lua_ffi.initialize():
                logger.warning("Failed to initialize Lua FFI, Lua integration will be limited")
            
            # Set up namespace
            self._setup_namespace()
            
            # Generate Lua bindings
            bindings_path = f"{self.namespace_path}/lib/atomspace.lua"
            if self.lua_ffi.initialized:
                self.lua_ffi.generate_ffi_bindings(bindings_path)
            
            # Start virtual processes
            for vproc in self.virtual_processes.values():
                vproc.start()
            
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Error starting bridge: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the bridge.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.running:
            return True
        
        try:
            # Stop virtual processes
            for vproc in self.virtual_processes.values():
                vproc.stop()
            
            self.running = False
            return True
        except Exception as e:
            logger.error(f"Error stopping bridge: {e}")
            return False
    
    def _setup_namespace(self) -> None:
        """Set up the node9 namespace."""
        # Create directories
        for path in [
            f"{self.namespace_path}/atoms",
            f"{self.namespace_path}/queries",
            f"{self.namespace_path}/modules",
            f"{self.namespace_path}/types",
            f"{self.namespace_path}/lib",
            f"{self.namespace_path}/control",
        ]:
            self.namespace._create_directory(path)
        
        # Create special files
        
        # Control files
        self.namespace.create_file(
            f"{self.namespace_path}/control/status",
            content="running",
            read_fn=lambda: "running" if self.running else "stopped"
        )
        
        self.namespace.create_file(
            f"{self.namespace_path}/control/stop",
            content="",
            write_fn=lambda _: self.stop()
        )
        
        self.namespace.create_file(
            f"{self.namespace_path}/control/start",
            content="",
            write_fn=lambda _: self.start()
        )
        
        # Query files
        self.namespace.create_file(
            f"{self.namespace_path}/queries/by_type",
            content="",
            write_fn=lambda content: self._handle_type_query(content)
        )
        
        self.namespace.create_file(
            f"{self.namespace_path}/queries/by_pattern",
            content="",
            write_fn=lambda content: self._handle_pattern_query(content)
        )
        
        self.namespace.create_file(
            f"{self.namespace_path}/queries/add_node",
            content="",
            write_fn=lambda content: self._handle_add_node(content)
        )
        
        self.namespace.create_file(
            f"{self.namespace_path}/queries/add_link",
            content="",
            write_fn=lambda content: self._handle_add_link(content)
        )
        
        # Expose all atom types
        self._create_type_files()
        
        # Expose all atoms
        self._create_atom_files()
    
    def _create_type_files(self) -> None:
        """Create files for atom types."""
        # Get all atom types from the AtomSpace
        # This is a placeholder - in a real implementation, we would get the types from the AtomSpace
        types = [
            "ConceptNode", "PredicateNode", "NumberNode", "VariableNode",
            "InheritanceLink", "EvaluationLink", "ListLink", "SetLink",
            "AndLink", "OrLink", "NotLink", "ExecutionLink"
        ]
        
        for atom_type in types:
            self.namespace.create_file(
                f"{self.namespace_path}/types/{atom_type}",
                content=atom_type,
                read_fn=lambda t=atom_type: t
            )
    
    def _create_atom_files(self) -> None:
        """Create files for atoms."""
        # In a real implementation, we would create files for all atoms in the AtomSpace
        # For now, we'll just create a few example files
        
        # Get all atoms from the AtomSpace
        # This is just a placeholder - in a real implementation, we would get all atoms from the AtomSpace
        atoms = []
        
        # Create a file for each atom
        for atom in atoms:
            atom_path = f"{self.namespace_path}/atoms/{atom.id}"
            
            self.namespace.create_file(
                atom_path,
                content=AtomSerializer.atom_to_json(atom),
                read_fn=lambda a=atom: AtomSerializer.atom_to_json(a),
                write_fn=lambda content, a=atom: self._handle_atom_update(a, content)
            )
    
    def _handle_type_query(self, content: str) -> bool:
        """Handle a query by type.
        
        Args:
            content: The type to query for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get atoms of the specified type
            atoms = self.atomspace.get_atoms_by_type(content.strip())
            
            # Create a result file
            result_id = str(uuid.uuid4())
            result_path = f"{self.namespace_path}/queries/results/{result_id}"
            
            # Convert atoms to JSON
            result = [AtomSerializer.atom_to_dict(atom) for atom in atoms]
            result_json = json.dumps(result)
            
            # Create the result file
            self.namespace.create_file(result_path, result_json)
            
            # Create a symlink to the result
            self.namespace.create_file(
                f"{self.namespace_path}/queries/last_result",
                content=result_id,
                read_fn=lambda: result_id
            )
            
            return True
        except Exception as e:
            logger.error(f"Error handling type query: {e}")
            return False
    
    def _handle_pattern_query(self, content: str) -> bool:
        """Handle a pattern query.
        
        Args:
            content: The pattern to query for (JSON)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the pattern
            pattern_data = json.loads(content)
            
            # Convert to an atom
            pattern = AtomSerializer.dict_to_atom(pattern_data)
            
            # Query the AtomSpace
            results = self.atomspace.query(pattern)
            
            # Create a result file
            result_id = str(uuid.uuid4())
            result_path = f"{self.namespace_path}/queries/results/{result_id}"
            
            # Convert results to JSON
            result = [AtomSerializer.atom_to_dict(atom) for atom in results]
            result_json = json.dumps(result)
            
            # Create the result file
            self.namespace.create_file(result_path, result_json)
            
            # Create a symlink to the result
            self.namespace.create_file(
                f"{self.namespace_path}/queries/last_result",
                content=result_id,
                read_fn=lambda: result_id
            )
            
            return True
        except Exception as e:
            logger.error(f"Error handling pattern query: {e}")
            return False
    
    def _handle_add_node(self, content: str) -> bool:
        """Handle adding a node.
        
        Args:
            content: JSON representation of the node to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the node data
            data = json.loads(content)
            
            # Create the node
            node = Node(data["type"], data["name"])
            
            # Set truth value if provided
            if "tv" in data:
                node.set_truth_value(TruthValue(
                    strength=data["tv"].get("strength", 1.0),
                    confidence=data["tv"].get("confidence", 1.0)
                ))
            
            # Set attention value if provided
            if "av" in data:
                node.set_attention_value(AttentionValue(
                    sti=data["av"].get("sti", 0.0),
                    lti=data["av"].get("lti", 0.0),
                    vlti=data["av"].get("vlti", False)
                ))
            
            # Add to AtomSpace
            result = self.atomspace.add(node)
            
            # Create a file for the new atom
            atom_path = f"{self.namespace_path}/atoms/{result.id}"
            
            self.namespace.create_file(
                atom_path,
                content=AtomSerializer.atom_to_json(result),
                read_fn=lambda: AtomSerializer.atom_to_json(result),
                write_fn=lambda c: self._handle_atom_update(result, c)
            )
            
            return True
        except Exception as e:
            logger.error(f"Error handling add node: {e}")
            return False
    
    def _handle_add_link(self, content: str) -> bool:
        """Handle adding a link.
        
        Args:
            content: JSON representation of the link to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the link data
            data = json.loads(content)
            
            # Get the outgoing atoms
            outgoing = []
            for atom_id in data.get("outgoing_ids", []):
                atom = self.atomspace.get_atom(atom_id)
                if atom:
                    outgoing.append(atom)
                else:
                    logger.error(f"Atom {atom_id} not found")
                    return False
            
            # Create the link
            link = Link(data["type"], outgoing)
            
            # Set truth value if provided
            if "tv" in data:
                link.set_truth_value(TruthValue(
                    strength=data["tv"].get("strength", 1.0),
                    confidence=data["tv"].get("confidence", 1.0)
                ))
            
            # Set attention value if provided
            if "av" in data:
                link.set_attention_value(AttentionValue(
                    sti=data["av"].get("sti", 0.0),
                    lti=data["av"].get("lti", 0.0),
                    vlti=data["av"].get("vlti", False)
                ))
            
            # Add to AtomSpace
            result = self.atomspace.add(link)
            
            # Create a file for the new atom
            atom_path = f"{self.namespace_path}/atoms/{result.id}"
            
            self.namespace.create_file(
                atom_path,
                content=AtomSerializer.atom_to_json(result),
                read_fn=lambda: AtomSerializer.atom_to_json(result),
                write_fn=lambda c: self._handle_atom_update(result, c)
            )
            
            return True
        except Exception as e:
            logger.error(f"Error handling add link: {e}")
            return False
    
    def _handle_atom_update(self, atom: Atom, content: str) -> bool:
        """Handle updating an atom.
        
        Args:
            atom: The atom to update
            content: JSON representation of the updated atom
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the updated atom data
            data = json.loads(content)
            
            # Update truth value if provided
            if "tv" in data:
                atom.set_truth_value(TruthValue(
                    strength=data["tv"].get("strength", 1.0),
                    confidence=data["tv"].get("confidence", 1.0)
                ))
            
            # Update attention value if provided
            if "av" in data:
                atom.set_attention_value(AttentionValue(
                    sti=data["av"].get("sti", 0.0),
                    lti=data["av"].get("lti", 0.0),
                    vlti=data["av"].get("vlti", False)
                ))
            
            # Update the atom in the AtomSpace
            self.atomspace.add(atom)
            
            return True
        except Exception as e:
            logger.error(f"Error handling atom update: {e}")
            return False
    
    def register_cognitive_module(self, name: str, module_path: str) -> bool:
        """Register a cognitive module as a virtual process.
        
        Args:
            name: Name of the module
            module_path: Path to the Python module
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a virtual process
            vproc = VirtualProcess(
                name=name,
                module_path=module_path,
                namespace_path=f"{self.namespace_path}/modules/{name}"
            )
            
            # Add to virtual processes
            self.virtual_processes[name] = vproc
            
            # Create directory for the module
            self.namespace._create_directory(f"{self.namespace_path}/modules/{name}")
            
            # Create control files for the module
            self.namespace.create_file(
                f"{self.namespace_path}/modules/{name}/status",
                content="stopped",
                read_fn=lambda: "running" if vproc.running else "stopped"
            )
            
            self.namespace.create_file(
                f"{self.namespace_path}/modules/{name}/start",
                content="",
                write_fn=lambda _: vproc.start()
            )
            
            self.namespace.create_file(
                f"{self.namespace_path}/modules/{name}/stop",
                content="",
                write_fn=lambda _: vproc.stop()
            )
            
            # Start the virtual process if the bridge is running
            if self.running:
                vproc.start()
            
            return True
        except Exception as e:
            logger.error(f"Error registering cognitive module {name}: {e}")
            return False
    
    def unregister_cognitive_module(self, name: str) -> bool:
        """Unregister a cognitive module.
        
        Args:
            name: Name of the module
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Stop the virtual process
            if name in self.virtual_processes:
                vproc = self.virtual_processes[name]
                vproc.stop()
                del self.virtual_processes[name]
            
            return True
        except Exception as e:
            logger.error(f"Error unregistering cognitive module {name}: {e}")
            return False


# Factory function for creating bridges

def create_bridge(atomspace: AtomSpace, namespace_path: str = "/cog") -> Node9AtomSpaceBridge:
    """Create a new bridge.
    
    Args:
        atomspace: The AtomSpace to bridge
        namespace_path: Path in the node9 namespace
        
    Returns:
        A new Node9AtomSpaceBridge instance
    """
    return Node9AtomSpaceBridge(atomspace, namespace_path)


# Command-line interface

def main():
    """Command-line interface for the bridge."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Node9 AtomSpace Bridge")
    parser.add_argument("--atomspace", type=str, default="local",
                       help="AtomSpace backend type (local, node9, mem0, distributed)")
    parser.add_argument("--namespace", type=str, default="/cog",
                       help="Path in the node9 namespace")
    parser.add_argument("--config", type=str, default="",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Create AtomSpace
    atomspace = AtomSpace(args.atomspace)
    
    # Create bridge
    bridge = Node9AtomSpaceBridge(atomspace, args.namespace)
    
    # Start bridge
    if bridge.start():
        print(f"Bridge started at {args.namespace}")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping bridge...")
            bridge.stop()
    else:
        print("Failed to start bridge")


if __name__ == "__main__":
    main()
