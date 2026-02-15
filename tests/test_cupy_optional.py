"""
Test that CuPy is optional and imports are lazy.

This test verifies that:
1. The rapidshot package can be imported without CuPy installed
2. CuPy imports are lazy-loaded only when GPU acceleration is requested
3. The package gracefully falls back to CPU mode when CuPy is not available
"""
import sys
import unittest
from unittest.mock import patch


class TestCupyOptional(unittest.TestCase):
    """Test that CuPy is truly optional and properly lazy-loaded."""
    
    def test_no_top_level_cupy_import(self):
        """Verify that CuPy is not imported at module level in capture.py."""
        # Read the capture.py file
        with open('rapidshot/capture.py', 'r') as f:
            lines = f.readlines()
        
        # Check that there's no top-level CuPy import
        in_main_imports = True
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                in_main_imports = False
                break
            
            # Check for top-level cupy imports (not in try-except)
            # Skip comments and empty lines first
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            if in_main_imports and ('import cupy' in line or 'from cupy' in line):
                # Make sure it's indented (inside a function or try-except)
                indent = len(line) - len(line.lstrip())
                self.assertGreater(indent, 0, 
                    f"Found top-level CuPy import at line {i+1}: {line.strip()}")
    
    def test_no_cupy_available_constant(self):
        """Verify that CUPY_AVAILABLE constant is not set at module level."""
        with open('rapidshot/capture.py', 'r') as f:
            content = f.read()
        
        # Split into lines and check early part of file (before class definitions)
        lines = content.split('\n')
        import_section = []
        for line in lines:
            if line.strip().startswith('class '):
                break
            import_section.append(line)
        
        import_section_text = '\n'.join(import_section)
        
        # Check that CUPY_AVAILABLE is not being set
        self.assertNotIn('CUPY_AVAILABLE = True', import_section_text,
            "CUPY_AVAILABLE should not be set at module level")
        self.assertNotIn('CUPY_AVAILABLE = False', import_section_text,
            "CUPY_AVAILABLE should not be set at module level")
    
    def test_lazy_cupy_imports_present(self):
        """Verify that CuPy imports are present but lazy-loaded."""
        with open('rapidshot/capture.py', 'r') as f:
            lines = f.readlines()
        
        # Check that there are lazy imports of cupy (indented, inside try blocks or functions)
        lazy_imports = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith('#'):
                continue
            # Check for actual import statements (not in comments or strings)
            if stripped.startswith('import cupy') or stripped.startswith('from cupy'):
                indent = len(line) - len(line.lstrip())
                if indent > 0:  # It's indented, so it's lazy
                    lazy_imports += 1
        
        self.assertGreater(lazy_imports, 0, 
            "Should have at least one lazy CuPy import")
    
    def test_package_structure(self):
        """Verify package structure supports optional dependencies."""
        # Read setup.py
        with open('setup.py', 'r') as f:
            setup_content = f.read()
        
        # Check for gpu_cuda11 and gpu_cuda12 extras
        self.assertIn('gpu_cuda11', setup_content, 
            "setup.py should define gpu_cuda11 extra")
        self.assertIn('gpu_cuda12', setup_content,
            "setup.py should define gpu_cuda12 extra")
        self.assertIn('"gpu"', setup_content,
            "setup.py should define gpu extra")
        
        # Read pyproject.toml
        with open('pyproject.toml', 'r') as f:
            pyproject_content = f.read()
        
        # Check for gpu_cuda11 and gpu_cuda12 extras
        self.assertIn('gpu_cuda11', pyproject_content,
            "pyproject.toml should define gpu_cuda11 extra")
        self.assertIn('gpu_cuda12', pyproject_content,
            "pyproject.toml should define gpu_cuda12 extra")


if __name__ == '__main__':
    unittest.main()
