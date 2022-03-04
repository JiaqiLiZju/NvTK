"""
Utilities for loading and instantiating extended modules and models
"""

import os, sys, importlib, types
import NvTK

def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class.
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)


def load_module(path):
    """
    Load extended module, link to NvTK, enable import from NvTK.ExtendedModuleName

    This method expects that you pass in the path to a valid Python module FilePath or Dir.
    FilePath: Load a module created based on a Python file path; 
    Dir: where the `__init__.py` file already imports the model class.
    (e.g. `__init__.py` contains the line `from <model_class_file> import <ModelClass>`).

    Example: `load_module("../pathDir/model_class_file.py")`
    Then `NvTK.model_class_file.ModelClass(args)` could be used in your code.

    Parameters
    ----------
    path : str

    Returns
    -------
    NvTK.ExtendedModuleName, None
    """

    module = None
    if os.path.isdir(path):
        module = module_from_dir(path)
        module_name = os.path.split(path)[-1]
    else:
        module = module_from_file(path)
        module_file = os.path.split(path)[-1]
        module_name = module_file[:-3]

    setattr(NvTK, module_name, module)

