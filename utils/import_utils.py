import importlib
import sys
import os
import types
import inspect
import importlib.util


def dynamic_import(class_name: str, module_name: str = "transformers"):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    return cls


def dynamic_import_function(path: str, base_path: str = None):
    try:
        module_name = ".".join(path.split(".")[:-1])
        function_name = path.split(".")[-1]

        if base_path is None:
            caller_file = inspect.stack()[1].filename
            base_path = os.path.abspath(
                os.path.join(os.path.dirname(caller_file), "..")
            )

        if base_path not in sys.path:
            sys.path.insert(0, base_path)

        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    except Exception as e:
        raise ImportError(f"Failed to import '{path}' (base_path='{base_path}'): {e}")


def import_function_from_file(file_path: str, function_name: str):
    try:
        file_path = os.path.abspath(file_path)
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            raise AttributeError(f"'{function_name}' not found in '{file_path}'")

        return getattr(module, function_name)

    except Exception as e:
        raise ImportError(
            f"Failed to load function '{function_name}' from file '{file_path}': {e}"
        )
