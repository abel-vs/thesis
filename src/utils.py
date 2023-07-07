import importlib.util
import ast
import importlib
from collections import OrderedDict
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import List

import torch
from fastapi import UploadFile
import general


def spooled_to_named(spooled_file: SpooledTemporaryFile, suffix=None):
    # Assume spooled_file is a SpooledTemporaryFile object
    spooled_file.seek(0)
    spooled_file_contents = spooled_file.read()

    # Create a new NamedTemporaryFile object
    named_file = NamedTemporaryFile(suffix=suffix, delete=False)

    # Write the contents of the spooled file to the named file
    named_file.write(spooled_file_contents)

    named_file.flush()

    # Close the spooled file
    spooled_file.close()

    print(named_file.file)

    # Don't forget to close named_file when done using!
    return named_file


def bytes_to_named(bytes_object: bytes, suffix=None):
    # Create a new NamedTemporaryFile object
    named_file = NamedTemporaryFile(suffix=suffix, delete=False)

    # Write the contents of the bytes object to the named file
    named_file.write(bytes_object)
    named_file.flush()

    print(named_file.name)

    # Don't forget to close named_file when done using!
    return named_file

def prepare_model_params(model_state: bytes, model_architecture: bytes):
    # Convert the model_state and model_architecture to NamedTemporaryFile objects
    model_state_file: NamedTemporaryFile = bytes_to_named(model_state, suffix=".pth")
    model_architecture_file: NamedTemporaryFile = bytes_to_named(model_architecture, suffix=".py")

    return model_state_file, model_architecture_file


# Method to extract all torch.nn.Module classes from a module
def extract_modules(uploaded_file: UploadFile) -> List[str]:
    nn_module_classes = []

    # Read the source code from the uploaded file
    uploaded_file.file.seek(0)  # Reset the file pointer to the beginning
    source_code = uploaded_file.file.read().decode("utf-8")

    # Parse the source code using ast
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (
                    isinstance(base, ast.Attribute) and
                    base.attr == "Module" and
                    isinstance(base.value, ast.Name) and
                    base.value.id == "nn"
                ):
                    nn_module_classes.append(node.name)

    return nn_module_classes


# Method to extract top-level method definitions from a module
def extract_methods(uploaded_file: UploadFile) -> List[str]:
    method_names = []

    # Read the source code from the uploaded file
    uploaded_file.file.seek(0)  # Reset the file pointer to the beginning
    source_code = uploaded_file.file.read().decode("utf-8")

    # Parse the source code using ast
    tree = ast.parse(source_code)

    # Iterate through the top-level nodes
    for node in tree.body:
        # Check if the node is a FunctionDef
        if isinstance(node, ast.FunctionDef):
            method_names.append(node.name)

    return method_names


# Method to detect the data type of a .pth file
def detect_pth_data_type(pth_file):
    device = general.get_device()
    try:
        data = torch.load(pth_file.name, map_location=device)
    except EOFError:
        return ("corrupted", None)

    if isinstance(data, torch.nn.Module):
        return ("model", data)

    if isinstance(data, OrderedDict):
        # Check if it's a state dictionary by looking for weight and bias keys
        for key in data.keys():
            if key.endswith(".weight") or key.endswith(".bias"):
                return ("state_dict", data)

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, OrderedDict):
                for key in value.keys():
                    if key.endswith(".weight") or key.endswith(".bias"):
                        return ("state_dict", value)

    return ("unknown", None)


# Method to import a model from a .pth file and a .py file
def import_model(model_state_file, model_architecture_file, model_definition):

    # Load the state dictionary from the .pth file
    (state_type, state_object) = detect_pth_data_type(model_state_file)

    if state_type == "model":
        return state_object

    # Import the model architectures from the .py file
    spec = importlib.util.spec_from_file_location(
        "model_architecture", model_architecture_file.name)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    device = general.get_device()

    # Find the specified model class or method in the module
    model_entity = getattr(model_module, model_definition.name, None)

    print("Model entity:", model_entity)
    if model_entity is None:
        raise ValueError(
            f"Selected PyTorch '{model_definition.type}' '{model_definition.name}' not found in the architecture file")

    # Create an instance of the specified model class or call the specified method
    if model_definition.type.lower() == "module":
        if not issubclass(model_entity, torch.nn.Module) or model_entity == torch.nn.Module:
            raise ValueError(
                f"Selected PyTorch '{model_definition.type}' '{model_definition.name}' is not a valid module in the architecture file")
        model = model_entity()
    elif model_definition.type.lower() == "method":
        model = model_entity()
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"Selected PyTorch '{model_definition.type}' '{model_definition.name}' method did not return a valid PyTorch model instance")
    else:
        raise ValueError(
            f"Unknown model_definition type: {model_definition.type}")
    
    print("Model type:", model)

    
    if state_type == "model":
        model = state_object
    elif state_type == "state_dict":
        state_dict = state_object
        model.load_state_dict(state_dict)
    elif state_type == "corrupted":
        raise ValueError(
            f"Corrupted state dictionary file: {model_state_file.name}")
    else:  # Unknown data type
        raise ValueError(
            f"Unknown data type in the state dictionary file: {state_type}")

    model.eval()

    return model
