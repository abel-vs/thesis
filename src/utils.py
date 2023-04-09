from typing import List, Dict, Union
import ast
from typing import List, Dict, Union, Any
from tempfile import SpooledTemporaryFile, NamedTemporaryFile

from fastapi import UploadFile


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


def extract_class_info(node: ast.ClassDef) -> Dict[str, Union[str, List[str]]]:
    class_info = {"name": node.name, "parents": []}
    for base in node.bases:
        if isinstance(base, ast.Name):
            class_info["parents"].append(base.id)
    return class_info


async def get_classes(py_file: UploadFile) -> List[Dict[str, Union[str, List[str]]]]:
    content = await py_file.read()
    content = content.decode("utf-8")
    parsed_ast = ast.parse(content)

    classes = []
    for node in parsed_ast.body:
        if isinstance(node, ast.ClassDef):
            class_info = extract_class_info(node)
            classes.append(class_info)
    return classes
