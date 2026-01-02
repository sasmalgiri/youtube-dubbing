def read_file(file_path):
    """Reads the contents of a file and returns it as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    """Writes the given content to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def append_to_file(file_path, content):
    """Appends the given content to a file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content)

def delete_file(file_path):
    """Deletes a file if it exists."""
    import os
    if os.path.exists(file_path):
        os.remove(file_path)

def list_files_in_directory(directory_path):
    """Returns a list of files in the specified directory."""
    import os
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]