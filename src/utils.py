from tempfile import SpooledTemporaryFile, NamedTemporaryFile


def spooled_to_named(spooled_file: SpooledTemporaryFile, suffix = None):
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

