
def add_docstring(new_docstring: str):
    """ A decorator for setting the docstring programmatically. """
    def _doc(func):
        func.__doc__ = new_docstring
        return func
    return _doc
