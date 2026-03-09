class DraftClass:
    """
    Just a draft class to test the API and importability of the package. Not intended for actual use.
    """

    def __init__(self):
        self.name = "DraftClass"

    def __str__(self):
        return f"{self.name}"
    
    def do_something(self):
        raise NotImplementedError("This method is not implemented yet.")