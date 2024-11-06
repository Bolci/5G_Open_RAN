class DataProcessorException(Exception):
    """
    Custom exception class for data processing errors.

    This exception is intended to be used for signaling errors that occur
    during data processing operations within the application.

    Attributes:
        message (str): Optional. The error message describing the exception.
    """

    def __init__(self, message=None):
        """
        Initialize the DataProcessorException.

        Args:
            message (str, optional): The error message. Defaults to None.
        """
        super().__init__(message)
