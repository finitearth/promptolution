"""Configuration class for the promptolution library."""
from logging import Logger
from typing import Set

logger = Logger(__name__)


class ExperimentConfig:
    """Configuration class for the promptolution library.

    This is a unified configuration class that handles all experiment settings.
    It provides validation and tracking of used fields.
    """

    def __init__(self, **kwargs):
        """Initialize the configuration with the provided keyword arguments."""
        self._used_attributes: Set[str] = set()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattribute__(self, name):
        """Override attribute access to track used attributes."""
        # Get the attribute using the standard mechanism
        value = object.__getattribute__(self, name)

        if not name.startswith("_") and not callable(value):
            self._used_attributes.add(name)

        return value

    def apply_to(self, obj):
        """Apply matching attributes from this config to an existing object.

        Examines each attribute of the target object and updates it if a matching
        attribute exists in the config.

        Args:
            obj: The object to update with config values

        Returns:
            The updated object
        """
        for attr_name in dir(obj):
            if attr_name.startswith("_") or not isinstance(
                getattr(obj, attr_name), (str, int, float, list, type(None))
            ):
                continue

            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                setattr(obj, attr_name, getattr(self, attr_name))

        return obj

    def validate(self):
        """Check if any attributes were not used and run validation.

        Does not raise an error, but logs a warning if any attributes are unused or validation fails.
        """
        all_attributes = {k for k in self.__dict__ if not k.startswith("_")}
        unused_attributes = all_attributes - self._used_attributes
        if unused_attributes:
            logger.warning(f"Unused configuration attributes: {unused_attributes}")
