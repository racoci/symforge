"""Domain exceptions for modular LoRA adapter registry, loading, and verification."""

from __future__ import annotations


class AdapterError(Exception):
    """Base exception for all adapter-related operations."""


class AdapterVerificationError(AdapterError):
    """Raised when an adapter artifact fails cryptographic integrity or checksum verification."""


class AdapterNotFoundError(AdapterError):
    """Raised when a requested adapter ID or collection is not registered."""


class AdapterAlreadyExistsError(AdapterError):
    """Raised when attempting to register an adapter with an ID that already exists."""


class AdapterMismatchError(AdapterError):
    """Raised when adapter tensor shapes or module names do not match target layers."""


class AdapterInjectionError(AdapterError):
    """Raised when transforming target base model modules into LoRA layers fails."""
