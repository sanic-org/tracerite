"""Tests for logging.py - logger configuration."""

import logging

import pytest


def test_logger_exists():
    """Test that the tracerite logger is created."""
    from tracerite.logging import logger
    
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tracerite"


def test_logger_level():
    """Test that the logger has INFO level set."""
    from tracerite.logging import logger
    
    assert logger.level == logging.INFO


def test_logger_can_log():
    """Test that the logger can log messages."""
    from tracerite.logging import logger
    
    # Should not raise any exceptions
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
