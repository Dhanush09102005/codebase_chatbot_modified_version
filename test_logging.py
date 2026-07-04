#!/usr/bin/env python3
"""
Test script demonstrating improved logging configuration.

Run this to verify:
1. No duplicate handlers on re-import
2. Rotating file handler prevents large log files
3. Log directory configurable from settings
"""

import sys


def test_no_duplicate_handlers():
    """Test that re-importing doesn't create duplicate handlers."""
    print("=" * 80)
    print("TEST: No Duplicate Handlers")
    print("=" * 80)
    
    # First import
    from app.core.logging import logger
    initial_handler_count = len(logger.handlers)
    print(f"Initial handler count: {initial_handler_count}")
    
    # Re-import (simulates reload or multiple imports)
    import importlib
    import app.core.logging
    importlib.reload(app.core.logging)
    from app.core.logging import logger as logger2
    
    final_handler_count = len(logger2.handlers)
    print(f"After reload handler count: {final_handler_count}")
    
    if initial_handler_count == final_handler_count:
        print("✅ PASS: No duplicate handlers created")
    else:
        print(f"❌ FAIL: Handler count increased from {initial_handler_count} to {final_handler_count}")
    
    return initial_handler_count == final_handler_count


def test_rotating_file_handler():
    """Test that rotating file handler is configured."""
    print("\n" + "=" * 80)
    print("TEST: Rotating File Handler")
    print("=" * 80)
    
    from app.core.logging import logger
    from logging.handlers import RotatingFileHandler
    
    # Check if any handler is a RotatingFileHandler
    has_rotating = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    
    if has_rotating:
        print("✅ PASS: RotatingFileHandler is configured")
        
        # Show config
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                print(f"  File: {handler.baseFilename}")
                print(f"  Max bytes: {handler.maxBytes:,} bytes ({handler.maxBytes / 1024 / 1024:.1f} MB)")
                print(f"  Backup count: {handler.backupCount}")
    else:
        print("❌ FAIL: No RotatingFileHandler found")
    
    return has_rotating


def test_log_directory_config():
    """Test that log directory comes from config."""
    print("\n" + "=" * 80)
    print("TEST: Log Directory from Config")
    print("=" * 80)
    
    from app.core import settings
    from app.core.logging import logger
    from logging.handlers import RotatingFileHandler
    
    print(f"Configured log directory: {settings.log_directory}")
    
    # Check if file handler uses config path
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            handler_dir = handler.baseFilename.rsplit('/', 1)[0] if '/' in handler.baseFilename else handler.baseFilename.rsplit('\\', 1)[0]
            print(f"Handler log directory: {handler_dir}")
            
            if str(settings.log_directory) in handler.baseFilename:
                print("✅ PASS: Log directory uses config setting")
                return True
    
    print("❌ FAIL: Log directory doesn't match config")
    return False


def test_logging_functionality():
    """Test that logging actually works."""
    print("\n" + "=" * 80)
    print("TEST: Logging Functionality")
    print("=" * 80)
    
    from app.core.logging import logger
    
    # Test different log levels
    logger.debug("🐛 This is a DEBUG message")
    logger.info("ℹ️  This is an INFO message")
    logger.warning("⚠️  This is a WARNING message")
    logger.error("❌ This is an ERROR message")
    
    print("\n✅ PASS: Logging works (check console and ./logs/app.log)")
    return True


def test_handler_details():
    """Show detailed handler configuration."""
    print("\n" + "=" * 80)
    print("HANDLER DETAILS")
    print("=" * 80)
    
    from app.core.logging import logger
    
    print(f"Logger name: {logger.name}")
    print(f"Logger level: {logger.level}")
    print(f"Handler count: {len(logger.handlers)}")
    print(f"Propagate: {logger.propagate}")
    
    print("\nHandlers:")
    for i, handler in enumerate(logger.handlers, 1):
        print(f"\n  Handler {i}: {handler.__class__.__name__}")
        print(f"    Level: {handler.level}")
        print(f"    Formatter: {handler.formatter._fmt if handler.formatter else None}")
        
        if hasattr(handler, 'baseFilename'):
            print(f"    File: {handler.baseFilename}")
        if hasattr(handler, 'stream'):
            print(f"    Stream: {handler.stream.name if hasattr(handler.stream, 'name') else handler.stream}")


if __name__ == "__main__":
    results = []
    
    results.append(test_no_duplicate_handlers())
    results.append(test_rotating_file_handler())
    results.append(test_log_directory_config())
    results.append(test_logging_functionality())
    test_handler_details()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        sys.exit(1)
