"""Compatibility shim for the tree-sitter language package.

The project and tests import ``tree_sitter_languages``. The installed
package exposes the module as ``tree_sitter_language_pack``.
This file preserves the existing import style without changing the
application structure.
"""

from tree_sitter_language_pack import *  # noqa: F401,F403
