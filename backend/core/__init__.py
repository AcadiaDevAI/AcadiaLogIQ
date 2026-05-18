"""Core infrastructure helpers.

Modules in here run at the *very* start of process startup — before
``backend.config`` instantiates ``Settings`` — so they MUST be
self-contained and may NOT import anything that triggers config
loading.
"""
