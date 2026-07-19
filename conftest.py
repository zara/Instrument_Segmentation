"""
Shared pytest configuration.

The Airflow plugin package (airflow/plugins/instrument_pipeline) lives
outside the `app` package and is normally made importable by Airflow itself
via its plugins folder mechanism. For tests run outside of Airflow, we add
it to sys.path explicitly here.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "airflow", "plugins"))
