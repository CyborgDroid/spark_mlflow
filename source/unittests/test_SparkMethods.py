import unittest, sys, os
from pathlib import Path
def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent
print(os.path.dirname(os.path.abspath(__file__)))
from source.functions import SparkMethods

class TestSparkMethods(unittest.TestCase):
    def test_is_databricks_autotracking_enabled(self):
        on_dbricks = sys.env.contains("DATABRICKS_RUNTIME_VERSION")
        if on_dbricks:
            self.assertTrue(SparkMethods.is_databricks_autotracking_enabled())
        else:
            self.assertFalse(SparkMethods.is_databricks_autotracking_enabled())


if __name__ == '__main__':
    unittest.main()