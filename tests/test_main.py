import pytest
import main

def test_tabularautoml_instantiation():
    # Verify that the class TabularAutoML is inspectable and loadable
    assert hasattr(main, 'TabularAutoML')

