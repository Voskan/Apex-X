import importlib

import apex_x
from apex_x import (
    ApexXConfig,
    ApexXModel,
    BudgetController,
    BudgetControllerProtocol,
    Exporter,
    Router,
    RouterProtocol,
    RuntimeAdapterProtocol,
    TilePack,
    TilePackerProtocol,
    TileUnpack,
    TileUnpackerProtocol,
)


def test_public_api_symbols_exist() -> None:
    assert ApexXConfig is not None
    assert ApexXModel is not None
    assert Router is not None
    assert RouterProtocol is not None
    assert BudgetController is not None
    assert BudgetControllerProtocol is not None
    assert TilePack is not None
    assert TilePackerProtocol is not None
    assert TileUnpack is not None
    assert TileUnpackerProtocol is not None
    assert RuntimeAdapterProtocol is not None
    assert Exporter is not None


def test_public_api_all_contains_required_names() -> None:
    required = {
        "ApexXConfig",
        "ApexXModel",
        "Router",
        "RouterProtocol",
        "BudgetController",
        "BudgetControllerProtocol",
        "TilePack",
        "TilePackerProtocol",
        "TileUnpack",
        "TileUnpackerProtocol",
        "RuntimeAdapterProtocol",
        "Exporter",
    }
    assert required.issubset(set(apex_x.__all__))


def test_subpackage_imports_smoke() -> None:
    modules = [
        "apex_x.config",
        "apex_x.model",
        "apex_x.tiles",
        "apex_x.routing",
        "apex_x.losses",
        "apex_x.train",
        "apex_x.infer",
        "apex_x.data",
        "apex_x.export",
        "apex_x.bench",
        "apex_x.runtime",
        "apex_x.utils",
    ]
    for module in modules:
        assert importlib.import_module(module) is not None
