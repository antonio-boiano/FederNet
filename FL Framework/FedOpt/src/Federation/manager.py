from FedOpt.src.Federation.ASOFed.ASOFed import ASOFed, ASOFedModel
from FedOpt.src.Federation.AsyncFedED.AsyncFedED import AsyncFedED, AsyncFedEDModel
from FedOpt.src.Federation.FedAsync.FedAsync import FedAsync, FedAsyncModel
from FedOpt.src.Federation.FedAverage.FedAvg import FedAvg, FedAvgModel
from FedOpt.src.Federation.SCAFFOLD.SCAFFOLD import Scaffold, ScaffoldModel
from FedOpt.src.Federation.FedDyn.FedDyn import FedDyn, FedDynModel
from FedOpt.src.Federation.FedAvgN.FedAvgN import FedAvgN, FedAvgNModel
from FedOpt.src.Federation.AsyncFedED.AsyncFedED import AsyncFedED, AsyncFedEDModel
from FedOpt.src.Federation.Unweighted.Unweighted import Unweighted, UnweightedModel

# server side aggregation
def federation_manager(config=None):
    method = config["fl_method"]
    if method == "FedAvg":
        return FedAvg(config)
    elif method == "FedAvgN":
        return FedAvgN(config)
    elif method == "SCAFFOLD":
        return Scaffold(config)
    elif method == "FedDyn":
        return FedDyn(config)
    elif method == "FedAsync":
        return FedAsync(config)
    elif method == "AsyncFedED":
        return AsyncFedED(config)
    elif method == "ASOFed":
        return ASOFed(config)
    elif method == "Unweighted":
        return Unweighted(config)
    else:
        raise ValueError("Unsupported federation method: " + config["fl_method"])


# client side model optimization
def model_manager(config=None):
    """
    Return the models manager for the client side
    """
    method = config["fl_method"]
    if method == "FedAvg":
        return FedAvgModel(config)
    elif method == "FedAvgN":
        return FedAvgNModel(config)
    elif method == "SCAFFOLD":
        return ScaffoldModel(config)
    elif method == "FedDyn":
        return FedDynModel(config)
    elif method == "FedAsync":
        return FedAsyncModel(config)
    elif method == "AsyncFedED":
        return AsyncFedEDModel(config)
    elif method == "ASOFed":
        return ASOFedModel(config)
    elif method == "Unweighted":
        return UnweightedModel(config)
    else:
        raise ValueError("Invalid parameter: " + config["fl_method"])
