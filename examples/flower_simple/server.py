import flwr as fl
from flwr.server.strategy import FedAvg


def main() -> None:
    # Simple FedAvg strategy, require 2 clients
    strategy = FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )

    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=600),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
