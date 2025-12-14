import os
import time

import numpy as np
import flwr as fl


class SimpleClient(fl.client.NumPyClient):
    def __init__(self, cid: str) -> None:
        self.cid = cid
        self.weights = np.zeros(2, dtype=np.float32)

    def get_parameters(self, config):
        print(f"[{self.cid}] get_parameters")
        return [self.weights]

    def fit(self, parameters, config):
        print(f"[{self.cid}] fit, received {parameters}")
        self.weights = parameters[0] + 0.1
        return [self.weights], 10, {}

    def evaluate(self, parameters, config):
        self.weights = parameters[0]
        loss = float(np.linalg.norm(self.weights))
        print(f"[{self.cid}] evaluate, loss={loss}")
        return loss, 10, {"loss": loss}


def main() -> None:
    cid = os.environ.get("CLIENT_ID", "client-unknown")
    server_address = os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")
    print(f"[{cid}] Will connect to server at {server_address}")

    time.sleep(5)  # small delay to be safe

    client = SimpleClient(cid=cid)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
