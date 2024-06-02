from torch import Tensor, nn, zeros


class BetClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_size: int,
        output_size: int = 2,
    ) -> None:
        super().__init__()

        self._batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._dense = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self._classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> Tensor:
        h0 = zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self._dense(x, h0)

        out = self._classifier(out[:, -1, :])

        return out
