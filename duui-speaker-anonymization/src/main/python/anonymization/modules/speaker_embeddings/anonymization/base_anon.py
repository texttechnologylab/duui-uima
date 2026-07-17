import torch
from typing import Union


class BaseAnonymizer:
    """
    Base class for speaker embedding anonymizers, defining the API,
    that consists of the following methods:
    - anonymize_embeddings
    - to
    """
    def __init__(
        self,
        vec_type: str,
        device: Union[str, torch.device, int, None],
        suffix: str,
        **kwargs,
    ):
        assert suffix[0] == "_", "Suffix must be a string and start with an underscore."

        # Base class for speaker embedding anonymization.
        self.vec_type = vec_type
        self.suffix = suffix

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # ensure dumpability
        self.kwargs = kwargs
        self.kwargs["vec_type"] = self.vec_type
        self.kwargs["device"] = str(self.device)
        self.kwargs["suffix"] = self.suffix

    def anonymize_embeddings(self, speaker_embeddings: torch.Tensor, emb_level: str = "spk") -> torch.Tensor:
        # Template method for anonymizing a dataset. Not implemented.
        raise NotImplementedError("anonymize_data")

    def to(self, device):
        self.device = device

