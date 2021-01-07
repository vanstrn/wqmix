REGISTRY = {}

from .rnn_agent import RNNAgent,CRNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent,ConvCentralRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["crnn"] = CRNNAgent
REGISTRY["ff"] = FFAgent

REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["central_crnn"] = ConvCentralRNNAgent
