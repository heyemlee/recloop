"""Generated gRPC code (`make proto`) for recloop.modelserver.v1.

protoc emits absolute imports rooted at the proto package (`from recloop.modelserver.v1 import
...`), so the generated tree is put on sys.path once here; everything else imports from this
module.
"""

import sys
from pathlib import Path

_GEN = str(Path(__file__).resolve().parent / "gen")
if _GEN not in sys.path:
    sys.path.insert(0, _GEN)

from recloop.modelserver.v1 import model_server_pb2 as pb  # noqa: E402
from recloop.modelserver.v1 import model_server_pb2_grpc as pb_grpc  # noqa: E402

SERVICE_NAME = pb.DESCRIPTOR.services_by_name["ModelServer"].full_name

__all__ = ["SERVICE_NAME", "pb", "pb_grpc"]
