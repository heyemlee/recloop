from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RetrieveRequest(_message.Message):
    __slots__ = ("history", "k", "ad_item_ids", "pin_version")
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    AD_ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    PIN_VERSION_FIELD_NUMBER: _ClassVar[int]
    history: _containers.RepeatedScalarFieldContainer[int]
    k: int
    ad_item_ids: _containers.RepeatedScalarFieldContainer[int]
    pin_version: str
    def __init__(self, history: _Optional[_Iterable[int]] = ..., k: _Optional[int] = ..., ad_item_ids: _Optional[_Iterable[int]] = ..., pin_version: _Optional[str] = ...) -> None: ...

class RetrieveResponse(_message.Message):
    __slots__ = ("model_version", "organic", "ads")
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ORGANIC_FIELD_NUMBER: _ClassVar[int]
    ADS_FIELD_NUMBER: _ClassVar[int]
    model_version: str
    organic: _containers.RepeatedCompositeFieldContainer[Scored]
    ads: _containers.RepeatedCompositeFieldContainer[Scored]
    def __init__(self, model_version: _Optional[str] = ..., organic: _Optional[_Iterable[_Union[Scored, _Mapping]]] = ..., ads: _Optional[_Iterable[_Union[Scored, _Mapping]]] = ...) -> None: ...

class Scored(_message.Message):
    __slots__ = ("item_id", "score")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    item_id: int
    score: float
    def __init__(self, item_id: _Optional[int] = ..., score: _Optional[float] = ...) -> None: ...

class RankRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RankResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
