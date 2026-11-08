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

class SourceHit(_message.Message):
    __slots__ = ("source", "rank", "score")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    source: str
    rank: int
    score: float
    def __init__(self, source: _Optional[str] = ..., rank: _Optional[int] = ..., score: _Optional[float] = ...) -> None: ...

class Candidate(_message.Message):
    __slots__ = ("item_id", "hits")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    item_id: int
    hits: _containers.RepeatedCompositeFieldContainer[SourceHit]
    def __init__(self, item_id: _Optional[int] = ..., hits: _Optional[_Iterable[_Union[SourceHit, _Mapping]]] = ...) -> None: ...

class AdCandidate(_message.Message):
    __slots__ = ("item_id", "relevance")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    RELEVANCE_FIELD_NUMBER: _ClassVar[int]
    item_id: int
    relevance: float
    def __init__(self, item_id: _Optional[int] = ..., relevance: _Optional[float] = ...) -> None: ...

class RankRequest(_message.Message):
    __slots__ = ("history", "candidates", "ads", "pin_version")
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    ADS_FIELD_NUMBER: _ClassVar[int]
    PIN_VERSION_FIELD_NUMBER: _ClassVar[int]
    history: _containers.RepeatedScalarFieldContainer[int]
    candidates: _containers.RepeatedCompositeFieldContainer[Candidate]
    ads: _containers.RepeatedCompositeFieldContainer[AdCandidate]
    pin_version: str
    def __init__(self, history: _Optional[_Iterable[int]] = ..., candidates: _Optional[_Iterable[_Union[Candidate, _Mapping]]] = ..., ads: _Optional[_Iterable[_Union[AdCandidate, _Mapping]]] = ..., pin_version: _Optional[str] = ...) -> None: ...

class RankResponse(_message.Message):
    __slots__ = ("model_version", "feature_names", "features", "scores", "ad_features", "ad_pctr", "ranker", "pctr_model")
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    FEATURE_NAMES_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    AD_FEATURES_FIELD_NUMBER: _ClassVar[int]
    AD_PCTR_FIELD_NUMBER: _ClassVar[int]
    RANKER_FIELD_NUMBER: _ClassVar[int]
    PCTR_MODEL_FIELD_NUMBER: _ClassVar[int]
    model_version: str
    feature_names: _containers.RepeatedScalarFieldContainer[str]
    features: _containers.RepeatedScalarFieldContainer[float]
    scores: _containers.RepeatedScalarFieldContainer[float]
    ad_features: _containers.RepeatedScalarFieldContainer[float]
    ad_pctr: _containers.RepeatedScalarFieldContainer[float]
    ranker: str
    pctr_model: str
    def __init__(self, model_version: _Optional[str] = ..., feature_names: _Optional[_Iterable[str]] = ..., features: _Optional[_Iterable[float]] = ..., scores: _Optional[_Iterable[float]] = ..., ad_features: _Optional[_Iterable[float]] = ..., ad_pctr: _Optional[_Iterable[float]] = ..., ranker: _Optional[str] = ..., pctr_model: _Optional[str] = ...) -> None: ...
