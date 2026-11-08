"""A LightGBM booster as an ONNX TreeEnsembleRegressor (raw scores), for the model server.

The model server runs ONNX Runtime and not LightGBM (like the SASRec encoder, design §19), so
the bundle carries the ranker as an ONNX tree ensemble. The conversion keeps LightGBM's
decisions exactly for float32 inputs:

  * a split `x <= t` has a double threshold t; ONNX stores float32. For a float32 x,
    x <= t  <=>  x <= the largest float32 not above t, so that value is stored (rounding to the
    nearest float32 instead flips inputs equal to the next float32 above t);
  * missing values: with missing type NaN a NaN follows `default_left`; with type None LightGBM
    reads a NaN as 0.0, so it goes where 0.0 goes. Type Zero (zero_as_missing) and categorical
    splits are not used here and are refused;
  * LightGBM's predictor reads any |x| <= 1e-35 as exactly 0 (its sparse "zero" threshold), so
    the inputs get the same treatment before ONNX sees them (`lightgbm_inputs`).

The bundle builder compares ONNX and LightGBM on every training row before publishing.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

ONNX_ML_OPSET = 3
ONNX_OPSET = 18
ZERO_THRESHOLD = 1e-35  # LightGBM's kZeroThreshold


def lightgbm_inputs(x: np.ndarray) -> np.ndarray:
    """float32 features as LightGBM's predictor reads them: tiny values become 0, NaN stays."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.abs(x) <= ZERO_THRESHOLD, np.float32(0), x)


def float32_floor(t: float) -> float:
    """The largest float32 value <= t."""
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def booster_to_onnx(booster, num_features: int) -> onnx.ModelProto:
    dump = booster.dump_model()
    a: dict[str, list] = {
        k: []
        for k in (
            "nodes_treeids",
            "nodes_nodeids",
            "nodes_featureids",
            "nodes_values",
            "nodes_modes",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
            "target_treeids",
            "target_nodeids",
            "target_ids",
            "target_weights",
        )
    }

    def add(tree: int, nid: int, feature: int, value: float, mode: str, t: int, f: int, miss: int):
        a["nodes_treeids"].append(tree)
        a["nodes_nodeids"].append(nid)
        a["nodes_featureids"].append(feature)
        a["nodes_values"].append(value)
        a["nodes_modes"].append(mode)
        a["nodes_truenodeids"].append(t)
        a["nodes_falsenodeids"].append(f)
        a["nodes_missing_value_tracks_true"].append(miss)

    for tree_idx, tree in enumerate(dump["tree_info"]):
        counter = [0]

        def visit(node: dict, tree_idx: int = tree_idx, counter: list = counter) -> int:
            nid = counter[0]
            counter[0] += 1
            if "split_index" not in node:  # a leaf
                add(tree_idx, nid, 0, 0.0, "LEAF", 0, 0, 0)
                a["target_treeids"].append(tree_idx)
                a["target_nodeids"].append(nid)
                a["target_ids"].append(0)
                a["target_weights"].append(float(node["leaf_value"]))
                return nid
            if node["decision_type"] != "<=":
                raise ValueError(f"unsupported split {node['decision_type']!r} (categorical?)")
            t = float(node["threshold"])
            missing = node["missing_type"]
            if missing == "NaN":
                miss = int(bool(node["default_left"]))
            elif missing == "None":
                miss = int(0.0 <= t)  # NaN is read as 0.0
            else:
                raise ValueError(f"unsupported missing type {missing!r}")
            pos = len(a["nodes_nodeids"])
            add(tree_idx, nid, int(node["split_feature"]), float32_floor(t), "BRANCH_LEQ", -1, -1, miss)
            left = visit(node["left_child"])
            right = visit(node["right_child"])
            a["nodes_truenodeids"][pos] = left
            a["nodes_falsenodeids"][pos] = right
            return nid

        visit(tree["tree_structure"])

    node = helper.make_node(
        "TreeEnsembleRegressor",
        ["features"],
        ["score"],
        domain="ai.onnx.ml",
        n_targets=1,
        aggregate_function="SUM",
        post_transform="NONE",
        base_values=[0.0],
        **a,
    )
    graph = helper.make_graph(
        [node],
        "ranker",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, num_features])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET), helper.make_opsetid("ai.onnx.ml", ONNX_ML_OPSET)],
        producer_name="recloop-ml",
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def onnx_scores(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    """Run the converted model with ONNX Runtime (single thread); (n, F) float32 -> (n,)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
    x = lightgbm_inputs(x)
    out = []
    for lo in range(0, len(x), 65536):
        out.append(sess.run(None, {"features": np.ascontiguousarray(x[lo : lo + 65536])})[0][:, 0])
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
