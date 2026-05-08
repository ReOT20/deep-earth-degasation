from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from deep_earth_degasation.candidates.fusion import fuse_static_dynamic_candidates
from deep_earth_degasation.morphology.static_detector import extract_static_candidates
from deep_earth_degasation.scoring import score_candidate_objects

CRS = "EPSG:32637"


def test_scored_candidates_have_separate_score_fields_and_evidence_classes() -> None:
    candidates = _candidates(
        [
            _row(
                candidate_id="static-dynamic",
                evidence_class="static_dynamic",
                static_score=0.90,
                per_feature_max={"NDMI": 3.0, "NDVI": 3.0, "BSI": 3.0, "VV_VH_ratio": 3.0},
                repeated_seasons=2,
            ),
            _row(candidate_id="static-only", evidence_class="static_only", static_score=0.95),
            _row(
                candidate_id="dynamic-only",
                evidence_class="dynamic_only",
                static_score=None,
                per_feature_max={"NDMI": 2.0},
            ),
        ]
    )

    scored = score_candidate_objects(candidates)

    assert list(scored["evidence_class"]) == ["static_dynamic", "static_only", "dynamic_only"]
    assert {"object_score", "static_score", "dynamic_score", "priority_class"}.issubset(
        scored.columns
    )
    assert scored["static_score"].iloc[0] == 0.90
    assert scored["dynamic_score"].iloc[0] > 0.0
    assert scored["object_score"].iloc[0] > 0.0
    assert scored["dominant_evidence"].iloc[0] == "cropland dynamic anomaly with static morphology"


def test_raw_dynamic_objects_without_static_score_receive_static_score_field() -> None:
    scored = score_candidate_objects(
        _candidates(
            [
                {
                    "candidate_id": "raw-dynamic",
                    "evidence_class": "dynamic_only",
                    "per_feature_max": {"NDMI": 3.0},
                    "landcover_branch": "cropland",
                    "false_positive_penalty": 0.0,
                    "missing_data_flags": (),
                    "geometry": box(0, 0, 10, 10),
                }
            ]
        )
    )

    assert scored["static_score"].iloc[0] == 0.0
    assert scored["dynamic_score"].iloc[0] > 0.0


def test_cropland_dynamic_only_object_score_uses_morphology() -> None:
    candidates = _candidates(
        [
            _row(
                candidate_id="regular-object",
                evidence_class="dynamic_only",
                static_score=None,
                per_feature_max={"NDMI": 3.0, "NDVI": 3.0},
                circularity=0.95,
                elongation=1.1,
            ),
            _row(
                candidate_id="irregular-object",
                evidence_class="dynamic_only",
                static_score=None,
                per_feature_max={"NDMI": 3.0, "NDVI": 3.0},
                circularity=0.15,
                elongation=6.0,
            ),
        ]
    )

    scored = score_candidate_objects(candidates)

    regular = scored[scored["candidate_id"] == "regular-object"].iloc[0]
    irregular = scored[scored["candidate_id"] == "irregular-object"].iloc[0]
    assert regular["object_score"] > irregular["object_score"]


def test_real_fused_rows_score_from_preserved_multisensor_dynamic_evidence() -> None:
    static_candidates = extract_static_candidates(
        [Point(0, 0).buffer(30)],
        candidate_ids=["static-match"],
    )
    dynamic_objects = _candidates(
        [
            {
                "object_id": "dynamic-match",
                "mean_anomaly": 0.15,
                "max_anomaly": 0.30,
                "per_feature_mean": {"NDMI": 2.0, "NDVI": 2.0},
                "per_feature_max": {
                    "NDMI": 3.0,
                    "NDVI": 3.0,
                    "BSI": 3.0,
                    "LST": 3.0,
                    "VV_VH_ratio": 3.0,
                },
                "source_feature_names": ("NDMI", "NDVI", "BSI", "LST", "VV_VH_ratio"),
                "dynamic_object_flags": (),
                "anomalous_dates": ("2023-05-01", "2024-05-01"),
                "source_layer_ids": ("layer-1", "layer-2"),
                "source_detection_count": 2,
                "repeated_seasons": 2,
                "geometry": Point(20, 0).buffer(25),
            }
        ]
    )

    fused = fuse_static_dynamic_candidates(static_candidates, dynamic_objects, crs=CRS)
    scored = score_candidate_objects(fused)

    row = scored.iloc[0]
    assert row["evidence_class"] == "static_dynamic"
    assert row["per_feature_max"]["NDMI"] == 3.0
    assert row["repeated_seasons"] == 2
    assert row["dynamic_score"] > 0.5
    assert row["object_score"] > 0.75


def test_multisensor_persistent_static_dynamic_candidate_outranks_static_only_geometry() -> None:
    scored = score_candidate_objects(
        _candidates(
            [
                _row(
                    candidate_id="strong-static-dynamic",
                    evidence_class="static_dynamic",
                    static_score=0.95,
                    per_feature_max={
                        "NDMI": 3.0,
                        "NDVI": 3.0,
                        "BSI": 3.0,
                        "LST": 3.0,
                        "VV_VH_ratio": 3.0,
                        "post_rain_drying": 3.0,
                    },
                    repeated_seasons=2,
                ),
                _row(
                    candidate_id="regular-static-only",
                    evidence_class="static_only",
                    static_score=0.95,
                ),
            ]
        )
    )

    strong = scored[scored["candidate_id"] == "strong-static-dynamic"].iloc[0]
    static_only = scored[scored["candidate_id"] == "regular-static-only"].iloc[0]
    assert strong["object_score"] > static_only["object_score"]
    assert static_only["dynamic_score"] == 0.0
    assert "missing_dynamic_evidence" in static_only["missing_data_flags"]


def test_false_positive_penalty_downgrades_strong_dynamic_geometry() -> None:
    base = _row(
        candidate_id="base",
        evidence_class="static_dynamic",
        static_score=0.95,
        per_feature_max={"NDMI": 3.0, "NDVI": 3.0, "BSI": 3.0, "VV_VH_ratio": 3.0},
        repeated_seasons=2,
    )
    penalized = {**base, "candidate_id": "penalized", "false_positive_penalty": 0.40}

    scored = score_candidate_objects(_candidates([base, penalized]))

    base_row = scored[scored["candidate_id"] == "base"].iloc[0]
    penalized_row = scored[scored["candidate_id"] == "penalized"].iloc[0]
    assert penalized_row["object_score"] < base_row["object_score"]
    assert base_row["priority_class"] == "A"
    assert penalized_row["priority_class"] == "C"


@pytest.mark.parametrize(
    ("branch", "expected_flag"),
    [
        ("forest", "missing_forest_branch_scoring"),
        ("mixed", "missing_mixed_branch_scoring"),
    ],
)
def test_forest_and_mixed_branches_keep_schema_with_missing_scoring_flags(
    branch: str, expected_flag: str
) -> None:
    scored = score_candidate_objects(
        _candidates(
            [
                _row(
                    candidate_id=f"{branch}-candidate",
                    evidence_class="dynamic_only",
                    static_score=None,
                    landcover_branch=branch,
                    per_feature_max={"NDMI": 2.0},
                    missing_data_flags=("existing_flag",),
                )
            ]
        )
    )

    row = scored.iloc[0]
    assert isinstance(row["object_score"], float)
    assert isinstance(row["dynamic_score"], float)
    assert row["landcover_branch"] == branch
    assert row["missing_data_flags"] == ("existing_flag", expected_flag)


def _candidates(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def _row(
    *,
    candidate_id: str,
    evidence_class: str,
    static_score: float | None,
    per_feature_max: dict[str, float] | None = None,
    repeated_seasons: int = 0,
    landcover_branch: str = "cropland",
    missing_data_flags: tuple[str, ...] = (),
    circularity: float = 0.8,
    elongation: float = 1.5,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "evidence_class": evidence_class,
        "static_score": static_score,
        "per_feature_max": per_feature_max or {},
        "repeated_seasons": repeated_seasons,
        "landcover_branch": landcover_branch,
        "circularity": circularity,
        "elongation": elongation,
        "false_positive_penalty": 0.0,
        "missing_data_flags": missing_data_flags,
        "geometry": box(0, 0, 10, 10),
    }
