from __future__ import annotations

from typing import cast

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.context.fields import (
    ContextAssignmentError,
    _require_column,
    _require_compatible_metric_crs,
)


def assign_landcover_context(
    objects: gpd.GeoDataFrame,
    landcover: gpd.GeoDataFrame,
    *,
    classes: dict[str, list[str]],
    class_column: str = "landcover_class",
    mixed_candidate_min_classes: int = 2,
) -> gpd.GeoDataFrame:
    """Attach land-cover branch and branch proportions to object geometries."""
    _require_compatible_metric_crs(
        objects,
        landcover,
        left_name="objects",
        right_name="landcover",
    )
    _require_column(landcover, class_column, "landcover")

    class_to_branch = _class_to_branch(classes)
    output = objects.copy()
    branches: list[str] = []
    dominant_branches: list[str] = []
    proportions_list: list[dict[str, float]] = []
    context_flags: list[list[str]] = []

    for geometry in output.geometry:
        object_geometry = cast(BaseGeometry, geometry)
        object_area = float(object_geometry.area)
        proportions = _branch_proportions(
            object_geometry,
            object_area=object_area,
            landcover=landcover,
            class_column=class_column,
            class_to_branch=class_to_branch,
        )
        flags: list[str] = []
        if not proportions:
            branches.append("unknown")
            dominant_branches.append("unknown")
            proportions_list.append({})
            context_flags.append(["missing_landcover_context"])
            continue

        dominant_branch = max(proportions, key=lambda branch: proportions[branch])
        branch = "mixed" if len(proportions) >= mixed_candidate_min_classes else dominant_branch
        if branch == "mixed":
            flags.append("mixed_landcover")

        branches.append(branch)
        dominant_branches.append(dominant_branch)
        proportions_list.append(proportions)
        context_flags.append(flags)

    output["landcover_branch"] = branches
    output["dominant_landcover_branch"] = dominant_branches
    output["landcover_proportions"] = proportions_list
    output["landcover_context_flags"] = context_flags
    return output


def _class_to_branch(classes: dict[str, list[str]]) -> dict[str, str]:
    return {
        landcover_class: branch
        for branch, branch_classes in classes.items()
        for landcover_class in branch_classes
    }


def _branch_proportions(
    object_geometry: BaseGeometry,
    *,
    object_area: float,
    landcover: gpd.GeoDataFrame,
    class_column: str,
    class_to_branch: dict[str, str],
) -> dict[str, float]:
    if object_area <= 0:
        raise ContextAssignmentError("Object geometry must have positive area.")

    branch_areas: dict[str, float] = {}
    for _, landcover_row in landcover.iterrows():
        branch = class_to_branch.get(str(landcover_row[class_column]))
        if branch is None:
            continue

        landcover_geometry = cast(BaseGeometry, landcover_row.geometry)
        intersection_area = float(object_geometry.intersection(landcover_geometry).area)
        if intersection_area <= 0:
            continue

        branch_areas[branch] = branch_areas.get(branch, 0.0) + intersection_area

    return {branch: area / object_area for branch, area in sorted(branch_areas.items()) if area > 0}
