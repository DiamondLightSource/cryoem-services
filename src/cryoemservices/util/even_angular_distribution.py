from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Generator, List, NamedTuple, Set

import mrcfile
import numpy as np
import polars as pl
import scipy
import starfile


class ClassImage(NamedTuple):
    classification_job: str
    index: int
    class_number: int
    particle_count: int


@lru_cache(maxsize=50)
def get_class_img_set(mrc_path: os.PathLike) -> np.array:
    return mrcfile.read(mrc_path)


def angular_cross_correlation(
    img: np.array, comp_img: np.array, angular_step: int = 10
) -> float:
    ang_scores = []
    for ang in range(0, 360, angular_step):
        corr = scipy.signal.correlate(
            scipy.ndimage.rotate(img, ang, reshape=False), comp_img, mode="same"
        )
        ang_scores.append(np.max(corr))
    return np.max(ang_scores)


def group_classes(
    particle_data_glob: Generator[Path, None, None], correlation_cutoff: float = 0.8
) -> List[Set[ClassImage]]:
    particle_data = None
    groups = []
    for f in particle_data_glob:
        data = pl.from_pandas(starfile.read(f)["particles"])
        if (f.parent / "class_averages.star").is_file():
            class_avgs_data = pl.from_pandas(
                starfile.read(f.parent / "class_averages.star")
            )
            class_avgs_data = class_avgs_data.with_column(
                pl.col("rlnClassImage")
                .str.split("@")
                .list[0]
                .cast(pl.Int64)
                .alias("rlnClassNumber")
            )
            data = data.join(class_avgs_data, on="rlnClassNumber")
            base_path = f.parent.parent.parent
            data = data.with_columns(
                (
                    pl.concat_str(
                        pl.lit(str(base_path)),
                        pl.lit("/"),
                        pl.col("rlnReferenceImage").str.split("@").list[1],
                        pl.lit(":"),
                        pl.col("rlnClassNumber").cast(pl.String),
                    )
                ).alias("classification_job_index")
            )
        else:
            data = data.with_columns(
                (
                    pl.concat_str(
                        pl.lit(str(f.parent)),
                        pl.lit(":"),
                        pl.col("rlnClassNumber").cast(pl.String),
                    )
                ).alias("classification_job_index")
            )
        if particle_data is None:
            particle_data = data
        else:
            particle_data = pl.concat([particle_data, data])
    for i, class_img_labels in enumerate(
        particle_data.group_by("classification_job_index", maintain_order=True)
        .len()
        .rows()
    ):
        mrc_path = class_img_labels[0].split(":")[0]
        index = int(class_img_labels[0].split(":")[1]) - 1
        img_group = get_class_img_set(mrc_path)
        img = img_group[index]
        img = (
            (img - np.mean(img)) / (np.std(img) * img.shape[0] * img.shape[1])
            if np.std(img)
            else np.zeros(img.shape)
        )
        for j, comp_img_labels in enumerate(
            particle_data.group_by("classification_job_index", maintain_order=True)
            .len()
            .rows()[i:]
        ):
            if i == i + j:
                continue
            mrc_path_comp = comp_img_labels[0].split(":")[0]
            comp_index = int(comp_img_labels[0].split(":")[1]) - 1
            img_group_comp = mrcfile.read(mrc_path_comp)
            comp_img = img_group_comp[comp_index]
            comp_img = (
                (comp_img - np.mean(comp_img)) / (np.std(comp_img))
                if np.std(comp_img)
                else np.zeros(comp_img.shape)
            )
            if angular_cross_correlation(img, comp_img) >= correlation_cutoff:
                for g in groups:
                    if index in g or comp_index in g:
                        g.add(
                            ClassImage(
                                classification_job=Path(mrc_path).parent.name,
                                index=index,
                                class_number=index + 1,
                                particle_count=class_img_labels[1],
                            )
                        )
                        g.add(
                            ClassImage(
                                classification_job=Path(mrc_path_comp).parent.name,
                                index=comp_index,
                                class_number=comp_index + 1,
                                particle_count=comp_img_labels[1],
                            )
                        )
                        break
                else:
                    groups.append(
                        {
                            ClassImage(
                                classification_job=Path(mrc_path).parent.name,
                                index=index,
                                class_number=index + 1,
                                particle_count=class_img_labels[1],
                            ),
                            ClassImage(
                                classification_job=Path(mrc_path_comp).parent.name,
                                index=comp_index,
                                class_number=comp_index + 1,
                                particle_count=comp_img_labels[1],
                            ),
                        }
                    )
                break
        else:
            if not any(index in g for g in groups):
                groups.append(
                    {
                        ClassImage(
                            classification_job=Path(mrc_path).parent.name,
                            index=index,
                            class_number=index + 1,
                            particle_count=class_img_labels[1],
                        )
                    }
                )
    return groups
