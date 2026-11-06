"""Shared trait filtering and population means for conditional phenotype effects."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_PHENO_COLUMNS = {"f2", "trait_id", "trait_value"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".md", ".tsv"}

def read_trait_phenotype(pheno_path: Path, trait_id: int) -> pd.DataFrame:
    """Read one trait from the phenotype file used to build the aggregation input."""
    suffix = pheno_path.suffix.lower()
    wanted = REQUIRED_PHENO_COLUMNS | {"sex"}
    if suffix in EXCEL_SUFFIXES:
        frame = pd.read_excel(pheno_path, usecols=lambda column: column in wanted)
    elif suffix in TEXT_SUFFIXES:
        with pheno_path.open("r", encoding="utf-8-sig") as handle:
            header = next((line for line in handle if line.strip()), "")
        separator = "\t" if "\t" in header else r"\s+"
        engine = "c" if separator == "\t" else "python"
        frame = pd.read_csv(
            pheno_path, sep=separator, engine=engine,
            usecols=lambda column: column in wanted,
        )
    else:
        raise ValueError(
            f"不支持的表型文件格式 {suffix!r}。支持 xls、xlsx、xlsm、txt、md 和 tsv"
        )

    missing = REQUIRED_PHENO_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"表型输入表缺少必要列 {sorted(missing)}")

    kept = ["f2", "trait_id", "trait_value"]
    if "sex" in frame.columns:
        kept.append("sex")
    frame = frame[kept].copy()

    for column in ("f2", "trait_id"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | (numeric % 1 != 0)
        if invalid.any():
            examples = frame.loc[invalid, column].head(5).tolist()
            raise ValueError(f"表型列 {column} 必须为整数。异常值示例 {examples}")
        frame[column] = numeric.astype("int64")

    frame = frame.loc[frame["trait_id"] == trait_id].copy()
    if frame.empty:
        raise ValueError(f"表型文件中没有 trait_id {trait_id}")

    frame["trait_value"] = pd.to_numeric(frame["trait_value"], errors="coerce")
    valid = frame["trait_value"].notna() & np.isfinite(frame["trait_value"])
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(f"trait_id {trait_id} 没有有效的 trait_value")

    frame = frame.drop_duplicates(subset=["f2"], keep="first")
    return frame.reset_index(drop=True)


def phenotype_means(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int], str]:
    """Return all, male, and female phenotype means together with sample sizes."""
    if "sex" in frame.columns:
        sex = pd.to_numeric(frame["sex"], errors="coerce")
        if sex.isna().any() or not sex.isin([1, 2]).all():
            raise ValueError("表型文件的 sex 列必须为 1 或 2")
        source = "表型文件的 sex 列"
    else:
        sex = pd.Series(np.where(frame["f2"] % 2 == 1, 1, 2), index=frame.index)
        source = "f2 编号奇偶，奇数记 1"

    values = frame["trait_value"]
    masks = {
        "all": pd.Series(True, index=frame.index),
        "male": sex == 1,
        "female": sex == 2,
    }
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, mask in masks.items():
        subset = values.loc[mask]
        counts[name] = int(len(subset))
        means[name] = float(subset.mean()) if len(subset) else float("nan")
    return means, counts, source
