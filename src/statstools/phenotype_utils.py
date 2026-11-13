"""Polars phenotype filtering and matched population references for effect tools."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)
REQUIRED_PHENO_COLUMNS = {"f2", "trait_id", "trait_value"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".md", ".tsv"}


def _read_table(path: Path) -> pl.DataFrame:
    wanted = ["f2", "trait_id", "trait_value", "sex"]
    suffix = path.suffix.lower()
    if suffix == ".xls":
        # xlrd only decodes the legacy spreadsheet; all analysis remains in Polars.
        import xlrd
        book = xlrd.open_workbook(path, on_demand=True)
        try:
            sheet = book.sheet_by_index(0)
            header = sheet.row_values(0)
            columns = [name for name in wanted if name in header]
            frame = pl.DataFrame({name: sheet.col_values(header.index(name), 1)
                                  for name in columns}, strict=False)
        finally:
            book.release_resources()
    elif suffix in {".xlsx", ".xlsm"}:
        # openpyxl is already a project dependency; no calamine dependency needed.
        frame = pl.read_excel(
            path, engine="openpyxl", infer_schema_length=None,
            schema_overrides={name: pl.String for name in wanted},
        )
    elif suffix in TEXT_SUFFIXES:
        with path.open("r", encoding="utf-8-sig") as handle:
            leading = 0
            for header in handle:
                if header.strip():
                    break
                leading += 1
            else:
                raise ValueError("表型文件为空")
        if "\t" in header:
            import csv
            names = next(csv.reader([header], delimiter="\t"))
            columns = [name for name in wanted if name in names]
            frame = pl.read_csv(path, separator="\t", columns=columns,
                                infer_schema=False, skip_lines=leading)
        else:
            # Decode whitespace-delimited text without a pandas regex parser.
            names = header.split()
            columns = [name for name in wanted if name in names]
            indices = [names.index(name) for name in columns]
            values = {name: [] for name in columns}
            with path.open("r", encoding="utf-8-sig") as handle:
                for _ in range(leading + 1):
                    next(handle)
                for line in handle:
                    if not line.strip():
                        continue
                    row = line.split()
                    if len(row) > len(names):
                        raise ValueError("空白分隔表型文件存在列数过多的记录")
                    row += [None] * (len(names) - len(row))
                    for name, index in zip(columns, indices):
                        values[name].append(row[index])
            frame = pl.DataFrame(values, schema={name: pl.String for name in columns})
    else:
        raise ValueError(f"不支持的表型文件格式 {suffix!r}。支持 xls、xlsx、xlsm、txt、md 和 tsv")
    missing = REQUIRED_PHENO_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"表型输入表缺少必要列 {sorted(missing)}")
    return frame.select([name for name in wanted if name in frame.columns])


def _number(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.String).str.strip_chars().cast(pl.Float64, strict=False)


def read_trait_phenotype(pheno_path: Path, trait_id: int) -> pl.DataFrame:
    """Keep the first valid observation per ID, preserving the previous rules."""
    frame = _read_table(pheno_path)
    for name in ("f2", "trait_id"):
        numeric = _number(name)
        integer = pl.coalesce(
            pl.col(name).cast(pl.String).str.strip_chars().cast(pl.Int64, strict=False),
            numeric.cast(pl.Int64, strict=False),
        )
        invalid = frame.filter(~(numeric.is_finite() & (numeric % 1 == 0)
                                 & integer.is_not_null()).fill_null(False))
        if invalid.height:
            raise ValueError(f"表型列 {name} 必须为整数。异常值示例 {invalid[name].head(5).to_list()}")
        frame = frame.with_columns(integer.alias(name))
    frame = frame.filter(pl.col("trait_id") == trait_id)
    if frame.is_empty():
        raise ValueError(f"表型文件中没有 trait_id {trait_id}")
    frame = frame.with_columns(_number("trait_value").alias("trait_value"))
    frame = frame.filter(pl.col("trait_value").is_finite())
    if frame.is_empty():
        raise ValueError(f"trait_id {trait_id} 没有有效的 trait_value")
    return frame.unique(subset="f2", keep="first", maintain_order=True)


def phenotype_means(frame: pl.DataFrame) -> tuple[dict[str, float], dict[str, int], str]:
    if "sex" in frame.columns:
        numeric = _number("sex")
        if frame.filter(~numeric.is_in([1.0, 2.0]).fill_null(False)).height:
            raise ValueError("表型文件的 sex 列必须为 1 或 2")
        frame = frame.with_columns(numeric.cast(pl.Int8).alias("sex"))
        source = "表型文件的 sex 列"
    else:
        frame = frame.with_columns(pl.when(pl.col("f2") % 2 == 1)
                                   .then(1).otherwise(2).alias("sex"))
        source = "f2 编号奇偶，奇数记 1"
    expressions = []
    for population, sex in (("all", None), ("male", 1), ("female", 2)):
        values = pl.col("trait_value")
        if sex is not None:
            values = values.filter(pl.col("sex") == sex)
        expressions.extend([values.mean().alias(f"{population}_mean"),
                            values.len().alias(f"{population}_count")])
    stats = frame.select(expressions).row(0, named=True)
    means = {p: float(stats[f"{p}_mean"]) if stats[f"{p}_mean"] is not None else float("nan")
             for p in ("all", "male", "female")}
    counts = {p: int(stats[f"{p}_count"]) for p in means}
    return means, counts, source


def matched_population_reference(pheno_path: Path, genotype_path: Path, trait_id: int):
    """Unique valid phenotype/genotype ID intersection, with genotype-derived sex."""
    if not genotype_path.is_file():
        raise ValueError(f"基因型文件不存在 {genotype_path}")
    phenotype = read_trait_phenotype(pheno_path, trait_id)
    scan = pl.scan_parquet(genotype_path)
    missing = {"f2", "sex"} - set(scan.collect_schema().names())
    if missing:
        raise ValueError(f"基因型输入缺少必要列 {sorted(missing)}")
    individuals = (scan.select("f2", "sex").unique().group_by("f2")
                   .agg(pl.col("sex").n_unique().alias("sex_count"),
                        pl.col("sex").first()).collect(engine="streaming"))
    conflicts = individuals.filter(pl.col("sex_count") != 1)
    if conflicts.height:
        raise ValueError(f"同一个体存在多个性别编码。个体示例 {conflicts['f2'].head(5).to_list()}")
    invalid = individuals.filter(pl.col("sex").is_null() | ~pl.col("sex").is_in([1, 2]))
    if invalid.height:
        raise ValueError(f"性别编码必须为 1 或 2。异常示例 {invalid.select('f2', 'sex').head(5).rows()}")
    # Cast the compact ID table, rather than every repeated double-window record.
    individuals = individuals.select(pl.col("f2").cast(pl.Int64), pl.col("sex"))
    matched = phenotype.drop("sex", strict=False).join(individuals, on="f2", how="inner")
    logger.info("  有效表型个体数 %s", phenotype.height)
    logger.info("  基因型个体数 %s", individuals.height)
    logger.info("  成功匹配基因型的个体数 %s", matched.height)
    logger.info("  没有基因型而被排除的表型个体数 %s", phenotype.height - matched.height)
    if matched.is_empty():
        raise ValueError("指定性状没有任何个体能够匹配基因型数据")
    means, counts, _ = phenotype_means(matched)
    return means, counts, "基因型文件的 sex 列"
