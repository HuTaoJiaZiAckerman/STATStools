"""Bound effect calculation by focal A windows, then stream parts to one file."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)
ROW_GROUP_SIZE = 65_536


def iter_window_batches(data: pl.LazyFrame, trait_id: int, batch_size: int):
    if batch_size < 1:
        raise ValueError("--window_batch_size 必须为正整数")
    # Narrow scan: only retain distinct A coordinates and the trait inventory.
    inventory = data.select("chra", "windowa", "trait_id").unique().collect(engine="streaming")
    traits = inventory["trait_id"].unique().to_list()
    if traits != [trait_id]:
        raise ValueError(f"输入文件中的 trait_id 为 {traits}，与参数 {trait_id} 不一致")
    if inventory.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("A 窗口坐标与 trait_id 不得为空")
    coordinates = inventory.select("chra", "windowa").sort("chra", "windowa")
    logger.info("A 窗口总数 %s；每批最多 %s 个 A 窗口", coordinates.height, batch_size)
    batch_index = 0
    for chromosome in coordinates.partition_by("chra", maintain_order=True):
        chra = chromosome["chra"][0]
        for start in range(0, chromosome.height, batch_size):
            windows = chromosome["windowa"].slice(start, batch_size)
            batch = data.filter((pl.col("chra") == chra)
                                & pl.col("windowa").is_between(windows[0], windows[-1]))
            frame = batch.collect(engine="streaming")
            logger.info("批次 %s：chr=%s，window=%s–%s，A 窗口数=%s，输入行数=%s",
                        batch_index + 1, chra, windows[0], windows[-1], len(windows), frame.height)
            yield batch_index, frame
            del frame, batch
            batch_index += 1


def write_part(result: pl.LazyFrame, path: Path) -> None:
    result.sink_parquet(path, compression="zstd", maintain_order=True,
                        row_group_size=ROW_GROUP_SIZE, engine="streaming")


def consolidate_parts(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("没有可写出的结果批次")
    # No concat collect, global sort, or parallel scan of all temporary parts.
    pl.scan_parquet(paths, parallel="none", low_memory=True).sink_parquet(
        output, compression="zstd", maintain_order=True,
        row_group_size=ROW_GROUP_SIZE, engine="streaming",
    )
