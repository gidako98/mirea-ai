from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    *,
    high_cardinality_unique_threshold: int = 50,
    high_cardinality_unique_share_threshold: float = 0.5,
    zero_share_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Эвристики «качества» данных.

    Базовые флаги опираются на DatasetSummary + missing_table().
    Дополнительные (про константность/дубликаты/кардинальность) требуют исходный df.

    Возвращаемый словарь предназначен для использования и в CLI-отчёте, и в тестах.
    """
    flags: Dict[str, Any] = {}

    # --- Базовые эвристики (как в S03) ---
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # --- Новые эвристики (HW03) ---
    constant_columns: List[str] = []
    high_cardinality_categoricals: List[str] = []
    id_duplicate_columns: List[str] = []
    zero_heavy_columns: List[str] = []
    duplicate_rows_count = 0

    if df is not None and not df.empty:
        # 1) Константные колонки (все НЕ-NA значения одинаковые)
        for name in df.columns:
            s = df[name].dropna()
            if s.empty:
                continue
            if int(s.nunique(dropna=True)) == 1:
                constant_columns.append(name)

        # 2) Высокая кардинальность категориальных признаков
        #    Либо абсолютный порог по количеству уникальных, либо доля уникальных > порога.
        for name in df.columns:
            s = df[name]
            if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                nunique = int(s.nunique(dropna=True))
                share = float(nunique / summary.n_rows) if summary.n_rows > 0 else 0.0
                if nunique >= high_cardinality_unique_threshold or share >= high_cardinality_unique_share_threshold:
                    high_cardinality_categoricals.append(name)

        # 3) Подозрительные дубликаты в id-полях
        #    Эвристика: если имя колонки похоже на идентификатор (id, *_id, *id*),
        #    то ожидаем уникальность среди непустых значений.
        for name in df.columns:
            low = str(name).lower()
            if low == "id" or low.endswith("_id") or "id" in low:
                s = df[name].dropna()
                if len(s) == 0:
                    continue
                if s.duplicated().any():
                    id_duplicate_columns.append(name)

        # 4) Дубли строк целиком
        duplicate_rows_count = int(df.duplicated().sum())

        # 5) Очень много нулей в числовых колонках
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            for name in numeric_df.columns:
                s = numeric_df[name]
                if len(s) == 0:
                    continue
                share_zeros = float((s == 0).mean())
                if share_zeros >= zero_share_threshold:
                    zero_heavy_columns.append(name)

    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns

    flags["has_high_cardinality_categoricals"] = len(high_cardinality_categoricals) > 0
    flags["high_cardinality_categoricals"] = high_cardinality_categoricals

    flags["has_suspicious_id_duplicates"] = len(id_duplicate_columns) > 0
    flags["id_duplicate_columns"] = id_duplicate_columns

    flags["has_duplicate_rows"] = duplicate_rows_count > 0
    flags["duplicate_rows_count"] = duplicate_rows_count

    flags["has_many_zero_values"] = len(zero_heavy_columns) > 0
    flags["zero_heavy_columns"] = zero_heavy_columns

    # --- Интегральный score ---
    # Чем больше проблем, тем ниже score.
    score = 1.0
    score -= max_missing_share
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.2
    if flags["has_duplicate_rows"]:
        score -= 0.05
    if flags["has_many_zero_values"]:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
