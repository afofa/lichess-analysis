import bz2
import chess.pgn
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import requests

from typing import Any, Dict, Generator, List, Optional, TextIO, Tuple


def get_download_links_for_variant(variant: str) -> List[str]:
    r = requests.get(f"https://database.lichess.org/{variant}/list.txt")
    links = r.text.split("\n")
    return links


def download_games_for_variant(
    variant: str,
    output_folder: str,
    since: Optional[str] = None,
    to: Optional[str] = None,
    verbose: bool = False,
) -> None:
    def get_year_and_month_from_str(date_str: str) -> Tuple[str, str]:
        splitted = date_str.split("-")
        return splitted[0], splitted[1]

    def is_valid_date(
        date: Tuple[str, str],
        since: Tuple[int, int],
        to: Tuple[int, int],
    ) -> bool:
        return since <= date < to

    def download_and_save_single_file(
        date: Tuple[int, int],
        url: str,
        output_path: str,
    ) -> None:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"Error fetching {url}")
            print(f"Cannot fetch and save data, status code {r.status_code}")

    if since is not None:
        since_year, since_month = get_year_and_month_from_str(since)
    else:
        since_year, since_month = "0000", "00"

    if to is not None:
        to_year, to_month = get_year_and_month_from_str(to)
    else:
        to_year, to_month = "9999", "99"

    links = get_download_links_for_variant(variant)
    dates = map(
        lambda x: tuple(x.split("_")[-1].split(".")[0].split("-")),
        links
    )

    valid_dates_links_dct = {
        date: link for date, link in zip(dates, links) if is_valid_date(
            date,
            (since_year, since_month),
            (to_year, to_month)
        )
    }

    os.makedirs(output_folder, exist_ok=True)

    for i, (date, url) in enumerate(valid_dates_links_dct.items(), start=1):
        if verbose:
            print(f"{i}/{len(valid_dates_links_dct)}")

        download_and_save_single_file(
            date,
            url,
            os.path.join(output_folder, f"{date[0]}_{date[1]}.bz2"),
        )


def decompress_and_save_bz2_files(
    input_dir_path: str,
    output_dir_path: str,
    verbose: bool = False,
) -> None:
    os.makedirs(output_dir_path, exist_ok=True)

    bz2_files = [
        f for f in os.listdir(input_dir_path) if f.endswith(".bz2")
    ]

    for i, filename in enumerate(bz2_files, start=1):
        if verbose:
            print(f"{i}/{len(bz2_files)}")

        compressed_path = os.path.join(input_dir_path, filename)
        decompressed_filename = filename[:-4] + ".pgn"
        decompressed_path = os.path.join(
            output_dir_path, decompressed_filename
        )

        with open(compressed_path, "rb") as f_old:
            with open(decompressed_path, "wb") as f_new:
                data = bz2.decompress(f_old.read())
                f_new.write(data)


def read_pgn_file(
    path: str,
) -> TextIO:
    pgn = open(path)
    return pgn


def filter_pgn_file_for_offsets(
    pgn: TextIO,
    filters: Dict[str, List[Any]] = {},
) -> Generator[int, None, None]:
    while True:
        offset = pgn.tell()

        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break

        for key, vals in filters.items():
            if headers.get(key) not in vals:
                break
        else:
            yield offset


def filter_pgn_file_for_headers(
    pgn: TextIO,
    filters: Dict[str, List[Any]] = {},
) -> Generator[chess.pgn.Headers, None, None]:
    while True:
        _ = pgn.tell()

        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break

        for key, vals in filters.items():
            if headers.get(key) not in vals:
                break
        else:
            yield headers


def filter_pgn_file_for_games(
    pgn: TextIO,
    filters: Dict[str, List[Any]] = {},
    verbose_period: Optional[int] = None,
) -> Generator[chess.pgn.Game, None, None]:
    count = 0
    while True:
        offset = pgn.tell()

        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break

        for key, vals in filters.items():
            if headers.get(key) not in vals:
                break
        else:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            count += 1
            if verbose_period is not None and count % verbose_period == 0:
                print(f"{count} games processed")
            yield game


def make_df_from_generator(
    generator: Generator[Any, None, None],
) -> pd.DataFrame:
    df = pd.DataFrame(generator)
    return df


def make_df_from_headers(
    headers_gen: Generator[chess.pgn.Headers, None, None],
) -> pd.DataFrame:
    # TODO: operations specific to headers data
    df = make_df_from_generator(headers_gen)
    return df


def make_df_from_games(
    game_gen: Generator[chess.pgn.Game, None, None],
) -> pd.DataFrame:
    # TODO: operations specific to game data
    row_gen = (
        {
            **game.headers,
            "PlyCount": len(list(game.mainline_moves())),
        } for game in game_gen
    )
    df = make_df_from_generator(row_gen)
    return df


def scatter_plot(
    x_series: pd.Series,
    y_series: pd.Series,
    output_path: str,
    **kwargs,
) -> None:
    plt.figure()
    plt.scatter(x_series, y_series, **kwargs)
    plt.savefig(output_path, bbox_inches="tight")


def scatter_plot_of_rating_difference_and_ply_count(
    df: pd.DataFrame,
    **kwargs,
) -> None:
    x_series = df["WhiteElo"] - df["BlackElo"]
    y_series = df["PlyCount"]
    scatter_plot(
        x_series,
        y_series,
        "scatter_plot_of_rating_difference_and_ply_count.png",
        **kwargs
    )


def scatter_plot_of_white_elo_and_black_elo(
    df: pd.DataFrame,
    **kwargs,
) -> None:
    x_series = df["WhiteElo"]
    y_series = df["BlackElo"]
    scatter_plot(
        x_series,
        y_series,
        "scatter_plot_of_white_elo_and_black_elo.png",
        **kwargs
    )


def scatter_plot_of_white_elo_and_white_rating_diff(
    df: pd.DataFrame,
    **kwargs,
) -> None:
    x_series = df["WhiteElo"]
    y_series = df["WhiteRatingDiff"]
    scatter_plot(
        x_series,
        y_series,
        "scatter_plot_of_white_elo_and_white_rating_diff.png",
        **kwargs
    )


def scatter_plot_of_black_elo_and_black_rating_diff(
    df: pd.DataFrame,
    **kwargs,
) -> None:
    x_series = df["BlackElo"]
    y_series = df["BlackRatingDiff"]
    scatter_plot(
        x_series,
        y_series,
        "scatter_plot_of_black_elo_and_black_rating_diff.png",
        **kwargs
    )


def table_plot_of_white_title_and_black_title(
    df: pd.DataFrame,
    **kwargs,
) -> None:
    # TODO: finish implementation
    df.fillna({"WhiteTitle": "-", "BlackTitle": "-"}, inplace=True)
    table = df.groupby(["WhiteTitle", "BlackTitle"]).count()
    print(table)


def create_graph(
    df: pd.DataFrame,
) -> nx.Graph:
    whites = df["White"].unique()
    blacks = df["Black"].unique()
    nodes = set(whites).union(blacks)
    edges = df.apply(lambda row: (row["White"], row["Black"]), axis=1).tolist()

    g = nx.Graph()
    g.add_nodes_from(nodes)

    for (node1, node2) in edges:
        if g.has_edge(node1, node2):
            g[node1][node2]["weight"] += 1
        else:
            g.add_edge(node1, node2, weight=1)

    return g


if __name__ == "__main__":
    # download_games_for_variant(
    #     "crazyhouse",
    #     "data/crazyhouse/rated_games",
    #     # "2017-04",
    #     # "2020-02",
    # )

    # decompress_and_save_bz2_files(
    #     "data/crazyhouse/rated_games",
    #     "data/crazyhouse/rated_games_decompressed",
    # )

    # date = "2016_01"

    # pgn = read_pgn_file(f"data/crazyhouse/rated_games_decompressed/{date}")
    # games = filter_pgn_file_for_games(pgn, verbose_period=1000)
    # df_games = make_df_from_games(games)
    # df_games.to_csv(f"{date}_games.csv", index=False)

    df = pd.read_csv("2016_01_games.csv")
    # scatter_plot_of_rating_difference_and_ply_count(df, alpha=0.5)
    # scatter_plot_of_white_elo_and_black_elo(df, alpha=0.5)
    # scatter_plot_of_white_elo_and_white_rating_diff(df, alpha=0.5)
    # scatter_plot_of_black_elo_and_black_rating_diff(df, alpha=0.5)
    # table_plot_of_white_title_and_black_title(df)

    create_graph(df)
