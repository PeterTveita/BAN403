from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


GROUPS = {
    "A": ["QAT", "ECU", "SEN", "NED"],
    "B": ["ENG", "IRN", "USA", "WAL"],
    "C": ["ARG", "KSA", "MEX", "POL"],
    "D": ["FRA", "AUS", "DEN", "TUN"],
    "E": ["ESP", "CRC", "GER", "JAP"],
    "F": ["BEL", "CAN", "MAR", "CRO"],
    "G": ["BRA", "SRB", "SUI", "CMR"],
    "H": ["POR", "GHA", "URU", "KOR"],
}

# 6 matches for each 4-team group.
GROUP_PAIRINGS = [(0, 1), (2, 3), (0, 2), (3, 1), (3, 0), (1, 2)]
TEAM_CODE_ALIASES = {"NED": "NET", "SUI": "SWI"}


@dataclass(frozen=True)
class TeamParams:
    alpha: float
    beta: float
    gamma: float
    delta: float


@dataclass
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int


def load_params(csv_path: Path) -> Dict[str, TeamParams]:
    params: Dict[str, TeamParams] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"team", "alpha", "beta", "gamma", "delta"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"CSV mangler kolonner: {missing}")

        for row in reader:
            team = row["team"]
            params[team] = TeamParams(
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                gamma=float(row["gamma"]),
                delta=float(row["delta"]),
            )
    for alias, base_code in TEAM_CODE_ALIASES.items():
        if alias not in params and base_code in params:
            params[alias] = params[base_code]
    return params


def poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate**k) / math.factorial(k)


def weighted_choice(cumulative_probs: List[float], rng: random.Random) -> int:
    u = rng.random()
    lo = 0
    hi = len(cumulative_probs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u <= cumulative_probs[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def match_distribution(
    home: str,
    away: str,
    params: Dict[str, TeamParams],
    cache: Dict[Tuple[str, str, bool], Tuple[List[float], List[int], List[int]]],
) -> Tuple[List[float], List[int], List[int]]:
    # Only matches involving Qatar apply home effect, with Qatar as home team.
    with_home_effect = home == "QAT"
    key = (home, away, with_home_effect)
    if key in cache:
        return cache[key]

    p_home = params[home]
    p_away = params[away]

    if with_home_effect:
        lam = p_home.alpha * p_away.beta * p_home.gamma
        mu = p_away.alpha * p_home.beta * p_home.delta
    else:
        lam = p_home.alpha * p_away.beta
        mu = p_away.alpha * p_home.beta

    probs: List[float] = []
    home_goals: List[int] = []
    away_goals: List[int] = []

    for x in range(6):
        px = poisson_pmf(x, lam)
        for y in range(6):
            py = poisson_pmf(y, mu)
            probs.append(px * py)
            home_goals.append(x)
            away_goals.append(y)

    total = sum(probs)
    probs = [p / total for p in probs]

    cumulative: List[float] = []
    running = 0.0
    for p in probs:
        running += p
        cumulative.append(running)
    cumulative[-1] = 1.0

    cache[key] = (cumulative, home_goals, away_goals)
    return cache[key]


def resolve_head_to_head(
    tied_teams: List[str], group_matches: List[MatchResult], rng: random.Random
) -> List[str]:
    h2h_points = {t: 0 for t in tied_teams}
    h2h_gf = {t: 0 for t in tied_teams}
    h2h_ga = {t: 0 for t in tied_teams}

    tied_set = set(tied_teams)
    for m in group_matches:
        if m.home not in tied_set or m.away not in tied_set:
            continue
        h2h_gf[m.home] += m.home_goals
        h2h_ga[m.home] += m.away_goals
        h2h_gf[m.away] += m.away_goals
        h2h_ga[m.away] += m.home_goals
        if m.home_goals > m.away_goals:
            h2h_points[m.home] += 3
        elif m.home_goals < m.away_goals:
            h2h_points[m.away] += 3
        else:
            h2h_points[m.home] += 1
            h2h_points[m.away] += 1

    h2h_gd = {t: h2h_gf[t] - h2h_ga[t] for t in tied_teams}
    sorted_teams = sorted(
        tied_teams,
        key=lambda t: (h2h_points[t], h2h_gd[t], h2h_gf[t]),
        reverse=True,
    )

    final: List[str] = []
    i = 0
    while i < len(sorted_teams):
        j = i + 1
        team_i = sorted_teams[i]
        key_i = (h2h_points[team_i], h2h_gd[team_i], h2h_gf[team_i])
        while j < len(sorted_teams):
            team_j = sorted_teams[j]
            key_j = (h2h_points[team_j], h2h_gd[team_j], h2h_gf[team_j])
            if key_j != key_i:
                break
            j += 1
        block = sorted_teams[i:j]
        if len(block) > 1:
            rng.shuffle(block)  # draw lots if still tied
        final.extend(block)
        i = j
    return final


def rank_group(group_teams: List[str], group_matches: List[MatchResult], rng: random.Random) -> List[str]:
    points = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    ga = {t: 0 for t in group_teams}

    for m in group_matches:
        gf[m.home] += m.home_goals
        ga[m.home] += m.away_goals
        gf[m.away] += m.away_goals
        ga[m.away] += m.home_goals
        if m.home_goals > m.away_goals:
            points[m.home] += 3
        elif m.home_goals < m.away_goals:
            points[m.away] += 3
        else:
            points[m.home] += 1
            points[m.away] += 1

    gd = {t: gf[t] - ga[t] for t in group_teams}
    ordered = sorted(group_teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)

    final_order: List[str] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        team_i = ordered[i]
        key_i = (points[team_i], gd[team_i], gf[team_i])
        while j < len(ordered):
            team_j = ordered[j]
            key_j = (points[team_j], gd[team_j], gf[team_j])
            if key_j != key_i:
                break
            j += 1
        block = ordered[i:j]
        if len(block) > 1:
            block = resolve_head_to_head(block, group_matches, rng)
        final_order.extend(block)
        i = j
    return final_order


def simulate_group_stage(params: Dict[str, TeamParams], n_iter: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    dist_cache: Dict[Tuple[str, str, bool], Tuple[List[float], List[int], List[int]]] = {}

    all_teams = [team for teams in GROUPS.values() for team in teams]
    qualifiers = {t: 0 for t in all_teams}
    winners = {t: 0 for t in all_teams}
    total_points = {t: 0.0 for t in all_teams}
    total_gd = {t: 0.0 for t in all_teams}

    for _ in range(n_iter):
        for _, teams in GROUPS.items():
            matches: List[MatchResult] = []
            points = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            ga = {t: 0 for t in teams}

            for a, b in GROUP_PAIRINGS:
                t1 = teams[a]
                t2 = teams[b]
                home, away = (t1, t2) if (t1 == "QAT" or t2 != "QAT") else (t2, t1)
                cumulative, hg, ag = match_distribution(home, away, params, dist_cache)
                draw_idx = weighted_choice(cumulative, rng)
                home_goals = hg[draw_idx]
                away_goals = ag[draw_idx]

                matches.append(MatchResult(home=home, away=away, home_goals=home_goals, away_goals=away_goals))

                gf[home] += home_goals
                ga[home] += away_goals
                gf[away] += away_goals
                ga[away] += home_goals
                if home_goals > away_goals:
                    points[home] += 3
                elif home_goals < away_goals:
                    points[away] += 3
                else:
                    points[home] += 1
                    points[away] += 1

            ranking = rank_group(teams, matches, rng)
            qualifiers[ranking[0]] += 1
            qualifiers[ranking[1]] += 1
            winners[ranking[0]] += 1

            for t in teams:
                total_points[t] += points[t]
                total_gd[t] += gf[t] - ga[t]

    rows: List[Dict[str, object]] = []
    for group_name, teams in GROUPS.items():
        for team in sorted(teams):
            rows.append(
                {
                    "group": group_name,
                    "team": team,
                    "qualify_prob": qualifiers[team] / n_iter,
                    "first_place_prob": winners[team] / n_iter,
                    "expected_points": total_points[team] / n_iter,
                    "expected_goal_diff": total_gd[team] / n_iter,
                }
            )
    return rows


def resolve_csv_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Fant ikke CSV: {path}")
        return path

    candidates = [
        Path("poisson_params_Quatar_2022.csv"),
        Path("poisson params Quatar 2022.csv"),
        Path(
            r"c:\Users\tveit\OneDrive - Norwegian School of Economics\V26\BAN403 - Simulation of Business Processes\Assignments\Assignment 1\poisson_params_Quatar_2022.csv"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Fant ikke parameterfilen. Oppgi sti med --csv.")


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "group",
        "team",
        "qualify_prob",
        "first_place_prob",
        "expected_points",
        "expected_goal_diff",
    ]
    widths = {h: len(h) for h in headers}

    formatted: List[Dict[str, str]] = []
    for row in rows:
        frow = {
            "group": str(row["group"]),
            "team": str(row["team"]),
            "qualify_prob": f"{float(row['qualify_prob']):.4f}",
            "first_place_prob": f"{float(row['first_place_prob']):.4f}",
            "expected_points": f"{float(row['expected_points']):.4f}",
            "expected_goal_diff": f"{float(row['expected_goal_diff']):.4f}",
        }
        formatted.append(frow)
        for h in headers:
            widths[h] = max(widths[h], len(frow[h]))

    print(" ".join(h.ljust(widths[h]) for h in headers))
    print(" ".join("-" * widths[h] for h in headers))
    for row in formatted:
        print(" ".join(row[h].ljust(widths[h]) for h in headers))


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "team",
                "qualify_prob",
                "first_place_prob",
                "expected_points",
                "expected_goal_diff",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo-simulering av VM 2022 gruppespill.")
    parser.add_argument("--csv", type=str, default=None, help="Sti til poisson-parameter CSV.")
    parser.add_argument("--iterations", type=int, default=10000, help="Antall simuleringer (minst 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default=None, help="Valgfri output-CSV.")
    args = parser.parse_args()

    if args.iterations < 10000:
        raise ValueError("Oppgaven krever minst 10000 iterasjoner.")

    csv_path = resolve_csv_path(args.csv)
    params = load_params(csv_path)
    rows = simulate_group_stage(params, n_iter=args.iterations, seed=args.seed)
    print_table(rows)

    if args.output:
        out = Path(args.output)
        write_csv(rows, out)
        print(f"\nLagret resultater til: {out.resolve()}")


if __name__ == "__main__":
    main()
