from soccer_predictor.markets import (
    both_teams_to_score,
    correct_score,
    double_chance,
    draw_no_bet,
    european_handicap_3way,
    fair_decimal_odds,
    match_result_and_over_under,
    over_under,
    remove_vig,
    score_matrix,
    three_way,
)


def print_market(title, probabilities):
    print(f"\n{title}")
    for outcome, probability in probabilities.items():
        print(f"  {outcome:>4}: {probability:6.2%} fair odds {fair_decimal_odds(probability):5.2f}")


def main():
    table = score_matrix(
        1.75,
        0.95,
        home_team="Portugal",
        away_team="Croatia",
    )

    print(f"{table.home_team} vs {table.away_team}")
    print(f"Matrix total probability: {table.total_probability:.6f}")

    print_market("3-Way", three_way(table))
    print_market("Double chance", double_chance(table))
    print_market("Over/Under 2.5", {k: v for k, v in over_under(table, 2.5).items() if k != "push"})
    print_market("Both teams to score", both_teams_to_score(table))
    print_market("Match Result & Over/Under 1.5", match_result_and_over_under(table, 1.5))
    print_market("European handicap 0:1", european_handicap_3way(table, away_goals_adjustment=1))

    dnb = draw_no_bet(table)
    print("\nDraw no bet")
    print(f"     1: win {dnb.home_win:6.2%}, push {dnb.draw_push:6.2%}, fair odds {dnb.home_fair_odds:5.2f}")
    print(f"     2: win {dnb.away_win:6.2%}, push {dnb.draw_push:6.2%}, fair odds {dnb.away_fair_odds:5.2f}")

    score_probs = correct_score(table)
    print("\nLikeliest correct scores")
    for score, probability in sorted(score_probs.items(), key=lambda item: item[1], reverse=True)[:8]:
        print(f"  {score:>4}: {probability:6.2%} fair odds {fair_decimal_odds(probability):5.2f}")

    bookmaker_odds = {"1": 1.67, "X": 3.70, "2": 5.50}
    print("\nBookmaker 3-way probabilities after margin removal")
    for outcome, probability in remove_vig(bookmaker_odds).items():
        print(f"  {outcome:>4}: {probability:6.2%}")


if __name__ == "__main__":
    main()

