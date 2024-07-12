import argparse
import datetime
import sys

tax_brackets_2024 = [(0, 0.1), (11000, 0.12), (44725, 0.22),
                     (95375, 0.24), (182100, 0.32), (231250, 0.35),
                     (578125, 0.37)]
default_deduction_2024 = 13850


def get_federal_tax(brackets: list[tuple[int, float]], income: float) -> float:
    if income <= brackets[1][0]:
        return income * brackets[0][1]

    remaining = income
    total_taxes = 0
    for i in range(len(brackets)):
        if i == len(brackets) - 1:
            total_taxes += remaining * brackets[i][1]
            return total_taxes
        else:
            if brackets[i+1][0] > income:
                slice_tax = brackets[i][1] * remaining
                return total_taxes + slice_tax
            tax_slice = brackets[i + 1][0] - brackets[i][0]
            tax_slice_amount = brackets[i][1] * tax_slice
            remaining -= tax_slice
            total_taxes += tax_slice_amount


def project_retirement(capital: int, roc: float, inflation: float, expected_income: int,
                       years_left: int, print_table=False):
    months = years_left * 12
    monthly_roc = roc / 12.0
    monthly_payment = expected_income / 12.0
    monthly_inflation = inflation / 12.0

    if print_table:
        print("Month\tYear\tMOTY\tRemaining capital\tMonthly Withdrawal\tMonthly Income,Yearly Income,Taxes Paid")

    moy = datetime.date.today().month - 1
    yearly_income = 0
    taxes = 0

    for i in range(months):
        moy = (moy + 1) % 12
        monthly_payment *= (1.0 + monthly_inflation)
        capital = capital - (monthly_payment + taxes)
        if moy == 0 and yearly_income > 0:
            taxes = get_federal_tax(tax_brackets_2024, yearly_income - default_deduction_2024)
            yearly_income = 0
        else:
            taxes = 0
        monthly_income = capital * monthly_roc
        yearly_income += monthly_income
        capital += monthly_income
        if print_table:
            print(f"{i+1}\t{int(i/12)+1}\t{moy + 1}\t{capital:.2f}\t{monthly_payment:.2f}\t{monthly_income:.2f}\t"
                  f"{yearly_income:.2f}\t{taxes:.2f}")

    return capital


def parse_args():
    parser = argparse.ArgumentParser(description="Retirement calculator utility")
    parser.add_argument(
        "--roc",
        help="Return on capital rate",
        type=float,
        required=True
    )
    parser.add_argument(
        "--capital",
        help="Total capital",
        type=int,
        required=True
    )
    parser.add_argument(
        "--inflation",
        help="Yearly inflation rate",
        type=float,
        required=True
    )
    parser.add_argument(
        "--expenses",
        help="Expected yearly expenses",
        type=int,
        required=True
    )
    parser.add_argument(
        "--years",
        help="Number of years to project",
        type=int,
        required=True
    )
    parser.add_argument(
        "--print",
        help="Print full amortization table in CSV",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--solve-for",
        help="Solve for a specific variable",
        type=str,
        default=None
    )
    return parser.parse_args()


def binary_search(func, guess, reverse) -> float:
    epsilon = 1e-2
    max_it = 100
    top = guess
    low = 0
    for i in range(100):
        if not reverse and func(top) < 0:
            low = top
            top *= 2
        elif reverse and func(top) > 0:
            low = top
            top *= 2

    for i in range(max_it):
        v = func(top)
        print(f"Trying value: {top} resulted in {v} with top {top} and low {low}")
        if abs(v) < epsilon:
            return top
        search_range = top - low
        if v < 0:
            if not reverse:
                low = top
                top = top + search_range / 2
            else:
                top = low + search_range / 2
        else:
            if not reverse:
                top = low + search_range / 2
            else:
                low = top
                top = top + search_range / 2
    raise ValueError("Binary search didn't converge")


def solve_for(variable: str, capital: int | None, roc: float | None,
              inflation: float | None, expenses: int | None, years: int) -> float:
    if variable.lower() == 'capital':
        return binary_search(lambda x: project_retirement(x, roc, inflation, expenses, years),
                             capital, False)
    elif variable.lower() == 'roc':
        return binary_search(lambda x: project_retirement(capital, x, inflation, expenses, years),
                             roc, False)
    elif variable.lower() == 'inflation':
        return binary_search(lambda x: project_retirement(capital, roc, x, expenses, years),
                             inflation, True)
    elif variable.lower() == 'expenses':
        return binary_search(lambda x: project_retirement(capital, roc, inflation, x, years),
                             expenses, True)
    else:
        raise ValueError("Invalid argument: " + variable)


if __name__ == '__main__':
    args = parse_args()
    if not args.solve_for:
        final_capital = project_retirement(args.capital, args.roc, args.inflation,
                                           args.expenses, args.years, args.print)
        if not args.print:
            print(f"Final capital\t{final_capital:.2f}")
            sys.exit(0)
    else:
        result = solve_for(args.solve_for, args.capital, args.roc, args.inflation,
                           args.expenses, args.years)
        if result <= 1.0:
            result *= 100.0
            result = f"{result:.2f}%"
        else:
            result = f"{result:.2f}"

        print(f"Solving for variable '{args.solve_for}' returned {result}")

    sys.exit(0)
