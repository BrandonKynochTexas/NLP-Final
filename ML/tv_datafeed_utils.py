import string
import re
from tvDatafeed import Interval as interval

"""
class Interval(enum.Enum):
    in_1_minute = "1"
    in_3_minute = "3"
    in_5_minute = "5"
    in_15_minute = "15"
    in_30_minute = "30"
    in_45_minute = "45"
    in_1_hour = "1H"
    in_2_hour = "2H"
    in_3_hour = "3H"
    in_4_hour = "4H"
    in_daily = "1D"
    in_weekly = "1W"
    in_monthly = "1M"
"""
def bars_per_day(interval):
    re_result = re.findall(r'\d+', interval)
    if len(re_result) != 1:
        return 0
    int_val = int(re_result[0])
    if interval.__contains__("H"):
        return 390 / (int_val * 60)
    elif interval.__contains__("D"):
        return 390 / (int_val * 390)
    elif interval.__contains__("W"):
        return 390 / (int_val * 1950)
    elif interval.__contains__("M"):
        return 390 / (int_val * 7800)

    if 0 < int_val < 60:
        return 390 / int_val
    else:
        print("Invalid time interval")


def __print_unit_test(interval_text, predicate):
    result = 'PASS' if predicate else 'FAIL'
    print(f'{interval_text}: {result}')

if __name__=="__main__":
    print("Unit tests for utils\n")
    __print_unit_test('Invalid input 1', bars_per_day('a89s2zS') == 0)
    __print_unit_test('Invalid input 2', bars_per_day('asdf') == 0)
    print("")
    __print_unit_test('1 minutes', bars_per_day(interval.in_1_minute.value) == 390)
    __print_unit_test('15 minutes', bars_per_day(interval.in_15_minute.value) == 26)
    __print_unit_test('30 minutes', bars_per_day(interval.in_30_minute.value) == 13)
    print("")
    __print_unit_test('4 minutes', bars_per_day("4") == 97.5)
    __print_unit_test('52 minutes', bars_per_day("52") == 7.5)
    print("")
    __print_unit_test('1 hour', bars_per_day(interval.in_1_hour.value) == 6.5)
    __print_unit_test('4 hour', bars_per_day("4H") == 1.625)
    print("")
    __print_unit_test('1 day', bars_per_day(interval.in_daily.value) == 1)
    __print_unit_test('4 day', bars_per_day("4D") == 0.25)
    print("")
    __print_unit_test('1 week', bars_per_day(interval.in_weekly.value) == 0.2)
    __print_unit_test('2 week', bars_per_day("2W") == 0.1)
    print("")
    __print_unit_test('1 month', bars_per_day(interval.in_monthly.value) == 0.05)
    __print_unit_test('2 month', bars_per_day("2M") == 0.025)
    
