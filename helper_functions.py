
def windows_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)





def plot_graph(plt, start_time, variable):
    plt.savefig(windows_filename(start_time)+'_' + variable + '.pdf')







def ftoc(f):
    return (f - 32) * 5.0/9.0


def mphtoms(mph):
    return mph * 0.44704




def to_hour_of_year(hour, day, month):
    # count from 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    m = sum(days_in_month[:month - 1]) * 24
    d = (day - 1) * 24
    h = hour

    return m + d + h


def get_eval_hours(hour_start, hour_end, day_start, day_end, month_start,
                   month_end):
    #count from 1
    if (month_end > 12):
        month_end = 12
    if (day_end > 31):
        day_end = 31
    if (hour_end > 24):
        hour_end = 24

    cnt = 0
    hoursToEvaluate = []

    for m in range(12):  # 0-11
        for d in range(31):  #0-30
            for h in range(24):  # 0-23

                # Check if already gone through month

                if (m == 2 and d > 27):
                    continue
                elif ((m == 4 or m == 6 or m == 9 or m == 10) and d > 29):
                    continue

                # Fill list
                cnt += 1

                if (m >= month_start and m < month_end and d >= day_start
                        and d < day_end and h >= hour_start and h < hour_end):

                    hoursToEvaluate.append(cnt)

    return hoursToEvaluate

#len(get_eval_hours(0, 24, 0, 31, 0, 12))

