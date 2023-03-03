import numpy as np

THRESHOLD = 0 / 100 # 1 percent of closing value

# takes in (n_bars, 2) array
# returns (n_bars, 3) array --> (bar_index, date, price, buy/sell/no action)
def strategy(bars):
    # bars = np.lib.pad(bars, ((0,0), (0, 1)), constant_values=(0)) # add one column to mark where to buy & sell
    closing = bars[:, 1]
    # long_positions = [(len(closing)-1, 0)]
    # previous_min = float('inf')
    # previous_max = 0

    for i in range(1, len(closing), 1):
        # closing[len(closing)-1] is earliest time
        # closing[0] is latest time

        # TODO: FIX THIS
        #   COME UP WITH PROPER ALGORITHM FOR DETERMINING MAXIMUM PROFIT


        dif = closing[i-1] - closing[i]
        threshold = closing[i] * THRESHOLD
        if dif > threshold:
            bars[i][2] = 1
        elif dif < threshold:
            bars[i][2] = -1
    
    # Second pass to remove duplicates
    markers_final = bars[:, 2].copy()
    for i in range(1, len(bars), 1):
        if bars[i][2] == bars[i-1][2]:
            markers_final[i-1] = 0
    # print(markers_final)

    bars[:, 2] = markers_final
    
    return bars