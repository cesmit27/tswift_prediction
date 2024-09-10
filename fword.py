import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def comma(value, tick_number):
    return f'{int(value):,}'

album_numbers = [1, 2, 3, 4]
album_names = ['Folklore', 'Evermore', 'Midnights', 'TDPD']
fword_counts = [3, 2, 7, 30]

linear_model = LinearRegression()
linear_model.fit(np.array(album_numbers).reshape(-1, 1), fword_counts)

params, params_covariance = curve_fit(exp_func, album_numbers, fword_counts, p0=[1, 0.1, 1])

next_album_numbers = [5, 6, 7, 8, 9]
next_album_names = [f"Future Album {i}" for i in range(1, 6)]
linear_predictions = linear_model.predict(np.array(next_album_numbers).reshape(-1, 1))
exp_predictions = exp_func(np.array(next_album_numbers), *params)

all_albums = album_names + next_album_names

plt.figure(figsize=(10, 6))
plt.plot(all_albums, fword_counts + list(exp_predictions), 'ro-', label='Exponential Prediction (The Good Ending)')
plt.plot(all_albums, fword_counts + list(linear_predictions), 'bo-', label='Linear Prediction')
plt.title('Number of F Words Predicted for Future Taylor Swift Albums')
plt.xlabel('Album')
plt.ylabel('Number of F Words')
plt.grid(True)
plt.legend()
plt.xticks(all_albums, rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(FuncFormatter(comma))
plt.savefig('Fword.png', bbox_inches='tight')
