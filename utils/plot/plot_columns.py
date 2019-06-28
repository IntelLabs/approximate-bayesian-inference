import matplotlib.pyplot as plt
import pandas as pd


def plot_columns(filename, column_names, title="", subtitle=""):
    data = pd.read_table(filename, delimiter=" ")

    data.columns = column_names

    plt.rc('text', usetex=True)

    data.plot()

    plt.xlabel(subtitle)

    plt.title(title)

    plt.show(block=True)


title = "Evolution of the cost function L"

filename = "../training_evolution.txt"

subtitle = r"$[Fc(3,3) \rightarrow ReLu()] \times 5 \rightarrow Fc(3,180)$"

column_names = ['$L_i$', r'$\frac{N*k}{2} ln(2\pi)$', r'$-ln(|g_{\phi_2}(z_i)|)$', r'$(x-f_{\phi_1}(z_i))^2 g_{\phi_2}(z_i)$', r'$(x-f_{\phi_1}(z_i))$']

plot_columns(filename=filename, column_names=column_names, title=title, subtitle=subtitle)
