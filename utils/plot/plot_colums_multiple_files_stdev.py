import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_multiple_file_column_avg_std(dir, fileregexp, y_column, ax, x_column, min_values=8, label="", color="black", linewidth=1.0, linestyle=":", marker=""):
    directory   = dir
    file_regexp = fileregexp
    files = glob.glob(directory+file_regexp)

    y_values_column = y_column
    x_values_column = x_column

    avgs = []  # Column-wise mean for each file
    stds = []  # Column-wise stdev for each file
    vals = []  # Error value list (at each step) for each file (nfiles, nsamples)
    x_values = []

    row_sizes = []
    for filename in files:

        values = np.loadtxt(filename)

        if len(values) <= 0:
            continue

        if len(x_values) <=0:
            x_values = values[:, x_values_column]

        ax.plot(values[:, x_values_column], values[:, y_values_column], alpha=0.0)

        avg = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        vals.append(values[:, y_values_column])
        avgs.append(avg)
        stds.append(std)
        row_sizes.append(len(values[:, y_values_column]))

    col_vals = []
    col_sizes = []
    for j in range( 0, max(row_sizes) ):
        col = []
        for i in range( 0, len(vals) ):
            if len(vals[i]) > j:
                col.append( vals[i][j] )
        col_vals.append(col)
        col_sizes.append( len(col) )

    col_avg = []
    col_std = []
    for c in col_vals:
        if len(c) > min_values:
            col_avg.append(np.mean(c))
            col_std.append(np.std(c))

    # x = np.arange( len(col_avg) )
    y = np.array(col_avg)
    z = np.array(col_std)
    y_max = y + z
    y_min = y - z

    # print("%2.1f $\pm$ %2.1f" % (col_avg[len(col_avg)/2]*100, col_std[len(col_std)/2]*100))

    ax.fill_between(x_values, y_max, y_min, facecolor=color, interpolate=True, alpha=0.0)
    ax.plot(x_values, y, label=label, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, markevery=3)
    ax.plot(x_values, y_max, color=color, alpha=0.0, linewidth=linewidth)
    ax.plot(x_values, y_min, color=color, alpha=0.0, linewidth=linewidth)

    plt.xlim( x_values[0],  x_values[-1])
    # plt.ylim( , np.log(np.max(y_max)) )
    plt.ylim( 0, np.max(y_max))

def get_multiple_file_column_avg(dir, fileregexp, col):
    files = glob.glob(dir+fileregexp)
    multi_file_vals = np.array([])
    for filename in files:
        values = np.loadtxt(filename)
        if len(values) <= 0:
            continue
        multi_file_vals = np.concatenate((multi_file_vals,values[:,col]))

    return np.mean(multi_file_vals), np.std(multi_file_vals)


if __name__ == "__main__":
    from itertools import cycle

    markers = [".", "v", "^", "<", ">", "o", "2",  "3", "4"]
    linecycler = cycle(markers)
    linestyle = "-"

    directory   = "../reaching_intent_estimation/results/"

    # 0 - error
    # 1 - time
    # 2 - percent
    # 3 - grid size
    # 4 - slack
    # 5 - Likelihood evals

    y_values_column = 0
    x_values_column = 2
    time_column = 1
    error_column = 0
    linewidth = 1.0

    savefile = directory+"comparison_nostdev.pdf"
    title = "Error vs. % of trajectory observed. Generative model: Surrogate"
    y_label = "Error(m)"
    x_label = "% observed"

    fig = plt.figure(figsize=(10, 8))
    # fig = plt.figure()
    ax = fig.add_subplot(111)

    # file_regexp = "*_results_grid0.005.dat"
    # time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    # plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 0.5cm. Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='red', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    # print ("%2.1f $\pm$ %2.1f" % (time*1000, time_stdev*1000))

    # file_regexp = "*_results_grid0.01.dat"
    # time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    # plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 1cm.   Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='red', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    # print ("%2.1f $\pm$ %2.1f" % (time * 1000, time_stdev * 1000))

    # file_regexp = "*_results_grid0.02.dat"
    # time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    # plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 2cm.   Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='blue', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    # print ("%2.1f $\pm$ %2.1f" % (time * 1000, time_stdev * 1000))

    # file_regexp = "*_results_grid0.05.dat"
    # time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    # plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 5cm.   Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='purple', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    # print ("%2.1f $\pm$ %2.1f" % (time * 1000, time_stdev * 1000))

    file_regexp = "*_results_grid0.1emu.dat"
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 10cm.  Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='black', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print ("%2.1f $\pm$ %2.1f" % (time * 1000, time_stdev * 1000))

    # file_regexp = "*_results_grid0.15.dat"
    # time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    # plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Grid size: 15cm.  Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='brown', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    # print ("%2.1f $\pm$ %2.1f" % (time * 1000, time_stdev * 1000))

    mode = "emu"
    # mode = "sim"
    grid = "0.005"

    # file_regexp = "*_results_pf%s%s.dat" % (grid,mode)
    file_regexp = "*_results_PF.dat"
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    error, error_stdev = get_multiple_file_column_avg(directory, file_regexp, error_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="Particle Filter.  Avg. Frame Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='blue', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print ("Time: %2.3f $\pm$ %2.3f" % (time * 1000, time_stdev * 1000))
    print("Error: %2.3f $\pm$ %2.3f" % (error, error_stdev))

    # file_regexp = "*_results_quadtree%s%s.dat" % (grid,mode)
    file_regexp = "*_results_quadtree%s.dat" % (grid)
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    error, error_stdev = get_multiple_file_column_avg(directory, file_regexp, error_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="2-BSPT (Ours).    Avg. Frame Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='green', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print ("Time: %2.3f $\pm$ %2.3f" % (time * 1000, time_stdev * 1000))
    print("Error: %2.3f $\pm$ %2.3f" % (error, error_stdev))

    file_regexp = "*results_abc-rejection%s%s.dat" % (grid,mode)
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    error, error_stdev = get_multiple_file_column_avg(directory, file_regexp, error_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="ABC-Reject.       Avg. Frame Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='red', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print("Time: %2.3f $\pm$ %2.3f" % (time * 1000, time_stdev * 1000))
    print("Error: %2.3f $\pm$ %2.3f" % (error, error_stdev))

    file_regexp = "*results_abc-smc%s%s.dat" % (grid,mode)
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    error, error_stdev = get_multiple_file_column_avg(directory, file_regexp, error_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="ABC-SMC.          Avg. Frame Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='purple', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print("Time: %2.3f $\pm$ %2.3f" % (time * 1000, time_stdev * 1000))
    print("Error: %2.3f $\pm$ %2.3f" % (error, error_stdev))

    file_regexp = "*results_mcmc%s%s.dat" % (grid,mode)
    time, time_stdev = get_multiple_file_column_avg(directory, file_regexp, time_column)
    error, error_stdev = get_multiple_file_column_avg(directory, file_regexp, error_column)
    plot_multiple_file_column_avg_std(directory,file_regexp,y_values_column,ax,x_values_column, label="MCMC-MH.          Avg. Frame Time:%3.3fms $\pm$%3.3fms" % (time*1000, time_stdev*1000), color='brown', linewidth=linewidth, linestyle=linestyle, marker=next(linecycler))
    print("Time: %2.3f $\pm$ %2.3f" % (time * 1000, time_stdev * 1000))
    print("Error: %2.3f $\pm$ %2.3f" % (error, error_stdev))

    lengend_font = matplotlib.font_manager.FontProperties(family="monospace", style=None, variant=None, weight=None, stretch=None, size=16, fname=None, _init=None)
    ax.legend(prop=lengend_font, loc='upper right')
    ax.set_title(title)
    ax.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # ax.set_yscale("log")

    plt.savefig(savefile, bbox_inches='tight')

    plt.show()

