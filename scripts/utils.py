import matplotlib.pyplot as plt

def nice_double_plot(data1, data2, extent, title1='', title2='', xlabel='', ylabel='', cbar=False, cval_min=0,
                     cval_max=1):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    print(data1.min(), data1.max())
    if cbar:
        im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis', vmin=cval_min, vmax=cval_max)
        fig.colorbar(im1, ax=ax1)
    else:
        im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis')

    ax1.set_title(title1)

    if cbar:
        im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis', vmin=cval_min, vmax=cval_max)
        fig.colorbar(im2, ax=ax2)
    else:
        im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis')

    ax2.set_title(title2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.yaxis.labelpad = 40
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.subplots_adjust(right=0.98, top=0.95, bottom=0.07, left=0.12, hspace=0.05, wspace=0.20)