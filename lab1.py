import mypkg.my2DPlot as myplt
import numpy as np

# this code creates a sin graph
plt = myplt(lambda x : np.sin(x) + 1,0.,10.)
#this line sets the labels of the axes
plt.labels('x','y')
plt.addPlot(lambda x : np.cos(x) + 1)
# sets the line style of the first graph to dotted
plt.dotted()
plt.color('black')
plt.logy()
plt.logx()
# saves the graph to the local directory with the name 'figure.pdf'
plt.save('figure.pdf')
# tells python to actually display the graph in a window
plt.show()
