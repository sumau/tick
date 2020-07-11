import numpy as np
from tick.hawkes import SimuHawkesExpKernels
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

run_time = 10
decay = 0.5
baseline = [0.2, 0.4]
adjacency = [[0.2, 0.1], [0, 0.3]]

hawkes = SimuHawkesExpKernels(baseline=baseline, decays=decay,
                              adjacency=adjacency, verbose=False, end_time=run_time,
                              seed=83)

hawkes.track_intensity(0.1)
hawkes.simulate()

print(hawkes.contribution_timestamps)
# [0.72350 5.95198 8.62946]

print(hawkes.contribution_nodes)
# [1 0 0]

print(np.array(hawkes.tracked_intensity))
# [[0.20000 0.25000 0.30366 0.32718]
#  [0.40000 0.55000 0.41098 0.40288]]

# [[                0.30366 0.32718]
#  [        0.55000                ]]

print(hawkes.contribution_intensities)
# [[0.40000 0.20000 0.20000]
#  [0.00000 0.00000 0.02622]
#  [0.00000 0.00366 0.00096]]

# intensity is different to tracked_intensity by amounts below because shows values after timestamps
#   +0.15    +0.1   +0.1

print(0.1*0.5*np.exp(-0.5*(5.95198 - 0.72350)))
# 0.0036611708675801376 (intensity has decreased from 0.05)
print(0.2*0.5*np.exp(-0.5*(8.62946 - 5.95198 )))
# 0.026217580206315472 (intensity has decreased from 0.1)

hawkes.reset()
hawkes.simulate()
print(hawkes.timestamps)


###############################################################################

import numpy as np
from tick.hawkes import SimuHawkesExpKernels
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from matplotlib import use
use('module://backend_interagg')
from sklearn.preprocessing import normalize

run_time = 1000
decays = 1/14
baseline = 0.0002
contagion = 0.11
n_nodes = 100
self_contagion = True

g = nx.barabasi_albert_graph(n_nodes, 1, seed=20)

self_contagion_matrix = np.eye(n_nodes) if self_contagion else np.zeros((n_nodes, n_nodes))
s = SimuHawkesExpKernels(adjacency=contagion * (nx.to_numpy_array(g) + self_contagion_matrix),
                                          decays=decays,
                                          baseline=baseline * np.ones(n_nodes),
                                          seed=2,
                                          end_time=run_time,
                                          verbose=True, )
s.track_intensity(run_time)
s.simulate()
all_intensities = np.transpose(s.contribution_intensities[1:])

###############################################################################
pos = nx.spring_layout(g)
nx.draw(g, pos=pos)
nx.draw_networkx_labels(g, pos=pos)
plt.savefig('nx_test.png')
plt.show()

G = nx.DiGraph()
for t, n, i in zip(s.contribution_timestamps,s.contribution_nodes,all_intensities):
    result = np.where(i > 0 )[0]
    for nj in result:
        G.add_edge(n,nj,timestamp=t)

# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
write_dot(G,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')
pos_cascade =graphviz_layout(G, prog='dot')
nx.draw(G, pos_cascade, with_labels=True, arrows=True)
plt.savefig('nx_test.png')
plt.show()

###############################################################################

upto = s.n_total_jumps
nodes_start = np.transpose(np.array(np.where(all_intensities[0:upto]>baseline)))
timestamps_end = s.contribution_timestamps[nodes_start[:,0]]
nodes_end = s.contribution_nodes[nodes_start[:,0]]

node_mapping= {}
i=0
for n in s.contribution_nodes:
    if n not in node_mapping.keys():
        node_mapping[n]=i
        i += 1
node_mapping_reverse = {value: key for key, value in node_mapping.items()}

fig, ax = plt.subplots()
ax.plot(np.vectorize(node_mapping.get)(s.contribution_nodes)[0:upto], s.contribution_timestamps[0:upto], '.',color='black',alpha=0.3)
for n_start,t_end,n_end in zip(nodes_start,timestamps_end,nodes_end):
    t_start_indexes = np.where((s.contribution_timestamps<t_end)&(s.contribution_nodes==n_start[1]))
    t_start = max(s.contribution_timestamps[t_start_indexes])
    ax.plot([node_mapping[n_start[1]],node_mapping[n_end]],[t_start,t_end],color='b',alpha=0.2)
ax.invert_yaxis()
plt.show()

# n = 34
# n = 9
n=8
print(node_mapping_reverse[n])
print(s.timestamps[node_mapping_reverse[n]])
print(np.where(s.contribution_nodes==node_mapping_reverse[n]))
i = 10
print(np.where(s.contribution_intensities[:,i]>0)[0]-1)
print(s.contribution_intensities[:,i][np.where(s.contribution_intensities[:,i]>0)])
# [0.0002     0.00046541 0.00097954]

###############################################################################

fig, ax = plt.subplots()

ax.plot(s.contribution_timestamps[0:upto],
        np.vectorize(node_mapping.get)(s.contribution_nodes)[0:upto],
        '.',color='black',alpha=0.3)
ax.invert_yaxis()
plt.show()