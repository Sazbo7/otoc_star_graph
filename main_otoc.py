import sys
from star_graph import *
import pandas as pd

SAVE_ADDRESS = '/Users/szabo.48/Desktop/OTOC'

### Data saved using Pandas DataFrame to .pickle files ####
df_columns = ["L", "t_max", "t_init", "t_step", "Jzz", "Jx", "Jz", "Boundary", "Seed","Coupling", "OTOC_t"];
df = pd.DataFrame(columns=df_columns);


coupling_vals = np.linspace(0, 3, 5);
#length_vals = np.arange(4, 12, 2);
length_vals = [4, 8]; #slow above L = 12, try to run as few otocs as possible (low number of couplings and maybe not OTOC between all sites)

count = 0;
t_max, t_step, t_init, seed, periodic = 15, 400, 0, None, False

Jzz, Jz, Jx = -1.0, 0.45, 1.05

for L in length_vals:
    t_max, t_init, values = run_OTOC(Jz, Jz, Jx, coupling_vals, initial_state="Haar", t_max = t_max, t_step=t_step, t_init=t_init, seed=seed, L=L, periodic=periodic);

    df = df.append({"L": L,"t_max":, "t_init":t_init, "t_step":t_step, "Jzz":Jzz, "Jx":Jx, "Jz":Jz, "Boundary": periodic, "Seed": seed, "Coupling": coupling_vals, "OTOC_t":values}, ignore_index=True);
    print(count)
    count+=1;

df.to_pickle(SAVE_ADDRESS + "_" + str(L) +".pickle");
print('Completed');