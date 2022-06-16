import sys
from star_graph import *
import pandas as pd

SAVE_ADDRESS = "/Users/szabo.48/Desktop/entanglement_entropy"

### Data saved using Pandas DataFrame to .pickle files ####
df_columns = ["L", "t_max", "t_init", "t_step", "Jzz", "Jx", "Jz", "Boundary", "Seed","Coupling", "Ent_Ent"];
df = pd.DataFrame(columns=df_columns);


coupling_vals = np.linspace(0, 3, 13);
length_vals = np.arange(10, 18, 1);

count = 0;
t_max, t_step, t_init, seed, periodic = 20, 1000, 0, None, False

Jzz, Jz, Jx = -1.0, 0.45, 1.05

for L in length_vals:
    t_max, t_init, values = run_entanglement_entropy(Jzz, Jz, Jx, coupling_vals, initial_state="pol_y", t_max = t_max, t_step=t_step, t_init=t_init, seed=seed, L=L, periodic=periodic);

    df = df.append({"L": L,"t_max":, "t_init":t_init, "t_step":t_step, "Jzz":Jzz, "Jx":Jx, "Jz":Jz, "Boundary": periodic, "Seed": seed, "Coupling": coupling_vals, "Ent_Ent":values}, ignore_index=True);
    print(count)
    count+=1;

df.to_pickle(SAVE_ADDRESS + "_" + str(L) +".pickle");
print('Completed');